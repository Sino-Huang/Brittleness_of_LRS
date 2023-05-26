import logging
import pickle
import random
import time

import hydra
import torch
from einops import repeat
from omegaconf import DictConfig, OmegaConf
from torch import multiprocessing
from torch.multiprocessing import Pipe
from torch.distributions.categorical import Categorical
import cv2

import gym
import numpy as np
import os
from hydra.utils import HydraConfig, get_original_cwd

from custom_utils import save_to_video
from envs import AtariEnvironmentFalsePositiveSim, MR_ATARI_ACTION
from model import CnnActorCriticNetwork
from lang_rew_module import create_lang_rew_network
import PySimpleGUI as sg
from pathlib import Path


def imgtoByte(img, is_rbg=False):
    # return imBytes
    img = cv2.resize(img, (256, 256))
    if is_rbg:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    imBytes = cv2.imencode('.png', img)[1].tobytes()
    return imBytes

def get_action(model, device, state):
    state = torch.Tensor(state).to(device)
    action_probs, value_ext, value_int = model(state)
    action_dist = Categorical(action_probs)
    action = action_dist.sample()
    return action.data.cpu().numpy().squeeze()

def bag_of_actions(action_seq, num_class = 8):
    action_frequncy_vector = np.zeros((num_class, ), dtype=np.float32)
    for i in action_seq:
        action_frequncy_vector[i] += 1

    # normalise
    action_frequncy_vector /= np.sum(action_frequncy_vector)
    return action_frequncy_vector

log = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path="../config", config_name="rnd_rl_train_test_false_positive")
def main(cfg: DictConfig):
    # print config
    cfg_raw = OmegaConf.to_container(cfg, resolve=True)  # OmegaConf is used for wandb
    cfg = DictConfig(cfg_raw)
    log.info("--------Model Config---------")
    log.info(OmegaConf.to_yaml(cfg))
    log.info("-----End of Model Config-----")
    jobname = HydraConfig.get().job.name

    # set up random seed
    np.random.seed(cfg.CONSTANT.RANDOM_SEED)
    torch.manual_seed(cfg.CONSTANT.RANDOM_SEED)
    random.seed(cfg.CONSTANT.RANDOM_SEED)

    device = torch.device(cfg.device_info.device)

    env = gym.make(cfg.CONSTANT.ENV_NAME)

    input_size = env.observation_space.shape  # 4 (210, 160, 3) not useful
    # output_size = env.action_space.n  # 2
    output_size = cfg.CONSTANT.N_ACTIONS  # 2

    if 'Breakout' in cfg.CONSTANT.ENV_NAME:
        output_size -= 1

    env.close()

    is_render = True

    num_worker = 1
    sticky_action = False

    # --- load lang rew model -----
    mean = np.asarray(eval(cfg.CONSTANT.IMG_MEAN), dtype=np.float32)
    std = np.asarray(eval(cfg.CONSTANT.IMG_STD), dtype=np.float32)
    mean = repeat(mean, 'd -> d w h', w=cfg.CONSTANT.LW_RESIZE, h=cfg.CONSTANT.LW_RESIZE)
    std = repeat(std, 'd -> d w h', w=cfg.CONSTANT.LW_RESIZE, h=cfg.CONSTANT.LW_RESIZE)

    walkthrough_data_arr, walkthrough_raw_sentence, walkthrough_embds, task_saved_state, task_id = create_lang_rew_network(cfg)

    # ---- end Load rew model ----
    parent_conn, child_conn = Pipe()
    parent_render_conn, child_render_conn =Pipe()

    work = AtariEnvironmentFalsePositiveSim(
        cfg.CONSTANT.ENV_NAME,
        is_render,
        0,
        child_conn,
        child_render_conn,
        walkthrough_data_arr,
        w=cfg.CONSTANT.RL_RESIZE,
        h=cfg.CONSTANT.RL_RESIZE,
        sticky_action=sticky_action,
        p=cfg.rl_params.sticky_action_prob,
        max_episode_steps=128000,
        walkthrough_raw_sentence=walkthrough_raw_sentence,
        walkthrough_embds=walkthrough_embds,
        max_decay_coef=cfg.rl_params.max_decay_coef,
        img_mean=mean,
        img_std=std,
        task_saved_state= task_saved_state,
        task_id = task_id,
        follow_temporal_order=cfg.rl_params.follow_temporal_order,
        more_restrict_flag=cfg.rl_params.more_restrict_flag,

    )
    work.start()

    #states = np.zeros([num_worker, 4, 84, 84])
    states = torch.zeros(num_worker, 4, 84, 84)


    sg.theme('DarkAmber')
    layout = [
        [sg.Frame("Lang Reward Info", [
            [sg.Text('Sentence'.ljust(100)), sg.Text('Imme Aux Reward'.ljust(30)),
             sg.Text('Sentence Activation'.ljust(30)), sg.Text('Accu Lang Rew'.ljust(30)),
             sg.Text('max Lang softmax'.ljust(30))],  # title
            *[[sg.Text(f"{sent_ind + 1}: {walkthrough_raw_sentence[sent_ind].ljust(100)}"),
               sg.Text('', key=f'l{sent_ind}'), sg.Text('', key=f'i{sent_ind}'), sg.Text(f'', key=f'a{sent_ind}'),
               sg.Text('', key=f'm{sent_ind}')] for sent_ind in
              range(len(walkthrough_raw_sentence))]
        ])],
        [sg.Push(), sg.vtop(sg.Text('Action:'.ljust(30), key='action_info')), sg.Frame("Game Content", [
            [sg.Image(key="obs_image", size=(256, 256))],  # use OpenCV later to import image
        ]), sg.Push()],
        [sg.Text("", key='n_step_info')],
        [sg.Text("", key='room_number_info')],
        [sg.Button("Screenshot current room"), sg.Button("Start Recording"), sg.Button("Record Location"), sg.Button("End Recording")],

        [sg.Frame("Other info", [
            [sg.Text('', key=f'other info')]
        ])]

    ]
    window = sg.Window('Evaluate MontezumasRevenge', layout, return_keyboard_events=True, use_default_focus=False)
    custom_n_step = 0
    record_frames = []
    record_avatar_location = [] # fixed stride size 4 x 4 = 16, but not every stride has a state constraint. tolerance +- 4
    record_actions_frequency = [] # fixed stride size 4 x 4 = 16, for every stride we upload action frequency vector
    action_freq_arr = [] # for each time get the action data
    action_in_episode = [] # store the action in the whole episode
    record_count = 1
    record_flags = False
    stride_sixteen_ind = 0
    record_location_flag = False
    while True:
        event, value = window.read()
        if event in (sg.WIN_CLOSED, 'Cancel'):
            break
        agent_act_flag = False
        # movement event
        if event == 'space:65':
            parent_conn.send(list(MR_ATARI_ACTION).index(MR_ATARI_ACTION.FIRE))
            window['action_info'].update(f'Action: {MR_ATARI_ACTION.FIRE.name}'.ljust(30))
            action_value = MR_ATARI_ACTION.FIRE.value
            action_freq_arr.append(list(MR_ATARI_ACTION).index(MR_ATARI_ACTION.FIRE))
            action_in_episode.append(MR_ATARI_ACTION.FIRE.name)
            agent_act_flag = True
        elif event == 'Left:113':
            parent_conn.send(list(MR_ATARI_ACTION).index(MR_ATARI_ACTION.LEFT))
            window['action_info'].update(f'Action: {MR_ATARI_ACTION.LEFT.name}'.ljust(30))
            action_value = MR_ATARI_ACTION.LEFT.value
            action_freq_arr.append(list(MR_ATARI_ACTION).index(MR_ATARI_ACTION.LEFT))
            action_in_episode.append(MR_ATARI_ACTION.LEFT.name)
            agent_act_flag = True
        elif event == 'Right:114':
            parent_conn.send(list(MR_ATARI_ACTION).index(MR_ATARI_ACTION.RIGHT))
            window['action_info'].update(f'Action: {MR_ATARI_ACTION.RIGHT.name}'.ljust(30))
            action_value = MR_ATARI_ACTION.RIGHT.value
            action_freq_arr.append(list(MR_ATARI_ACTION).index(MR_ATARI_ACTION.RIGHT))
            action_in_episode.append(MR_ATARI_ACTION.RIGHT.name)
            agent_act_flag = True
        elif event == 'Down:116':
            parent_conn.send(list(MR_ATARI_ACTION).index(MR_ATARI_ACTION.DOWN))
            window['action_info'].update(f'Action: {MR_ATARI_ACTION.DOWN.name}'.ljust(30))
            action_value = MR_ATARI_ACTION.DOWN.value
            action_freq_arr.append(list(MR_ATARI_ACTION).index(MR_ATARI_ACTION.DOWN))
            action_in_episode.append(MR_ATARI_ACTION.DOWN.name)
            agent_act_flag = True
        elif event == 'Up:111':
            parent_conn.send(list(MR_ATARI_ACTION).index(MR_ATARI_ACTION.UP))
            window['action_info'].update(f'Action: {MR_ATARI_ACTION.UP.name}'.ljust(30))
            action_value = MR_ATARI_ACTION.UP.value
            action_freq_arr.append(list(MR_ATARI_ACTION).index(MR_ATARI_ACTION.UP))
            action_in_episode.append(MR_ATARI_ACTION.UP.name)
            agent_act_flag = True
        elif event == 'z:52':
            parent_conn.send(list(MR_ATARI_ACTION).index(MR_ATARI_ACTION.LEFTFIRE))
            window['action_info'].update(f'Action: {MR_ATARI_ACTION.LEFTFIRE.name}'.ljust(30))
            action_value = MR_ATARI_ACTION.LEFTFIRE.value
            action_freq_arr.append(list(MR_ATARI_ACTION).index(MR_ATARI_ACTION.LEFTFIRE))
            action_in_episode.append(MR_ATARI_ACTION.LEFTFIRE.name)
            agent_act_flag = True
        elif event == 'x:53':
            parent_conn.send(list(MR_ATARI_ACTION).index(MR_ATARI_ACTION.RIGHTFIRE))
            window['action_info'].update(f'Action: {MR_ATARI_ACTION.RIGHTFIRE.name}'.ljust(30))
            action_value = MR_ATARI_ACTION.RIGHTFIRE.value
            action_freq_arr.append(list(MR_ATARI_ACTION).index(MR_ATARI_ACTION.RIGHTFIRE))
            action_in_episode.append(MR_ATARI_ACTION.RIGHTFIRE.name)
            agent_act_flag = True
        elif event == 'c:54':
            parent_conn.send(list(MR_ATARI_ACTION).index(MR_ATARI_ACTION.NOOP))
            window['action_info'].update(f'Action: {MR_ATARI_ACTION.NOOP.name}'.ljust(30))
            action_value = MR_ATARI_ACTION.NOOP.value
            action_freq_arr.append(list(MR_ATARI_ACTION).index(MR_ATARI_ACTION.NOOP))
            action_in_episode.append(MR_ATARI_ACTION.NOOP.name)
            agent_act_flag = True
        elif event == "Screenshot current room":
            print("Screenshot")
            folderpath = os.path.join(get_original_cwd(), f"mr_room_samples/{window['room_number_info'].get().split(' ')[-1]}")
            folderpath = Path(folderpath)
            folderpath.mkdir(exist_ok=True, parents=True)
            files = next(os.walk(folderpath))[2]
            len_files = len(files)
            png_name = f"{len_files}.png"
            png_path = os.path.join(folderpath, png_name)
            print("path", png_path)
            with open(png_path, 'wb') as f:
                f.write(imBytes)
        elif event == "Start Recording":
            print("Start Recording")
            record_flags = True
            action_freq_arr = []

        elif event == "Record Location":
            record_location_flag = True

        elif event == "End Recording":
            print("End Recording")
            record_flags = False
            output_path = os.path.join(get_original_cwd(), f"walkthrough_sentence_{record_count}.mp4")
            save_to_video(record_frames, output_path, fps=6.25, img_size=256)
            with open(os.path.join(get_original_cwd(), f"constraints_info_walkthrough_sentence_{record_count}.pkl") ,'wb') as f:
                data_dict = dict()
                data_dict['action_freq'] = record_actions_frequency
                data_dict['avatar_location'] = record_avatar_location
                pickle.dump(data_dict, f)

            with open(os.path.join(get_original_cwd(), "action_overall.txt"), 'wt') as f:
                for ind, act in enumerate(action_in_episode):
                    f.writelines(f'{ind}: {act}\n')

            record_frames = []
            record_actions_frequency = []
            record_avatar_location = []
            action_freq_arr = []
            record_count += 1

        if agent_act_flag:
            # update image
            render_img = parent_render_conn.recv()

            imBytes = imgtoByte(render_img)
            window['obs_image'].update(data=imBytes)

            window['n_step_info'].update(f'n_step: {custom_n_step}')
            custom_n_step += 1


            # update info
            next_state, reward, done, real_done, log_rewar, eval_info = parent_conn.recv()
            if walkthrough_data_arr is not None:
                no_ofwins, no_achieve_state_constraints, no_achieve_action_constraints, walkthrough_accu_rew, n_steps, immediate_lang_reward, obs_seq_memory, current_room, accu_rew, agent_pos, walkthrough_activation_flag = eval_info
                window['room_number_info'].update(f'current_room: {current_room}, agent pos: {agent_pos}, record len: {len(record_avatar_location), len(record_actions_frequency)}')


                # update reward
                for ind in range(len(walkthrough_raw_sentence)):
                    # immediate reward
                    window[f'l{ind}'].update(
                        f'{np.round(immediate_lang_reward[ind], 6) if immediate_lang_reward[ind] is not None else None}'.ljust(
                            30))

                    # activation flag
                    window[f'i{ind}'].update(
                        f'{walkthrough_activation_flag[ind]}'.ljust(
                            30))

                # length
                window[f'other info'].update(
                    # f'{round(float(atari_env.accu_lang_reward[accu_rew_ind]), 3) if atari_env.accu_lang_reward is not None else None}')
                    f'obs_len: {len(obs_seq_memory)}\naccumulated reward: {accu_rew}'.ljust(70))

            if record_flags:
                stride_sixteen_ind = (stride_sixteen_ind + 1) % 4
                if stride_sixteen_ind % 4 == 0: # the default env has sticky action for 4 frames, and since we set stride to be 8, so, we record for every two steps
                    record_frames.append(render_img[:, :, ::-1])
                    # construct the action frequency value
                    action_freq_arr = bag_of_actions(action_freq_arr, num_class=8)
                    record_actions_frequency.append(action_freq_arr)
                    action_freq_arr = []
                    if record_location_flag:
                        print("Record Avatar Position")
                        record_avatar_location.append(agent_pos)
                    else:
                        record_avatar_location.append(None)
                    record_location_flag = False

    window.close()

if __name__ == '__main__':
    multiprocessing.set_start_method('spawn')
    main()
