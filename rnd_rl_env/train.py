import logging
import random
from pathlib import Path

import hydra
import numpy as np
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import wandb
from einops import repeat
from omegaconf import DictConfig, OmegaConf
from hydra.utils import HydraConfig, get_original_cwd
from torch import multiprocessing
from torch.distributions.categorical import Categorical
from torch.multiprocessing import Pipe
from tqdm import tqdm

from model import CnnActorCriticNetwork, RNDModel
from envs import *
from rnd_rl_env.lang_rew_module import create_lang_rew_network
from utils import RunningMeanStd, RewardForwardFilter

def get_action(model, device, state):
    state = torch.Tensor(state).to(device)
    state = state.float()
    action_probs, value_ext, value_int = model(state)
    action_dist = Categorical(action_probs)
    action = action_dist.sample()

    return action.data.cpu().numpy().squeeze(), value_ext.data.cpu().numpy().squeeze(), value_int.data.cpu().numpy().squeeze(), action_probs.detach().cpu()


def compute_intrinsic_reward(rnd, device, next_obs):
    next_obs = torch.FloatTensor(next_obs).to(device)

    target_next_feature = rnd.target(next_obs)
    predict_next_feature = rnd.predictor(next_obs)
    intrinsic_reward = (target_next_feature - predict_next_feature).pow(2).sum(1) / 2

    return intrinsic_reward.data.cpu().numpy()


def make_train_data(reward, done, value, gamma, gae_lambda, num_step, num_worker, use_gae):
    discounted_return = np.empty([num_worker, num_step])

    # Discounted Return
    if use_gae:
        gae = np.zeros_like([num_worker, ])
        for t in range(num_step - 1, -1, -1):
            delta = reward[:, t] + gamma * value[:, t + 1] * (1 - done[:, t]) - value[:, t]
            gae = delta + gamma * gae_lambda * (1 - done[:, t]) * gae

            discounted_return[:, t] = gae + value[:, t]

            # For Actor
        adv = discounted_return - value[:, :-1]

    else:
        running_add = value[:, -1]
        for t in range(num_step - 1, -1, -1):
            running_add = reward[:, t] + gamma * running_add * (1 - done[:, t])
            discounted_return[:, t] = running_add

        # For Actor
        adv = discounted_return - value[:, :-1]

    return discounted_return.reshape([-1]), adv.reshape([-1])


def train_model(cfg, device, output_size, model, rnd, optimizer, s_batch, target_ext_batch, target_int_batch, y_batch, adv_batch, next_obs_batch, old_action_probs):
    epoch = 3
    update_proportion = 0.25
    s_batch = torch.FloatTensor(s_batch).to(device)
    target_ext_batch = torch.FloatTensor(target_ext_batch).to(device)
    target_int_batch = torch.FloatTensor(target_int_batch).to(device)
    y_batch = torch.LongTensor(y_batch).to(device)
    adv_batch = torch.FloatTensor(adv_batch).to(device)
    next_obs_batch = torch.FloatTensor(next_obs_batch).to(device)

    sample_range = np.arange(len(s_batch))
    forward_mse = nn.MSELoss(reduction='none')

    with torch.no_grad():
        action_probs_old_list = torch.stack(old_action_probs).permute(1, 0, 2).contiguous().view(-1, output_size).to(device)

        m_old = Categorical(action_probs_old_list)
        log_prob_old = m_old.log_prob(y_batch)
        # ------------------------------------------------------------

    for i in range(epoch):
        np.random.shuffle(sample_range)
        for j in range(int(len(s_batch) / cfg.rl_params.batch_size)):
            sample_idx = sample_range[cfg.rl_params.batch_size * j:cfg.rl_params.batch_size * (j + 1)]

            # --------------------------------------------------------------------------------
            # for Curiosity-driven(Random Network Distillation)
            predict_next_state_feature, target_next_state_feature = rnd(next_obs_batch[sample_idx])

            forward_loss = forward_mse(predict_next_state_feature, target_next_state_feature.detach()).mean(-1)
            # Proportion of exp used for predictor update
            mask = torch.rand(len(forward_loss)).to(device)
            mask = (mask < update_proportion).type(torch.FloatTensor).to(device)
            forward_loss = (forward_loss * mask).sum() / torch.max(mask.sum(), torch.Tensor([1]).to(device))
            # ---------------------------------------------------------------------------------

            action_probs, value_ext, value_int = model(s_batch[sample_idx])
            m = Categorical(action_probs)
            log_prob = m.log_prob(y_batch[sample_idx])

            ratio = torch.exp(log_prob - log_prob_old[sample_idx])

            surr1 = ratio * adv_batch[sample_idx]
            surr2 = torch.clamp(
                ratio,
                1.0 - cfg.rl_params.eps,
                1.0 + cfg.rl_params.eps) * adv_batch[sample_idx]

            actor_loss = -torch.min(surr1, surr2).mean()
            critic_ext_loss = F.mse_loss(value_ext.sum(1), target_ext_batch[sample_idx])
            critic_int_loss = F.mse_loss(value_int.sum(1), target_int_batch[sample_idx])

            critic_loss = critic_ext_loss + critic_int_loss

            entropy = m.entropy().mean()

            optimizer.zero_grad()
            loss = actor_loss + 0.5 * critic_loss - cfg.rl_params.entropy_coef * entropy + forward_loss
            loss.backward()
            optimizer.step()


def wandb_init(cfg: DictConfig, jobname, tagslist, notes=None):
    wandb.init(
        project='Atari Montezuma RL',
        name=jobname,
        notes=notes,
        tags=tagslist,
        config=cfg
    )


log = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path="../config", config_name="rnd_rl_train")
def main(cfg: DictConfig):
    # print config
    cfg_raw = OmegaConf.to_container(cfg, resolve=True)  # OmegaConf is used for wandb
    cfg = DictConfig(cfg_raw)
    log.info("--------Model Config---------")
    log.info(OmegaConf.to_yaml(cfg))
    log.info("-----End of Model Config-----")
    jobname = HydraConfig.get().job.name
    if cfg.data_files.use_wandb:
        wandb_init(cfg_raw, jobname, ["rl_Nov", f'task_id-{cfg.rl_data_files.rl_task}'])

    # set up random seed
    np.random.seed(cfg.CONSTANT.RANDOM_SEED)
    torch.manual_seed(cfg.CONSTANT.RANDOM_SEED)
    random.seed(cfg.CONSTANT.RANDOM_SEED)

    device = torch.device(cfg.device_info.device)

    env = gym.make(cfg.CONSTANT.ENV_NAME)

    input_size = env.observation_space.shape  # 4
    # output_size = env.action_space.n  # 2
    output_size = cfg.CONSTANT.N_ACTIONS  # 2

    if 'Breakout' in cfg.CONSTANT.ENV_NAME:
        output_size -= 1

    env.close()

    is_render = False
    if not os.path.exists(cfg.rl_params.save_dir):
        os.makedirs(cfg.rl_params.save_dir)
    model_path = os.path.join(get_original_cwd(), cfg.rl_params.save_dir, f"task_id-{cfg.rl_data_files.rl_task}-"+cfg.CONSTANT.ENV_NAME + '.model')
    predictor_path = os.path.join(get_original_cwd(), cfg.rl_params.save_dir, f"task_id-{cfg.rl_data_files.rl_task}-"+cfg.CONSTANT.ENV_NAME + '.pred')
    target_path = os.path.join(get_original_cwd(), cfg.rl_params.save_dir, f"task_id-{cfg.rl_data_files.rl_task}-"+cfg.CONSTANT.ENV_NAME + '.target')


    reward_rms = RunningMeanStd()
    obs_rms = RunningMeanStd(shape=(1, 1, 84, 84))
    discounted_reward = RewardForwardFilter(cfg.rl_params.ext_gamma)

    # ---- setup RL model -----
    model = CnnActorCriticNetwork(input_size, output_size, cfg.rl_params.use_noisy_net)
    rnd = RNDModel(input_size, output_size)
    model = model.to(device)
    rnd = rnd.to(device)
    optimizer = optim.Adam(list(model.parameters()) + list(rnd.predictor.parameters()), lr=cfg.rl_params.lr)
   
    if cfg.rl_params.load_model:
        model.load_state_dict(torch.load(model_path))
    # --- End setup RL model --------

    # --- load lang rew model -----
    mean = np.asarray(eval(cfg.CONSTANT.IMG_MEAN), dtype=np.float32)
    std = np.asarray(eval(cfg.CONSTANT.IMG_STD), dtype=np.float32)
    mean = repeat(mean, 'd -> d w h', w=cfg.CONSTANT.LW_RESIZE, h=cfg.CONSTANT.LW_RESIZE)
    std = repeat(std, 'd -> d w h', w=cfg.CONSTANT.LW_RESIZE, h=cfg.CONSTANT.LW_RESIZE)

    lang_network, walkthrough_raw_sentence, walkthrough_embds, task_saved_state, task_id = create_lang_rew_network(cfg)
    # ---- end Load rew model ----

    works = []
    parent_conns = []
    child_conns = []

    for idx in range(cfg.rl_params.num_worker):
        parent_conn, child_conn = Pipe()

        work = AtariEnvironment(
            cfg.CONSTANT.ENV_NAME,
            is_render,
            idx,
            child_conn,
            None,
            lang_network,
            w=cfg.CONSTANT.RL_RESIZE,
            h=cfg.CONSTANT.RL_RESIZE,
            sticky_action=cfg.rl_params.sticky_action,
            p=cfg.rl_params.sticky_action_prob,
            max_episode_steps=cfg.rl_params.max_episode_steps,
            lang_rew_memory_length=cfg.rl_params.memory_length,
            lang_rew_window_size=cfg.rl_params.window_size,
            walkthrough_raw_sentence=walkthrough_raw_sentence,
            walkthrough_embds=walkthrough_embds,
            max_decay_coef=cfg.rl_params.max_decay_coef,
            lang_accu_rew_maximum=cfg.rl_params.lang_accu_rew_maximum,
            minimum_memory_len=cfg.rl_params.minimum_memory_len,
            lang_softmax_threshold_current=cfg.rl_params.lang_softmax_threshold_current,
            lang_softmax_threshold_old=cfg.rl_params.lang_softmax_threshold_old,
            img_mean=mean,
            img_std=std,
            fixed_lang_reward=cfg.rl_params.fixed_lang_reward,
            task_saved_state=task_saved_state,
            task_id = task_id
        )

        work.start()
        works.append(work)
        parent_conns.append(parent_conn)
        child_conns.append(child_conn)

    states = np.zeros([cfg.rl_params.num_worker, 4, 84, 84])


    sample_env_index = 0   # Sample Environment index to log fixed, as 0
    sample_episode = 0
    sample_rall = 0
    sample_step = 0
    sample_i_rall = 0
    global_update = 0
    global_step = 0
    no_of_win = 0


    # normalize observation
    print('Initializes observation normalization...')
    next_obs = []
    for step in tqdm(range(cfg.rl_params.num_step * cfg.rl_params.pre_obs_norm_steps)):
        actions = np.random.randint(0, output_size, size=(cfg.rl_params.num_worker,))

        for parent_conn, action in zip(parent_conns, actions):
            parent_conn.send(action)

        for parent_conn in parent_conns:
            next_state, reward, done, realdone, log_reward, _ = parent_conn.recv()
            next_obs.append(next_state[3, :, :].reshape([1, 84, 84]))

        if len(next_obs) % (cfg.rl_params.num_step * cfg.rl_params.num_worker) == 0:
            next_obs = np.stack(next_obs)
            obs_rms.update(next_obs)
            next_obs = []

    print('Training...')
    while True:
        total_state, total_reward, total_done, total_next_state, total_action, total_int_reward, total_next_obs, total_ext_values, total_int_values, total_action_probs = [], [], [], [], [], [], [], [], [], []
        global_step += (cfg.rl_params.num_worker * cfg.rl_params.num_step)
        global_update += 1

        # Step 1. n-step rollout
        for _ in range(cfg.rl_params.num_step):
            actions, value_ext, value_int, action_probs = get_action(model, device, np.float32(states) / 255.)

            for parent_conn, action in zip(parent_conns, actions):
                parent_conn.send(action)

            next_states, rewards, dones, real_dones, log_rewards, next_obs, no_of_wins_arr = [], [], [], [], [], [], []
            for parent_conn in parent_conns:
                next_state, reward, done, real_done, log_reward, otherinfo = parent_conn.recv()
                no_ofwins = otherinfo[0]
                next_states.append(next_state)
                rewards.append(reward)
                dones.append(done)
                real_dones.append(real_done)
                log_rewards.append(log_reward)
                no_of_wins_arr.append(no_ofwins)
                next_obs.append(next_state[3, :, :].reshape([1, 84, 84]))

            next_states = np.stack(next_states)
            rewards = np.hstack(rewards)
            dones = np.hstack(dones)
            real_dones = np.hstack(real_dones)
            next_obs = np.stack(next_obs)
            no_of_wins_arr = np.hstack(no_of_wins_arr)

            # total reward = int reward + ext Reward
            intrinsic_reward = compute_intrinsic_reward(rnd, device,
                ((next_obs - obs_rms.mean) / np.sqrt(obs_rms.var)).clip(-5, 5))
            intrinsic_reward = np.hstack(intrinsic_reward)
            sample_i_rall += intrinsic_reward[sample_env_index]

            total_next_obs.append(next_obs)
            total_int_reward.append(intrinsic_reward)
            total_state.append(states)
            total_reward.append(rewards)
            total_done.append(dones)
            total_action.append(actions)
            total_ext_values.append(value_ext)
            total_int_values.append(value_int)
            total_action_probs.append(action_probs)

            states = next_states[:, :, :, :]

            sample_rall += log_rewards[sample_env_index]


            sample_step += 1
            if real_dones[sample_env_index]:
                sample_episode += 1
                no_of_win = no_of_wins_arr[sample_env_index]
                # print(f'sample_episode is {sample_episode}')
                if cfg.data_files.use_wandb:
                    wandb.log(
                        {
                            'sample_reward_per_episode': sample_rall,
                            'sample_step_per_episode': sample_step,
                            "sample_episode": sample_episode,
                            "no_of_win": no_of_win,
                        }
                    )

                sample_rall = 0
                sample_step = 0
                sample_i_rall = 0

        # calculate last next value
        _, value_ext, value_int, _ = get_action(model, device, np.float32(states) / 255.)
        total_ext_values.append(value_ext)
        total_int_values.append(value_int)
        # --------------------------------------------------

        total_state = np.stack(total_state).transpose([1, 0, 2, 3, 4]).reshape([-1, 4, 84, 84])
        total_reward = np.stack(total_reward).transpose().clip(-1, 1)
        total_action = np.stack(total_action).transpose().reshape([-1])
        total_done = np.stack(total_done).transpose()
        total_next_obs = np.stack(total_next_obs).transpose([1, 0, 2, 3, 4]).reshape([-1, 1, 84, 84])
        total_ext_values = np.stack(total_ext_values).transpose()
        total_int_values = np.stack(total_int_values).transpose()
        total_logging_action_probs = np.vstack(total_action_probs)

        # Step 2. calculate intrinsic reward
        # running mean intrinsic reward
        total_int_reward = np.stack(total_int_reward).transpose()
        total_reward_per_env = np.array([discounted_reward.update(reward_per_step) for reward_per_step in total_int_reward.T])
        mean, std, count = np.mean(total_reward_per_env), np.std(total_reward_per_env), len(total_reward_per_env)
        reward_rms.update_from_moments(mean, std ** 2, count)

        # normalize intrinsic reward
        total_int_reward /= np.sqrt(reward_rms.var)
        if cfg.data_files.use_wandb:
            wandb.log(
                {
                    'max_prob_per_global_update': total_logging_action_probs.max(1).mean(),
                    'int_reward_per_global_update': np.sum(total_int_reward) / cfg.rl_params.num_worker,
                    'global_step': global_step,
                    'global_update': global_update,
                },
            )

        # -------------------------------------------------------------------------------------------

        # logging Max action probability

        # Step 3. make target and advantage
        # extrinsic reward calculate
        ext_target, ext_adv = make_train_data(total_reward,
                                              total_done,
                                              total_ext_values,
                                              cfg.rl_params.ext_gamma,
                                              cfg.rl_params.gae_lambda,
                                              cfg.rl_params.num_step,
                                              cfg.rl_params.num_worker,
                                              cfg.rl_params.use_gae)

        # intrinsic reward calculate
        # None Episodic
        int_target, int_adv = make_train_data(total_int_reward,
                                              np.zeros_like(total_int_reward),
                                              total_int_values,
                                              cfg.rl_params.int_gamma,
                                              cfg.rl_params.gae_lambda,
                                              cfg.rl_params.num_step,
                                              cfg.rl_params.num_worker,
                                              cfg.rl_params.use_gae)

        # add ext adv and int adv
        total_adv = int_adv * cfg.rl_params.int_coef + ext_adv * cfg.rl_params.ext_coef
        # -----------------------------------------------

        # Step 4. update obs normalize param
        obs_rms.update(total_next_obs)
        # -----------------------------------------------

        # Step 5. Training!
        train_model(cfg, device, output_size, model, rnd, optimizer,
                        np.float32(total_state) / 255., ext_target, int_target, total_action,
                        total_adv, ((total_next_obs - obs_rms.mean) / np.sqrt(obs_rms.var)).clip(-5, 5),
                        total_action_probs)

        if global_step % (cfg.rl_params.num_worker * cfg.rl_params.num_step * cfg.rl_params.save_interval) == 0:
            print('Now Global Step :{}'.format(global_step))
            Path(os.path.join(get_original_cwd(), cfg.rl_params.save_dir)).mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), model_path)
            torch.save(rnd.predictor.state_dict(), predictor_path)
            torch.save(rnd.target.state_dict(), target_path)

        if cfg.rl_params.want_lang_rew:
            if global_step >= 4000000:
                # we stop it
                break
            if no_of_win >= 1500:
                break
        else:
            if global_step >= 4000000:
                break
            if no_of_win >= 1500:
                break


    # finish wandb automatically for multirun
    if cfg.data_files.use_wandb:
        wandb.finish()

if __name__ == '__main__':
    multiprocessing.set_start_method('spawn')
    main()
