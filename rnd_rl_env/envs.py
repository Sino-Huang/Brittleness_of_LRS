from enum import IntEnum

import cv2
import gym

import numpy as np

from collections import deque
from copy import copy

from einops import rearrange, repeat
from gym.spaces import Discrete
from torch.multiprocessing import Pipe, Process
from multiprocessing.dummy import DummyProcess
import torch

from PIL import Image

def bag_of_actions_goyal(action_seq, num_class = 18):
    action_frequncy_vector = np.zeros((num_class, ), dtype=np.float64)
    for i in action_seq:
        action_frequncy_vector[i] += 1

    # normalise
    action_frequncy_vector /= np.sum(action_frequncy_vector)
    return action_frequncy_vector

class MR_ATARI_ACTION(IntEnum):
    NOOP = 0
    FIRE = 1
    UP = 2
    RIGHT = 3
    LEFT = 4
    DOWN = 5
    RIGHTFIRE = 11
    LEFTFIRE = 12

class MR_ActionsWrapper(gym.ActionWrapper):
    def __init__(self, env):
        gym.ActionWrapper.__init__(self, env)
        self.action_space = Discrete(len(MR_ATARI_ACTION))
        self.action_choice_list = list(MR_ATARI_ACTION)

    def action(self, act):
        return self.action_choice_list[act].value



class MaxAndSkipEnv(gym.Wrapper):
    def __init__(self, env, skip=4):
        """Return only every `skip`-th frame"""
        gym.Wrapper.__init__(self, env)
        # most recent raw observations (for max pooling across time steps)
        self._obs_buffer = np.zeros((2,) + env.observation_space.shape, dtype=np.uint8)
        self._skip = skip

    def step(self, action):
        """Repeat action, sum reward, and max over last observations."""
        total_reward = 0.0
        done = None
        for i in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            if i == self._skip - 2: self._obs_buffer[0] = obs
            if i == self._skip - 1: self._obs_buffer[1] = obs
            total_reward += reward
            if done:
                break
        # Note that the observation on the done=True frame
        # doesn't matter
        max_frame = self._obs_buffer.max(axis=0)

        return max_frame, total_reward, done, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)
        

class MontezumaInfoWrapper(gym.Wrapper):
    def __init__(self, env, room_address):
        gym.Wrapper.__init__(self, env)
        self.room_address = room_address
        self.visited_rooms = set()
        self.lives = 0
        self.current_room = 0

    def _unwrap(self, env):
        if hasattr(env, "unwrapped"):
            return env.unwrapped
        elif hasattr(env, "env"):
            return self._unwrap(env.env)
        elif hasattr(env, "leg_env"):
            return self._unwrap(env.leg_env)
        else:
            return env

    def get_current_room(self):
        ram = self._unwrap(self.env).ale.getRAM()
        assert len(ram) == 128
        return int(ram[self.room_address])

    def get_current_lives(self):
        lives = self._unwrap(self.env).ale.lives()
        return lives

    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        self.visited_rooms.add(self.get_current_room())
        self.lives = self.get_current_lives()
        self.current_room = self.get_current_room()

        if 'episode' not in info:
            info['episode'] = {}
        info['episode'].update(visited_rooms=copy(self.visited_rooms), lives=self.lives)

        if done:
            self.visited_rooms.clear()
        return obs, rew, done, info

    def reset(self):
        return self.env.reset()


class AtariEnvironment(Process):
    def __init__(
            self,
            env_name,
            is_render,
            env_idx,
            child_conn,
            child_render_conn,
            lang_rew_net, # activate if it is not none
            history_size=4,
            h=84,
            w=84,
            sticky_action=True,
            p=0.25,
            max_episode_steps=18000,
            lang_rew_memory_length = None,
            lang_rew_window_size = None,
            walkthrough_raw_sentence = None,
            walkthrough_embds = None, # need to be ndarray type
            max_decay_coef = None,
            lang_accu_rew_maximum = None,
            minimum_memory_len = None,
            lang_softmax_threshold_current = None,
            lang_softmax_threshold_old = None,
            img_mean =None, # numpy ndarray shape (3 w h)
            img_std = None,
            fixed_lang_reward = None,
            task_saved_state = None,
            task_id = None,

    ):
        # super().__init__()
        super(AtariEnvironment, self).__init__()
        self.daemon = True # if set True, main process end will cause subprocess end
        render_mode = 'human' if is_render else 'rgb_array'
        self.env_name = env_name
        self.is_render = is_render
        self.env_idx = env_idx
        self.steps = 0
        self.episode = 0
        self.no_of_wins = 0
        self.rall = 0
        self.recent_rlist = deque(maxlen=100) # recent reward list
        self.child_conn = child_conn
        self.child_render_conn = child_render_conn
        self.lang_rew_net = lang_rew_net

        self.sticky_action = sticky_action
        self.last_action = 0
        self.p = p
        self.max_episode_steps = max_episode_steps

        self.history_size = history_size
        self.history = np.zeros([history_size, h, w])
        self.h = h
        self.w = w
        if self.lang_rew_net is None:
            self.lang_rew_memory_length = lang_rew_memory_length
            self.lang_rew_window_size = lang_rew_window_size
        else:
            self.lang_rew_memory_length = lang_rew_memory_length//2 if self.lang_rew_net.is_binary_cls else lang_rew_memory_length
            self.lang_rew_window_size = lang_rew_window_size
        self.walkthrough_raw_sentence = walkthrough_raw_sentence
        self.walkthrough_embds = walkthrough_embds
        self.max_decay_coef = max_decay_coef
        self.lang_accu_rew_maximum = lang_accu_rew_maximum
        self.minimum_memory_len = minimum_memory_len
        self.lang_rew_act_count = int(minimum_memory_len /3) # 6
        self.lang_softmax_threshold_current = lang_softmax_threshold_current
        self.lang_softmax_threshold_old = lang_softmax_threshold_old
        self.img_mean = img_mean
        self.img_std = img_std # numpy ndarray shape (3 w h)
        self.fixed_lang_reward = fixed_lang_reward
        self.task_saved_state = task_saved_state
        self.task_id = task_id

    def reset_sentence_memory(self, sentence_id):
        self.potentials_list[sentence_id] = []
        self.obs_seq_memory = []
        self.action_seq_memory = []
        self._window_size_count = self.lang_rew_window_size
        self._lang_rew_act_count = self.lang_rew_act_count
        self.immediate_lang_reward[sentence_id] = None
        self.accu_lang_reward[sentence_id] = 0.0
        self.max_softmax_reward[sentence_id] = 0.0

    def run(self):
        super().run()
        # init the environment inside run method
        self.env = MaxAndSkipEnv(gym.make(
            self.env_name))  # the inside env will execute step first then the outer wrapper will recieve result from step()
        if 'Montezuma' in self.env_name:
            self.env = MontezumaInfoWrapper(self.env, room_address=3 if 'Montezuma' in self.env_name else 1)
            self.env = MR_ActionsWrapper(self.env)
        self.reset() # reset inside run method due to respawn
        while True:
            action = self.child_conn.recv()

            if 'Breakout' in self.env_name:
                action += 1

            # sticky action
            if self.sticky_action:
                if np.random.rand() <= self.p:
                    action = self.last_action
                self.last_action = action
            prev_lives = self.env.lives

            s, reward, done, info = self.env.step(action)

            cur_lives = self.env.lives
            if cur_lives != prev_lives:
                # if lose lives, skip the waiting

                for _ in range(15):
                    s, reward, done, info = self.env.step(0) # 0 is NOOP

                # # also reset the lang rew
                # if self.lang_rew_net is not None:
                #     self.reset_lang_rew_stuffs()

            if self.is_render:
                # send observation back to parent process
                self.child_render_conn.send(s[:, :, ::-1]) # this is GBR version now

            # conditions that the game is done
            if self.max_episode_steps < self.steps: # exceed max steps
                done = True
            if self.env.current_room != self.task_room:
                done = True

            # consider different win condition
            if self.task_id == 2 and reward >= 300:
                done = True
                self.no_of_wins += 1
            elif reward >= 1000:
                done = True
                self.no_of_wins += 1

            log_reward = reward # log_reward will have no language reward

            # ------------- Lang Rew Steps -------------------
            if self.lang_rew_net is not None:
                self._window_size_count -= 1
                self._lang_rew_act_count -= 1
                if self._window_size_count <= 0 :
                    self._window_size_count = self.lang_rew_window_size
                    # update memory
                    obs = cv2.resize(s, (self.w, self.h), interpolation=cv2.INTER_AREA)
                    obs = rearrange(obs, "w h c -> c w h")
                    # feed previous memory directly
                    if len(self.obs_seq_memory) == 0:
                        # add previous memory
                        for i in range(self.minimum_memory_len):
                            self.obs_seq_memory.append(obs)
                            self.action_seq_memory.append(0)

                    self.obs_seq_memory.append(obs)  # 3, w, h
                    self.action_seq_memory.append(list(MR_ATARI_ACTION)[action].value)

                # Calculate Lang Rew
                lang_rew_total_imme = 0.0
                # update image
                # resize, permutation s shape: w, h, 3

                # ----------------- want to calcualte lang rew
                if len(self.obs_seq_memory) >= self.minimum_memory_len and self._lang_rew_act_count <= 0:  # give enough observations
                    self._lang_rew_act_count= self.lang_rew_act_count

                    prev_score = 1.0
                    for each_ind in range(len(self.walkthrough_raw_sentence)):
                        if self.accu_lang_reward[each_ind] < self.lang_accu_rew_maximum:  # calculate only when it is activated, does not exceed maximum, update memory only when we reach the window size
                            # language reward update for every window size
                            imme_lang_rew = self.compute_language_reward(each_ind) # we also need to check the logits reward
                            self.immediate_lang_reward[each_ind] = imme_lang_rew
                            lang_rew_total_imme += imme_lang_rew + self.max_softmax_reward[np.max([each_ind -1, 0])] * imme_lang_rew # current + current * previous this address the temporal order

                            if self.accu_lang_reward[each_ind] is not None:
                                self.accu_lang_reward[each_ind] += self.immediate_lang_reward[each_ind]
                            else:
                                self.accu_lang_reward[each_ind] = self.immediate_lang_reward[each_ind]
                                # also check if activate next sentence, activate when logit reach lang_softmax_threshold_current
                                # deactivate old task when reaching lang_softmax_threshold_old

                            #  --- Decay Max Softmax ----
                            for each_sentence_ind in range(len(self.walkthrough_raw_sentence)):
                                self.max_softmax_reward[each_ind] = self.max_softmax_reward[
                                                                        each_ind] * self.max_decay_coef
                                # self.accu_lang_reward[each_ind] = self.accu_lang_reward[each_ind] * self.max_decay_coef
                            # --- End Decay Max Softmax ---
                        # limit the memory buffer
                if len(self.obs_seq_memory) >= self.lang_rew_memory_length:
                    # pop the oldest element
                    self.obs_seq_memory.pop(0)
                    self.action_seq_memory.pop(0)

                reward += lang_rew_total_imme


            # ------------- End of Lang Rew Steps ------------

            force_done = done

            self.history[:-1, :, :] = self.history[1:, :, :] # this is kind of push...
            self.history[-1, :, :] = self.pre_proc(s)

            self.rall += reward # self.rall reward until now
            self.steps += 1


            if self.lang_rew_net is not None:
                self.child_conn.send(
                    [self.history[:, :, :], reward, force_done, done, log_reward,
                     [
                        self.no_of_wins, self.steps, self.potentials_list, self.immediate_lang_reward, self.accu_lang_reward,
                         self.max_softmax_reward, self.obs_seq_memory, self.env.current_room, self.rall
                    ]
                     ])
            else:
                self.child_conn.send(
                    [self.history[:, :, :], reward, force_done, done, log_reward,
                     [self.no_of_wins]
                     ])

            if done:
                self.recent_rlist.append(self.rall)
                # if 'Montezuma' in self.env_name:
                #     print("[Episode {}({})] Step: {}  Reward: {}  Recent Reward: {}  Visited Room: [{}] No. of wins: {}".format(
                #         self.episode, self.env_idx, self.steps, self.rall, np.mean(self.recent_rlist),
                #         info.get('episode', {}).get('visited_rooms', {}), self.no_of_wins))
                #     print('env visited room 5', self.env.visited_rooms)
                #
                # else:
                #     print("[Episode {}({})] Step: {}  Reward: {}  Recent Reward: {}".format(
                #         self.episode, self.env_idx, self.steps, self.rall, np.mean(self.recent_rlist)))

                self.history = self.reset()

    def reset_lang_rew_stuffs(self):
        # memory for reward shaping module
        self.potentials_list = [[] for _ in range(
            len(self.walkthrough_raw_sentence))]  # store the previous lang reward potential and thus can conduct subtraction operation
        self.obs_seq_memory = []
        self.action_seq_memory = []
        self._window_size_count = self.lang_rew_window_size
        self._lang_rew_act_count = self.lang_rew_act_count

        # lang reward
        self.immediate_lang_reward = [None for _ in
                                      range(
                                          len(self.walkthrough_raw_sentence))]  # length = num of instruction sentence
        self.accu_lang_reward = [0.0 for _ in
                                 range(len(self.walkthrough_raw_sentence))]  # sum of all the lang reward

        self.max_softmax_reward = [0.0 for _ in
                                   range(
                                       len(self.walkthrough_raw_sentence))]



    def reset(self):
        self.env.reset()
        self.env.ale.restoreSystemState(self.task_saved_state)
        self.env.visited_rooms.clear()

        s, *_ = self.env.step(0)

        self.last_action = 0
        self.steps = 0
        self.episode += 1
        self.rall = 0
        self.task_room = self.env.current_room

        self.get_init_state(
            self.pre_proc(s))

        # ----- Set Up Lang Rew Stuffs -------
        if self.lang_rew_net is not None:
            self.reset_lang_rew_stuffs()

        return self.history[:, :, :]

    def compute_language_reward(self, sentence_ind):
        # langs, lang_lens, images
        # image shape: torch.Size([Numofactivated, 15, 3, 84, 84])
        # image are normalised by ImageNet mean and std
        # lang shape: torch.Size([Numofactivated, 768]) for sent emb
        # lang_len shape: torch.Size([1]) for language length.

        # similarly, we need to preprocess the image vector to fit the lang module input spec
        # in the future we will integrate this preprocessing part
        lang_input = np.asarray(self.walkthrough_embds[sentence_ind], dtype=np.float16)
        lang_input = lang_input[None, :]
        lang_input = torch.asarray(lang_input, dtype=torch.float16).to(self.lang_rew_net.device, non_blocking=True)

        if self.lang_rew_net.is_goyal:
            action_input = np.asarray(bag_of_actions_goyal(self.action_seq_memory), dtype=np.float16)
            action_input = action_input[None, :]
            action_input = torch.asarray(action_input, dtype=torch.float16).to(self.lang_rew_net.device, non_blocking=True)
            with torch.inference_mode():
                logit = self.lang_rew_net([lang_input, action_input])  # shape [1, 2]
        else:
            img_input = np.asarray(self.obs_seq_memory, dtype=np.float16)
            # add batch dim
            img_input = img_input[None,:]
            B, L = img_input.shape[:2]
            mean = repeat(self.img_mean, 'c w h -> b l c w h', b=B, l =L)
            std = repeat(self.img_std, 'c w h -> b l c w h', b=B, l =L)
            # let them do normalisation
            img_input = (img_input - mean) / std
            img_input = torch.asarray(img_input, dtype=torch.float16).to(self.lang_rew_net.device, non_blocking=True)
            with torch.inference_mode():
                logit = self.lang_rew_net([lang_input, img_input])  # shape [1, 2]
        if self.lang_rew_net.is_binary_cls:
            # let them return the logits_norm softmax
            logit_norm = torch.softmax(logit, dim=-1)
            logit_norm = logit_norm[0][1].detach().cpu().numpy()
            self.potentials_list[sentence_ind].append(logit_norm)

        else:
            logit_norm = logit[0] # 0 start point, 1 duration
            logit_norm = logit_norm.detach().cpu().numpy()
            self.potentials_list[sentence_ind].append(logit_norm)
            logit_norm = logit_norm[1]
            # logit_norm = logit[:,0][0]
            # logit_norm = (logit_norm/2.0 + 0.9).detach().cpu().numpy() # normalise with offset 0.4

        # # ---- in Goyal's method, imme_lang_rew = rew_t - rew_t-1 ---
        # if len(self.potentials_list[sentence_ind]) > 1:
        #     lang_result = np.max([np.max([self.potentials_list[sentence_ind][-1] - 0.5, 0.0]) - np.max([self.potentials_list[sentence_ind][-2] - 0.5, 0.0]), 0.0])
        # else:
        #     lang_result = np.max([self.potentials_list[sentence_ind][-1] - 0.5, 0.0])
        # # ---- end of Goyal's  method -------

        # ---- our proposed method: imme_lang_rew = rew_t - max_rew_in_the_past ----

        if self.lang_rew_net.is_binary_cls:
            if sentence_ind > 1 and np.sum(self.max_softmax_reward[
                                           sentence_ind - 2:sentence_ind]) < self.lang_softmax_threshold_current:  # do not activate the later sentence
                lang_result = 0.0
            elif np.mean(self.potentials_list[sentence_ind][-3:]) > self.lang_softmax_threshold_current: # get lang reward only when we continuously get the reward 2 times
                lang_result = np.max([self.potentials_list[sentence_ind][-1] - self.lang_softmax_threshold_current, 0.0]) - np.max(
                    [self.max_softmax_reward[sentence_ind] - self.lang_softmax_threshold_current, 0.0])
                if lang_result > 0.0: # round for stabilisation because some Critic Model might find it hard
                    if self.fixed_lang_reward: # if fixed_lang is true, we only a fixed lang reward and then the sentence will be deactivated
                        lang_result = self.fixed_lang_reward
                    else:
                        lang_result = np.round(lang_result, 2)
                else:
                    lang_result = 0.0
            else:
                lang_result = 0.0
        else:
            temp = np.asarray(self.potentials_list[sentence_ind][-2:])
            if sentence_ind > 1 and np.sum(self.max_softmax_reward[
                                           sentence_ind - 2:sentence_ind]) < self.lang_softmax_threshold_old:  # do not activate the later sentence
                lang_result = 0.0
            elif np.mean(temp[:,-1]) > self.lang_softmax_threshold_old: # get lang reward only when we continuously get the reward twice
                lang_result = np.max([self.potentials_list[sentence_ind][-1][-1] - self.lang_softmax_threshold_old, 0.0]) - np.max(
                    [self.max_softmax_reward[sentence_ind] - self.lang_softmax_threshold_old, 0.0])
                if lang_result > 0.0: # round for stabilisation because some Critic Model might find it hard
                    if self.fixed_lang_reward: # if fixed_lang is true, we only a fixed lang reward and then the sentence will be deactivated
                        lang_result = self.fixed_lang_reward
                    else:
                        lang_result = np.round(lang_result, 2)
                else:
                    lang_result = 0.0
            else:
                lang_result = 0.0
        # ---- end of our proposed method ------

        if lang_result > 0.0 and logit_norm > self.max_softmax_reward[sentence_ind]:
            self.max_softmax_reward[sentence_ind] =  logit_norm

        return lang_result


    def pre_proc(self, X):
        #X = np.array(Image.fromarray(X).convert('L')).astype('float32')
        #x = cv2.resize(X, (self.h, self.w))
        #return x
        frame = Image.fromarray(X).convert('L') # convert to gray scale
        frame = np.array(frame.resize((self.h, self.w)))
        return frame.astype(np.float32)

    def get_init_state(self, s):
        for i in range(self.history_size):
            self.history[i, :, :] = self.pre_proc(s)

