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

def bag_of_actions(action_seq, num_class = 8):
    action_frequncy_vector = np.zeros((num_class, ), dtype=np.float32)
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
        self.agent_pos = (0,0)

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

    def get_agent_pos(self):
        x, y = self._unwrap(self.env).ale.getRAM()[42:44]
        return int(x), int(y)

    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        self.visited_rooms.add(self.get_current_room())
        self.lives = self.get_current_lives()
        self.current_room = self.get_current_room()
        self.agent_pos = self.get_agent_pos()

        if 'episode' not in info:
            info['episode'] = {}
        info['episode'].update(visited_rooms=copy(self.visited_rooms), lives=self.lives)

        if done:
            self.visited_rooms.clear()
        return obs, rew, done, info

    def reset(self):
        return self.env.reset()


class AtariEnvironmentFalsePositiveSim(Process):
    def __init__(
            self,
            env_name,
            is_render,
            env_idx,
            child_conn,
            child_render_conn,
            walkthrough_data_arr, # activate if it is not none
            history_size=4,
            h=84,
            w=84,
            sticky_action=True,
            p=0.25,
            max_episode_steps=18000,
            walkthrough_raw_sentence = None,
            walkthrough_embds = None, # need to be ndarray type
            max_decay_coef = None,
            img_mean =None, # numpy ndarray shape (3 w h)
            img_std = None,
            task_saved_state = None,
            task_id = None,
            follow_temporal_order = None,
            more_restrict_flag = None

    ):
        # super().__init__()
        super(AtariEnvironmentFalsePositiveSim, self).__init__()
        self.daemon = True # if set True, main process end will cause subprocess end
        render_mode = 'human' if is_render else 'rgb_array'
        self.env_name = env_name
        self.is_render = is_render
        self.env_idx = env_idx
        self.steps = 0
        self.episode = 0
        # ------ log stuffs ------
        self.no_of_wins = 0
        self.no_achieve_action_constraints = 0
        self.no_achieve_state_constraints = 0
        self.walkthrough_accu_rew = [0.0 for i in range(len(walkthrough_data_arr))]
        # ------ end log stuffs ------

        self.rall = 0
        self.recent_rlist = deque(maxlen=100) # recent reward list
        self.child_conn = child_conn
        self.child_render_conn = child_render_conn
        self.walkthrough_data_arr = walkthrough_data_arr

        self.sticky_action = sticky_action
        self.last_action = 0
        self.p = p
        self.max_episode_steps = max_episode_steps

        self.history_size = history_size
        self.history = np.zeros([history_size, h, w])
        self.h = h
        self.w = w

        self.walkthrough_raw_sentence = walkthrough_raw_sentence
        self.walkthrough_embds = walkthrough_embds
        self.max_decay_coef = max_decay_coef
        self.img_mean = img_mean
        self.img_std = img_std # numpy ndarray shape (3 w h)
        self.task_saved_state = task_saved_state
        self.task_id = task_id

        self.stride_count = 4
        self.memory_length = 15 # 150 * 2 / 4 / 4 approx = 15
        self.follow_temporal_order = follow_temporal_order
        self.more_restrict_flag = more_restrict_flag


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
            action = self.child_conn.recv() # already 1 to 8 rather than 1 to 18

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
            agent_pos = self.env.agent_pos
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

            # since PPO has reward clipping, we need to scale down the reward
            reward = np.clip(reward, -1, 1) # 300 -> 1

            self.temp_action_arr.append(action)

            # ------------- Walkthrough Steps -------------------
            if self.walkthrough_data_arr is not None:
                self._stride_count -= 1
                if self._stride_count <= 0 :
                    self._stride_count = self.stride_count
                    # feed previous memory directly
                    if len(self.location_seq_memory) == 0:
                        # add previous memory
                        for i in range(self.memory_length):
                            self.location_seq_memory.append(agent_pos)
                            self.action_freq_seq_memory.append(bag_of_actions([list(MR_ATARI_ACTION).index(MR_ATARI_ACTION.NOOP)]))

                    self.location_seq_memory.append(agent_pos)
                    self.action_freq_seq_memory.append(bag_of_actions(self.temp_action_arr))
                    self.temp_action_arr = [] # refresh temp action arr

                    # pop if memory exceed
                    if len(self.location_seq_memory) > self.memory_length:
                        # pop the oldest element
                        self.location_seq_memory.pop(0)
                        self.action_freq_seq_memory.pop(0)

                    # Calculate Lang Rew
                    auxiliary_rew_total_imme = 0.0
                    # -------- calculate auxiliary rewards -----------
                    for ind, activation_flag in enumerate(self.walkthrough_activation_flag):
                        if activation_flag:
                            # meaning that we measure the memory for this walkthrough sentence
                            walkthrough_action_requirement = self.walkthrough_data_arr[ind]['action_freq']
                            walkthrough_location_requirement = self.walkthrough_data_arr[ind]['avatar_location']
                            is_match_action_constraint = True

                            target_action_memory = self.action_freq_seq_memory[-len(walkthrough_action_requirement):]

                            for t_ind, item in enumerate(walkthrough_action_requirement):
                                diff_candidates = [np.abs(item - target_action_memory[t_ind]).sum()]
                                if t_ind > 0:
                                    diff_candidates.append(np.abs(item - target_action_memory[t_ind-1]).sum())
                                if t_ind < len(walkthrough_location_requirement) - 1:
                                    diff_candidates.append(np.abs(item - target_action_memory[t_ind + 1]).sum())
                                if np.min(diff_candidates) > 0.6:
                                    is_match_action_constraint = False
                                    break

                            target_location_memory = self.location_seq_memory[-len(walkthrough_location_requirement):]
                            is_match_state_constraint = True
                            for t_ind, item in enumerate(walkthrough_location_requirement):
                                if item is None:
                                    continue
                                else:
                                    tar_loc_candidates = [target_location_memory[t_ind]]
                                    if t_ind > 0:
                                        tar_loc_candidates.append(target_location_memory[t_ind-1])
                                    if t_ind < len(walkthrough_location_requirement) - 1 :
                                        tar_loc_candidates.append(target_location_memory[t_ind+1])

                                    all_false_flag = True
                                    for candidatee in tar_loc_candidates:
                                        if (item[0] -5 < candidatee[0] < item[0] + 5) and (item[1] -5 < candidatee[1] < item[1] + 5):
                                            all_false_flag = False
                                            break

                                    if all_false_flag:
                                        is_match_state_constraint = False
                                        break



                            activate_next_flag = False
                            if self.more_restrict_flag:
                                if is_match_state_constraint and is_match_action_constraint:
                                    self.immediate_auxiliary_reward[ind] = 1.0
                                    activate_next_flag = True
                                    auxiliary_rew_total_imme += 1.0
                                    self.walkthrough_accu_rew[ind] += 1.0


                            else:
                                accu_temp = 0.0
                                if is_match_state_constraint:
                                    accu_temp += 0.5
                                # only state rewards
                                if is_match_action_constraint:
                                    accu_temp += 0.5
                                self.immediate_auxiliary_reward[ind] = accu_temp
                                auxiliary_rew_total_imme += accu_temp
                                self.walkthrough_accu_rew[ind] += accu_temp
                                if accu_temp != 0.0:
                                    activate_next_flag = True
                            if activate_next_flag and ind < (len(self.walkthrough_activation_flag) - 1):
                                self.walkthrough_activation_flag[ind+1] = True
                                # deactivate once we get some rewards
                                self.walkthrough_activation_flag[ind] = False
                            if is_match_state_constraint:
                                self.no_achieve_state_constraints += 1
                            if is_match_action_constraint:
                                self.no_achieve_action_constraints += 1

                    # intermediate
                    # auxiliary reward maximum is 1
                    # fair enough
                    auxiliary_rew_total_imme = auxiliary_rew_total_imme * 1.0
                    # auxiliary_rew_total_imme = 0 # TODO this means no auxiliary
                    reward += auxiliary_rew_total_imme

            # ------------- End of Lang Rew Steps ------------

            force_done = done

            self.history[:-1, :, :] = self.history[1:, :, :] # this is kind of push...
            self.history[-1, :, :] = self.pre_proc(s)

            self.rall += reward # self.rall reward until now
            self.steps += 1


            if self.walkthrough_data_arr is not None:
                self.child_conn.send(
                    [self.history[:, :, :], reward, force_done, done, log_reward,
                     [
                        self.no_of_wins, self.no_achieve_state_constraints, self.no_achieve_action_constraints, self.walkthrough_accu_rew, self.steps, self.immediate_auxiliary_reward,
                         self.location_seq_memory, self.env.current_room, self.rall, agent_pos, self.walkthrough_activation_flag
                    ]
                     ])
            else:
                self.child_conn.send(
                    [self.history[:, :, :], reward, force_done, done, log_reward,
                     [self.no_of_wins, self.no_achieve_state_constraints, self.no_achieve_action_constraints, self.walkthrough_accu_rew]
                     ])

            if done:
                self.recent_rlist.append(self.rall)
                if 'Montezuma' in self.env_name:
                    print("[Episode {}({})] Step: {}  Reward: {}  Recent Reward: {}  Visited Room: [{}] No. of wins: {}".format(
                        self.episode, self.env_idx, self.steps, self.rall, np.mean(self.recent_rlist),
                        info.get('episode', {}).get('visited_rooms', {}), self.no_of_wins))
                    print('env visited room 5', self.env.visited_rooms)

                else:
                    print("[Episode {}({})] Step: {}  Reward: {}  Recent Reward: {}".format(
                        self.episode, self.env_idx, self.steps, self.rall, np.mean(self.recent_rlist)))

                self.history = self.reset()

    def reset_auxiliary_rew_stuffs(self):
        # memory for reward shaping module

        self.location_seq_memory = []
        self.action_freq_seq_memory = []
        self.temp_action_arr = []
        self._stride_count = self.stride_count



        if self.follow_temporal_order:
            self.walkthrough_activation_flag = [False for _ in range(len(self.walkthrough_raw_sentence))]
            self.walkthrough_activation_flag[0] = True
        else:
            self.walkthrough_activation_flag = [True for _ in range(len(self.walkthrough_raw_sentence))]

        # lang reward
        self.immediate_auxiliary_reward = [0.0 for _ in
                                           range(
                                          len(self.walkthrough_raw_sentence))]  # length = num of instruction sentence




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
        if self.walkthrough_data_arr is not None:
            self.reset_auxiliary_rew_stuffs()

        return self.history[:, :, :]


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

