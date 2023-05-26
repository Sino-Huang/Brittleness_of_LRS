import pickle

import cv2
import gym
import PySimpleGUI as sg

from rnd_rl_env.envs import MR_ATARI_ACTION

def imgtoByte(img, is_rbg=False):
    # return imBytes
    img = cv2.resize(img, (256, 256))
    if is_rbg:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    imBytes = cv2.imencode('.png', img)[1].tobytes()
    return imBytes

def main():
    env = gym.make('MontezumaRevengeNoFrameskip-v4', render_mode='rgb_array')
    env.reset()
    with open('./easy_mr_system_state_cross_laser_gate_room_0.pkl', 'rb') as f:
        restorestate = pickle.load(f)
    env.ale.restoreSystemState(restorestate)

    sg.theme('DarkAmber')
    layout = [
        [sg.Push(), sg.vtop(sg.Text('Action:'.ljust(30), key='action_info')), sg.Frame("Game Content", [
            [sg.Image(key="obs_image", size=(256, 256))],  # use OpenCV later to import image
        ]), sg.Push()],
        [sg.Text("", key='n_step_info')],
        [sg.Text("", key='room_number_info')],
        [sg.Button("Save State"), sg.Button("Reset State")],

        [sg.Frame("Other info", [
            [sg.Text('', key=f'other info')]
        ])]

    ]
    window = sg.Window('Evaluate MontezumasRevenge', layout, return_keyboard_events=True, use_default_focus=False)

    while True:
        event, value = window.read()
        if event in (sg.WIN_CLOSED, 'Cancel'):
            break
        agent_act_flag = False
        # movement event
        if event == 'space:65':
            new_state, reward, done, info = env.step(MR_ATARI_ACTION.FIRE.value)
            window['action_info'].update(f'Action: {MR_ATARI_ACTION.FIRE.name}'.ljust(30))
            agent_act_flag = True
        elif event == 'Left:113':
            new_state, reward, done, info = env.step(MR_ATARI_ACTION.LEFT.value)
            window['action_info'].update(f'Action: {MR_ATARI_ACTION.LEFT.name}'.ljust(30))
            agent_act_flag = True
        elif event == 'Right:114':
            new_state, reward, done, info = env.step(MR_ATARI_ACTION.RIGHT.value)
            window['action_info'].update(f'Action: {MR_ATARI_ACTION.RIGHT.name}'.ljust(30))
            agent_act_flag = True
        elif event == 'Down:116':
            new_state, reward, done, info = env.step(MR_ATARI_ACTION.DOWN.value)
            window['action_info'].update(f'Action: {MR_ATARI_ACTION.DOWN.name}'.ljust(30))
            agent_act_flag = True
        elif event == 'Up:111':
            new_state, reward, done, info = env.step(MR_ATARI_ACTION.UP.value)
            window['action_info'].update(f'Action: {MR_ATARI_ACTION.UP.name}'.ljust(30))
            agent_act_flag = True
        elif event == 'z:52':
            new_state, reward, done, info = env.step(MR_ATARI_ACTION.LEFTFIRE.value)
            window['action_info'].update(f'Action: {MR_ATARI_ACTION.LEFTFIRE.name}'.ljust(30))
            agent_act_flag = True
        elif event == 'x:53':
            new_state, reward, done, info = env.step(MR_ATARI_ACTION.RIGHTFIRE.value)
            window['action_info'].update(f'Action: {MR_ATARI_ACTION.RIGHTFIRE.name}'.ljust(30))
            agent_act_flag = True
        elif event == 'c:54':
            new_state, reward, done, info = env.step(MR_ATARI_ACTION.NOOP.value)
            window['action_info'].update(f'Action: {MR_ATARI_ACTION.NOOP.name}'.ljust(30))
            agent_act_flag = True

        if agent_act_flag:
            # update image
            render_img = new_state
            imBytes = imgtoByte(render_img, is_rbg=True)
            window['obs_image'].update(data=imBytes)
            if reward >0:
                window['other info'].update(f"reward: {reward}")
            window['room_number_info'].update(f'room: {env.ale.getRAM()[3]}')


        if event == 'Save State':
            print('save state')
            systemstate = env.ale.cloneSystemState() # cloneSystemState vs restoreSystemState
            with open('mr_system_state.pkl', 'wb') as f:
                pickle.dump(systemstate, f)

        if event == 'Reset State':
            print('reset state')
            print(type(env.ale.restoreSystemState(restorestate)))


if __name__ == '__main__':
    main()