# Config
# import packages
import cv2
import hydra
import numpy as np
import torch
import re
import pickle
import os

from hydra.utils import get_original_cwd
from tqdm import tqdm_notebook as tqdm
from scipy.spatial.distance import cosine
from transformers import AutoModel, AutoTokenizer
from sentence_transformers import SentenceTransformer
from flair.data import Sentence
from flair.models import SequenceTagger
from omegaconf import OmegaConf, DictConfig

from backbones import RMSNorm
from custom_utils import readVideo
from models.lang_rew_shaping_event_detection_ver import RewardShapingModule
import PySimpleGUI as sg

def imgtoByte(img, resize=256, is_rbg=False):
    # return imBytes
    img = cv2.resize(img, (resize, resize))
    if is_rbg:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    imBytes = cv2.imencode('.png', img)[1].tobytes()
    return imBytes

@hydra.main(version_base=None, config_path="../../config", config_name="lang_rew_module_sentence_emb_hard_negative_version")
def main(cfg:DictConfig):
    cfg = DictConfig(OmegaConf.to_container(cfg, resolve=True))
    SEN_EMB_DIM = 768
    img_mean = eval(cfg.CONSTANT.IMG_MEAN)
    img_std = eval(cfg.CONSTANT.IMG_STD)
    # import sentence embedding model

    sent_emb_model = SentenceTransformer('sentence-transformers/gtr-t5-xl') # use by call .encode(), output is ndarray (768,)

    # data video path
    instruction_sent_emb_phrase_chunk_data_path = os.path.join(get_original_cwd(), 'data/train_data/training_instruction_sen_emb_phrase_chunk.pkl')
    with open(instruction_sent_emb_phrase_chunk_data_path, 'rb') as f:
        data_dict = pickle.load(f)
    # data instruction dict path (processed)
    # ```
    # dict: {<data_id> : {
    #                     "spell_correction_txt": type->str,
    #                     "embedding": type-> ndarray(768,),
    #                     "phrase_chunking_sentence": type -> lst(str)
    #                     "phrase_chunking_tag": type-> lst(str)
    #                     "verb_phrase_arr",
    #                     "noun_phrase_arr"
    #                     }
    #       }
    # ```

    # load reward shaping module
    model_path = os.path.join(get_original_cwd(), 'saved_model/pretrained_model/reward_shaping_module_best_state_dict_final.pth')
    hidden_dim = cfg.lang_rew_shaping_params.hidden_dim
    sentence_dim = cfg.CONSTANT.SEN_EMB_DIM
    img_size = cfg.CONSTANT.RESIZE
    action_size = cfg.CONSTANT.N_ACTIONS
    visual_encoder_cfg = cfg.visual_encoder_params
    visual_action_predictor_cfg = cfg.visual_action_predictor_params
    freeze_torso = cfg.lang_rew_shaping_params.freeze_torso
    transformer_depth = cfg.lang_rew_shaping_params.transformer_depth
    dropout_rate = cfg.lang_rew_shaping_params.dropout_rate
    norm_layer = cfg.lang_rew_shaping_params.norm_layer
    if norm_layer == "rmsnorm":
        norm_layer = RMSNorm
    elif norm_layer == "identity":
        norm_layer = torch.nn.Identity
    elif norm_layer == "batchnorm":
        norm_layer = torch.nn.BatchNorm1d

    visual_encoder_dict_path = os.path.join(get_original_cwd(), cfg.data_files.pretrain_visual_encoder_dict_path)
    visual_action_predictor_dict_path = os.path.join(get_original_cwd(),
                                                     cfg.data_files.pretrain_action_predictor_dict_path)
    reward_shaping_module = RewardShapingModule(hidden_dim, sentence_dim, cfg.device_info.device, img_size,
                                                action_size,
                                                visual_encoder_cfg, visual_action_predictor_cfg,
                                                freeze_torso, transformer_depth, dropout_rate, norm_layer,
                                                visual_encoder_dict_path, visual_action_predictor_dict_path).to(
        cfg.device_info.device)  # param (hidden_dim, sentence_dim, device, img_size, visual_encoder_cfg=None)

    reward_shaping_module.load_state_dict(
        torch.load(model_path))
    # maybe we can half it
    reward_shaping_module = reward_shaping_module.half()
    reward_shaping_module.eval()

    # video path
    video_folder_path = "/home/sukai/Extrastorage/Montezuma_Project/video_clip_ffmpeg_annotated_256"



    # FOR EACH STEP
    dict_len = len(data_dict)
    data_id = 0
    count = 0
    accumu_correct = 0
    AVERAGE_ACC = 0.683
    data_update_flag = True

    # GUI
    sg.theme('DarkAmber')
    sg.set_options(font="Arial 20")
    layout = [
        [sg.Frame("Genral Info", [
            [sg.Text(f'Average Accuracy is low, which is {AVERAGE_ACC}'), ],
            [sg.Text(f'Go To Data:'), sg.Input("", key="data_id_input"), sg.Button('Go', key="data_input_button") ],
            [sg.Text("Window Size") ,sg.Spin([4, 8, 12, 16], initial_value= 16, size=(5,5), enable_events=True, key="vid_window_size")]
        ])],
        [sg.Frame('Video Info',[
            [sg.Image(key='video_frame', size=(500, 500))],
            [sg.Text('', key='data_id')],
            [sg.Text('', key='video_frame_no')],
        ])],

        [sg.Frame("Sent Info", [
            [sg.Text('', key='raw_instruction_sent', font="Arial 30")],
            [sg.Text('Green means having this phrase will increase the matched probability\t',background_color='green')],
            [sg.Text('Red means having this phrase will decrease the matched probability,\nthe model think this phrase distracting \t',background_color='red')],
            [sg.Multiline('', key='verb_instruction_sent_phrase_chunk', size=(50,5))],
            [sg.Multiline('', key='noun_instruction_sent_phrase_chunk', size=(50,5))],
            [sg.Text('', key='raw_prediction_result')],
            [sg.Text("Prediction is: "), sg.Text('', key='raw_prediction_correct')],
        ])],
        [sg.Button("Next Data")],

    ]
    window = sg.Window('Evaluate MontezumasRevenge', layout, return_keyboard_events=True, use_default_focus=False)

    while True:
        # Set GUI loop
        event, value = window.read(timeout=300)
        if event in (sg.WIN_CLOSED, 'Cancel'):
            break

        if data_update_flag:
            data = data_dict.get(data_id)
            window_size = window['vid_window_size'].get()
            if data is not None:
                play_frame_no = 0
                ori_sent = data['spell_correction_txt']
                ori_sent_emb = data['embedding']
                video_data_path = os.path.join(video_folder_path, f'{data_id}.mp4')
                video_data = readVideo(video_data_path, img_mean, img_std, window_size, resize=cfg.CONSTANT.RESIZE)
                video_data_large = readVideo(video_data_path, windowsize= window_size)
                total_play_frame_no = len(video_data)

                ori_sent_emb = torch.asarray(ori_sent_emb, dtype=torch.float16).to(cfg.device_info.device, non_blocking=True).unsqueeze(0)
                video_data = torch.asarray(video_data, dtype=torch.float16).to(cfg.device_info.device, non_blocking=True).unsqueeze(0)

                logits = reward_shaping_module([ori_sent_emb, video_data]) # shape [B, 2]
                probability = torch.nn.functional.softmax(logits, dim=-1, dtype=torch.float16)[0]
                pred_class = torch.max(probability, 0)[1].detach().cpu().numpy()  # should be always 1
                if pred_class == 1:
                    accumu_correct += 1
                    correct_predict = True
                else:
                    correct_predict = False

                probability_raw = probability.detach().cpu().numpy()
                matched_probability = probability_raw[1] # type: int

                # do the phrase chunking stuffs
                verb_phrase_chunking_sentence = data['verb_phrase_arr']
                noun_phrase_chunking_sentence = data['noun_phrase_arr']
                verb_phrase_prob_diff = []
                noun_phrase_prob_diff = []


                for phrase in verb_phrase_chunking_sentence:
                    phrase_len = len(phrase) + 1
                    phrase_ind = ori_sent.find(phrase)
                    sent_without_phrase = ori_sent[:phrase_ind] \
                                          + ori_sent[phrase_ind + phrase_len:]

                    # print(sent_without_phrase, find_start_ind)
                    sent_without_phrase_emb = sent_emb_model.encode(sent_without_phrase)
                    sent_without_phrase_emb = torch.asarray(sent_without_phrase_emb, dtype=torch.float16).to(cfg.device_info.device, non_blocking=True).unsqueeze(0)
                    tlogits = reward_shaping_module([sent_without_phrase_emb, video_data])  # shape [B, 2]
                    tprobability = torch.nn.functional.softmax(tlogits, dim=-1, dtype=torch.float16)[0]
                    tprobability_raw = tprobability.detach().cpu().numpy()
                    tmatched_probability = tprobability_raw[1]  # type: int
                    prob_diff = matched_probability - tmatched_probability # + means this phrase is important in making it true
                    prob_diff_str = f'{"+" if prob_diff >0 else ""}{prob_diff:.3f}'
                    verb_phrase_prob_diff.append(prob_diff_str)

                for phrase in noun_phrase_chunking_sentence:
                    phrase_len = len(phrase) + 1
                    phrase_ind = ori_sent.find(phrase)
                    sent_without_phrase = ori_sent[:phrase_ind] \
                                          + ori_sent[phrase_ind + phrase_len:]

                    # print(sent_without_phrase, find_start_ind)
                    sent_without_phrase_emb = sent_emb_model.encode(sent_without_phrase)
                    sent_without_phrase_emb = torch.asarray(sent_without_phrase_emb, dtype=torch.float16).to(cfg.device_info.device, non_blocking=True).unsqueeze(0)
                    tlogits = reward_shaping_module([sent_without_phrase_emb, video_data])  # shape [B, 2]
                    tprobability = torch.nn.functional.softmax(tlogits, dim=-1, dtype=torch.float16)[0]
                    tprobability_raw = tprobability.detach().cpu().numpy()
                    tmatched_probability = tprobability_raw[1]  # type: int
                    prob_diff = matched_probability - tmatched_probability # + means this phrase is important in making it true
                    prob_diff_str = f'{"+" if prob_diff > 0 else ""}{prob_diff:.3f}'
                    noun_phrase_prob_diff.append(prob_diff_str)

                count += 1


                # print("sentence:", ori_sent, "probability", probability_raw)
                # print(f"{count}/{dict_len}: Accuracy is {accumu_correct / count:.3f}")
            else:
                data_id += 1  # changed by GUI
                continue
        if count >= dict_len: # meaning we run through all the data
            break


        # get video data
        imgbytes = imgtoByte(video_data_large[play_frame_no], resize=500, is_rbg=True)
        window['video_frame'].update(data=imgbytes)
        window['video_frame_no'].update(f"Frame: {play_frame_no}/{total_play_frame_no}")
        play_frame_no = (play_frame_no + 1) % total_play_frame_no

        # update data info
        if data_update_flag:
            window['data_id'].update(f"Data ID:{data_id}")
            window['raw_instruction_sent'].update(f'sentence: {ori_sent}')
            window['verb_instruction_sent_phrase_chunk'].update(f"verb phrase chunk:\n")
            window['noun_instruction_sent_phrase_chunk'].update(f"noun phrase chunk:\n")
            for ind, phrase in enumerate(verb_phrase_chunking_sentence):
                prob_diff = verb_phrase_prob_diff[ind]
                window['verb_instruction_sent_phrase_chunk'].print(f"{phrase} : {prob_diff}", background_color='green' if prob_diff[0] == "+" else "red")
            for ind, phrase in enumerate(noun_phrase_chunking_sentence):
                prob_diff = noun_phrase_prob_diff[ind]
                window['noun_instruction_sent_phrase_chunk'].print(f"{phrase} : {prob_diff}", background_color='green' if prob_diff[0] == "+" else "red")
            window['raw_prediction_result'].update(f'matched probability: {matched_probability}')
            window['raw_prediction_correct'].update(f'{"CORRECT" if correct_predict else "WRONG"}', background_color='green' if correct_predict else 'red')

            data_update_flag = False


        if event == "Next Data":
            data_id += 1
            data_update_flag = True
        if event == "vid_window_size":
            data_update_flag = True

        if event == "data_input_button":
            data_id = int(window['data_id_input'].get())
            data_update_flag = True

    window.close()
    # print(f"Final Accuracy is {accumu_correct / count:.3f}")
    # for best acc model -> Final Accuracy for all positive training dataset is 0.688 BAD
    # for final model -> Final Accuracy for all positive training dataset is 0.678, window size 15, BAD
    # for final model -> Final Accuracy for all positive training dataset is 0.683, window size 16, BAD
    # for final model -> Final Accuracy for all positive training dataset is 0.704, window size 10, BAD

    # calculate the match probability

    # create a group of phrase masked sentence and associated embedding using sentence embedding model

    # use language reward module to calculate the match probability

    # count the probability difference

    # create GUI to match video and instruction

if __name__ == '__main__':
    main()