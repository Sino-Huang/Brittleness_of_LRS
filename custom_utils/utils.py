import io

import numpy as np
import cv2
import torch
from hydra.utils import to_absolute_path
from einops import reduce, repeat, rearrange
from omegaconf import DictConfig, OmegaConf
import wandb

def fromfile(file, dtype, count, *args, **kwargs):
    """Read from any file-like object into a numpy array
       im = io.BytesIO(f.read())
       m = fromfile(im, dtype=np.uint8, count=im.__sizeof__())"""

    itemsize = np.dtype(dtype).itemsize
    buffer = np.zeros(count * itemsize, np.uint8)
    bytes_read = -1
    offset = 0
    while bytes_read != 0:
        bytes_read = file.readinto(buffer[offset:])
        offset += bytes_read
    rounded_bytes = (offset // itemsize) * itemsize
    buffer = buffer[:rounded_bytes]
    buffer.dtype = dtype
    return buffer

def readimage(filepath, img_size = 96, mean = None, std = None):
    r'''

    :param filepath:
    :param img_size:
    :param mean:
    :param std:
    :return: output shape: -> [C, W, H]
    '''
    with open(to_absolute_path(filepath),
              'rb') as f:
        im = io.BytesIO(f.read()) # this speed up the read as we will bring all data to the buffer
        im = fromfile(im, dtype=np.uint8, count=im.__sizeof__())
        im = cv2.imdecode(im, cv2.IMREAD_COLOR)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        # rescale
        if im.shape[0] != img_size:
            print(f"image size is {im.shape[0]}, going to change to {img_size}")
            im = cv2.resize(im, (img_size, img_size), interpolation=cv2.INTER_AREA)
        # apply mean and std
        mean = np.asarray(mean)
        std = np.asarray(std)
        mean = repeat(mean, 'd -> w h d', w=img_size, h=img_size)
        std = repeat(std, 'd -> w h d', w=img_size, h=img_size)
        # im = im / 255.0
        im = (im - mean) / std
        # transpose
        im = rearrange(im, "w h d -> d w h")
    return im

def readVideo(filepath, mean=None, std=None, windowsize=10, resize=None):
    '''
    reprocessed (original-mean) / std
    '''
    cap = cv2.VideoCapture(filepath)
    if not cap.isOpened():
        print("Error opening video stream or file")
    # read until video is completed
    frames = []
    count = 0
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret:
            if count % windowsize == 0:
                # origin shape (84, 84, 3)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # resize
                if resize is not None:
                    frame = cv2.resize(frame, (resize, resize), interpolation = cv2.INTER_AREA)
                frames.append(frame)
            count+=1
        else:
            break
    cap.release()
    if mean is None or std is None:
        frames = np.asarray(frames, dtype=np.uint8) # frame, 84, 84, 3
        return frames
    else:
        frames = np.asarray(frames, dtype=np.float32) # type ndarray(15, 84, 84, 3)
        mean = np.asarray(mean)
        std = np.asarray(std)
        # print(frames.shape)
        mean = repeat(mean, 'd -> f w h d', f=frames.shape[0], w=frames.shape[1], h=frames.shape[2])
        std = repeat(std, 'd -> f w h d', f=frames.shape[0], w=frames.shape[1], h=frames.shape[2])
        # im = im / 255.0
        frames = (frames - mean) / std
        frames = rearrange(frames, 'f w h d -> f d w h')
        return frames

def save_to_video(data, outputpath, fps=50, img_size=84):
    '''
    data type ndarray (frame, w, h, 3) uint8
    '''
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(outputpath, fourcc, fps, (img_size, img_size))
    for f in data:
        img = cv2.cvtColor(f, cv2.COLOR_RGB2BGR)
        img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_AREA)
        writer.write(img)
    writer.release()


def wandb_init(cfg: DictConfig, jobname, tagslist, notes=None):
    wandb.init(
        project='Goyal Reward Shaping Improvement',
        name=jobname,
        notes=notes,
        tags=tagslist,
        config=cfg
    )



def generalized_time_iou(event_time_box_p, event_time_box_g):
    """
    Generalized IoU from https://giou.stanford.edu/
    The event_time_box should in shape in [Batch, 2] format 2 are event_start, and event_dur
    """

    # Calculating start point
    start_p = event_time_box_p[:, 0]
    start_g = event_time_box_g[:, 0]

    # Calculating end point
    end_p = event_time_box_p[:, 0] + event_time_box_p[:, 1] # start + dur = end
    end_g = event_time_box_g[:, 0] + event_time_box_g[:, 1] # start + dur = end # shape: [B]

    # Calculating area
    area_p = event_time_box_p[:, 1] # shape: [B]
    area_g = event_time_box_g[:, 1]

    # Calculating intersection I
    i_start = torch.max(start_p, start_g)
    i_end = torch.min(end_p, end_g)

    inter = (i_end - i_start).clamp(min=0.0)

    # calculate Union
    union = area_p + area_g - inter

    # calculate area C
    smallest_start = torch.min(start_p, start_g)
    largest_end = torch.max(end_p, end_g)
    area_c = largest_end - smallest_start

    # calcualte GIoU
    iou = inter / union
    giou = iou - (area_c - union) / area_c

    return giou # shape [B]





def get_relative_offset(x, ratio=0.5): # x shape (batch, frame, c, w, h)
    # action predictor input shape (batch, LEN(2), channel, W, H)
    # action predictor output shape (batch, action_size)
    # output shape [batch, frame-1, 2, channel, W, H]
    front = x[:, :-1] * ratio
    end = x[:, 1:]
    output = end - front
    return output # output shape [batch, frame-1, channel, W, H]


def test_method():
    output = readVideo(
        "/home/sukai/Documents/Goyal_reward_shaping_improvement/data/human_replay_data_video_raw/0/0.mp4",
        mean=[0.485 * 255.0, 0.456 * 255.0, 0.406 * 255.0],
        std=[0.229 * 255.0, 0.224 * 255.0, 0.225 * 255.0],
    )
    a = 1




if __name__ == '__main__':
    print(generalized_time_iou(torch.tensor([[0.0,-0.5]]), torch.tensor([[0.4,0.1]])))