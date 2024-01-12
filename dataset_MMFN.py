## coding:utf-8
import os

import numpy as np
import torch

from torch.utils.data import Dataset

class DataSetMMFN_semiChs(Dataset):
    def __init__(self, dataAll, length):
        self.data = dataAll
        self.length = length

    def __getitem__(self, idx: int):
        return self.data[0][idx], self.data[1][idx], self.data[2][idx], self.data[3][idx], self.data[4][idx]

    def __len__(self):
        return self.length


# def data_get_binChs(paths):
#     mark = False
#     xlnet = []
#     swin = []
#     clip_txt = []
#     clip_img = []
#     for path in paths:
#         if 'nonrumor' in path or 'real' in path:
#             continue
#         if 'xlnet' in path:
#             if xlnet == []:
#                 xlnet = np.fromfile(path, dtype=np.float32).reshape(-1, 144, 768)
#             else:
#                 xlnet = np.concatenate((xlnet, np.fromfile(path, dtype=np.float32).reshape(-1, 144, 768)), axis=0)
#         elif 'swin' in path:
#             if swin == []:
#                 swin = np.fromfile(path, dtype=np.float32).reshape(-1, 144, 1024)
#             else:
#                 swin = np.concatenate((swin, np.fromfile(path, dtype=np.float32).reshape(-1, 144, 1024)), axis=0)
#         elif 'text' in path:
#             if clip_txt == []:
#                 clip_txt = np.fromfile(path, dtype=np.float16).reshape(-1, 512)
#             else:
#                 clip_txt = np.concatenate((clip_txt, np.fromfile(path, dtype=np.float16).reshape(-1, 512)), axis=0)
#         elif 'image' in path:
#             if clip_img == []:
#                 clip_img = np.fromfile(path, dtype=np.float16).reshape(-1, 512)
#             else:
#                 clip_img =  np.concatenate((clip_img, np.fromfile(path, dtype=np.float16).reshape(-1, 512)), axis=0)
#     label = torch.full((xlnet.shape[0],), 0, dtype=torch.int64)
#
#     xlnet2 = []
#     swin2 = []
#     clip_txt2 = []
#     clip_img2 = []
#     mark = True
#     for path in paths:
#         if '_rumor_' in path or 'fake' in path:
#             continue
#         if 'xlnet' in path:
#             if xlnet2 == []:
#                 xlnet2 = np.fromfile(path, dtype=np.float32).reshape(-1, 144, 768)
#             else:
#                 xlnet2 = np.concatenate((xlnet2, np.fromfile(path, dtype=np.float32).reshape(-1, 144, 768)), axis=0)
#         elif 'swin' in path:
#             if swin2 == []:
#                 swin2 = np.fromfile(path, dtype=np.float32).reshape(-1, 144, 1024)
#             else:
#                 swin2 = np.concatenate((swin2, np.fromfile(path, dtype=np.float32).reshape(-1, 144, 1024)), axis=0)
#         elif 'text' in path:
#             if clip_txt2 == []:
#                 clip_txt2 = np.fromfile(path, dtype=np.float16).reshape(-1, 512)
#             else:
#                 clip_txt2 = np.concatenate((clip_txt2, np.fromfile(path, dtype=np.float16).reshape(-1, 512)), axis=0)
#         elif 'image' in path:
#             if clip_img2 == []:
#                 clip_img2 = np.fromfile(path, dtype=np.float16).reshape(-1, 512)
#             else:
#                 clip_img2 =  np.concatenate((clip_img2, np.fromfile(path, dtype=np.float16).reshape(-1, 512)), axis=0)
#
#     label = torch.cat((label, torch.full((xlnet.shape[0],), 1, dtype=torch.int64)))
#
#
#     return torch.tensor(np.concatenate((xlnet, xlnet2), axis=0)), torch.tensor(np.concatenate((swin, swin2), axis=0)), \
#              torch.tensor(np.concatenate((clip_txt, clip_txt2), axis=0)), torch.tensor(np.concatenate((clip_img, clip_img2), axis=0)), \
#                 label

#


def data_get_binChs(paths):
    for path in paths:
        if 'bert' in path:
            bert_features = torch.load(path)
        elif 'swin' in path:
            swin_features = torch.load(path)
        elif 'text' in path:
            clip_txt_features = torch.load(path)
        elif 'image' in path:
            clip_img_features = torch.load(path)
        elif 'label' in path:
            label = torch.load(path)
    return bert_features, swin_features, clip_txt_features, clip_img_features, label

if __name__ == '__main__':
    abs_path = './data/weibo16/'

    paths = []
    for file in os.listdir(abs_path):
        if '.pt' in file:
            p = abs_path + file
            paths.append(p)

    data = data_get_binChs(paths)
    print()