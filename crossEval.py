#coding:utf-8
import torch
import time
import os
from torch import nn, optim
from dataset_MMFN import DataSetMMFN_semiChs, data_get_binChs
from torch.utils.data import random_split, DataLoader
from tqdm import tqdm
from sklearn.metrics import recall_score, classification_report, precision_score, precision_recall_fscore_support, \
    accuracy_score
from MMFN_model import MMFN_semi_Texture_Branch, MMFN_semi_Visual_Branch, CTCoAttentionTransformer, MMFN_classifier
from MMFN_config import MMFN_config
import numpy as np

device = "cuda:1" if torch.cuda.is_available() else "cpu"


def fit(model, dataloads, device):

    torch.set_grad_enabled(False)
    model.eval()

    running_loss = 0.0
    acc = 0.0
    step = 0
    all = 0
    rec = 0
    report = 0

    all_label = []
    all_pred = []
    for input_xlnet, input_swin, input_clip_text, input_clip_img, batchLabel in tqdm(dataloads, desc='进度', leave=True, ncols=100):
        step += 1
        all += batchLabel.view(-1).shape[0]

        input_clip_text = input_clip_text.to(torch.float32)
        input_clip_img = input_clip_img.to(torch.float32)
        detector_decode = model(
            input_xlnet.to(device), input_swin.to(device), input_clip_text.to(device), input_clip_img.to(device))

        _, pred = detector_decode.topk(1)

        # _, pred = label_pred.topk(1)


        all_label.append(batchLabel.view(-1).cpu().numpy())
        all_pred.append(pred.view(-1).cpu().numpy())


    precision, recall, f1, support = precision_recall_fscore_support(np.hstack(all_label), np.hstack(all_pred))
    accuracy = accuracy_score(np.hstack(all_label), np.hstack(all_pred))
    report = classification_report(np.hstack(all_label), np.hstack(all_pred))
    return accuracy, precision, recall, f1, support, report


def train_process(ext, device='cpu', epochs=10, batch_size=32, datapath = './data/weibo16/'):
    datapath = './data/weibo21/'

    paths = []

    for file in os.listdir(datapath):
        if '.pt' in file and 'single' not in file:
            p = datapath + file
            paths.append(p)

    datas = data_get_binChs(paths)

    mm_ds = DataSetMMFN_semiChs(datas, datas[0].shape[0])

    model = MMFN_classifier(device)
    modelPath = './exist_model/_MMFN_semiChs_xlnetSingle0915_99.pth'
    model.load_state_dict(torch.load(modelPath))
    model.to(device)


    model.eval()

    eval_loader = DataLoader(mm_ds, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=False)

    eval_results = {
        'running_loss' :[],
        'accuracy': [],
        'precision': [],
        'recall': [],
        'f1': [],
        'support': []
    }


    eval_accuracy, eval_precision, eval_recall, eval_f1, eval_support, eval_report = fit(model,
                                                             eval_loader, device)

    message = ('\n\teval_report' + eval_report
               )
    print(message)

if __name__ == '__main__':
    # weibo16_path = './data/weibo16/'
    # weibo21_path = './data/weibo21/'
    #
    ext = '123'
    # print("begin")
    # ext = input("训练即将开始，请输入字段以判明哪次训练：")
    # data_select = input("weibo16 or weibo 21:")
    # if data_select == 'weibo16':
    #     datapath = weibo16_path
    # else:
    #     datapath = weibo21_path

    train_process(ext, device=device, epochs=100, batch_size=250)