#coding:utf-8
import torch
import time
import os
from torch import nn, optim
from dataset_MMFN import DataSetMMFN_semiChs, data_get_binChs
from torch.utils.data import random_split, DataLoader
from tqdm import tqdm
from sklearn.metrics import recall_score, classification_report, precision_score
from MMFN_model import MMFN_semi_Texture_Branch, MMFN_semi_Visual_Branch, CTCoAttentionTransformer, MMFN_classifier
from MMFN_config import MMFN_config

device = "cuda:1" if torch.cuda.is_available() else "cpu"


def fit(model, dataloads, optimizer, criterion, device, batch_size, train=True):
    if train:
        torch.set_grad_enabled(True)
        model.train()
    else:
        torch.set_grad_enabled(False)
        model.eval()

    running_loss = 0.0
    acc = 0.0
    step = 0
    all = 0
    rec = 0
    report = 0
    for input_xlnet, input_swin, input_clip_text, input_clip_img, batchLabel in tqdm(dataloads, desc='进度', leave=True, ncols=100):
        step += 1
        all += batchLabel.view(-1).shape[0]
        if train:
            optimizer.zero_grad()

        input_clip_text = input_clip_text.to(torch.float32)
        input_clip_img = input_clip_img.to(torch.float32)
        detector_decode = model(
            input_xlnet.to(device), input_swin.to(device), input_clip_text.to(device), input_clip_img.to(device))

        _, pred = detector_decode.topk(1)

        # _, pred = label_pred.topk(1)

        loss = criterion(detector_decode.to(device), batchLabel.to(device))
        running_loss += loss

        recall_mark = recall_score(batchLabel.view(-1).cpu().numpy(), pred.view(-1).cpu().numpy())
        rec = rec + recall_mark
        acc += (pred.view(-1).data.cpu() == batchLabel.view(-1).data).sum()
        report = classification_report(batchLabel.view(-1).cpu().numpy(), pred.view(-1).cpu().numpy())

        if train:
            optimizer.zero_grad()  # 清除上一轮的梯度，防止累积.cpu().numpy()
            loss.backward()
            optimizer.step()

        running_loss = float(running_loss) / step
        avg_acc = float(acc) / all
        avg_rec = rec / step
        # if train:
        #   scheduler.step()
    return running_loss, avg_acc, avg_rec, report


def train_process(ext, device='cpu', epochs=10, batch_size=32, datapath = './data/weibo16/'):
    type_name = 'MMFN_semiChs_' + ext

    paths = []

    for file in os.listdir(datapath):
        if '.pt' in file:
            p = datapath + file
            paths.append(p)

    datas = data_get_binChs(paths)

    mm_ds = DataSetMMFN_semiChs(datas, datas[0].shape[0])

    train_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    fd = open(r'./exist_model/' + type_name + '_log.txt', 'a+')
    fd.write(train_time)
    fd.close()

    model = MMFN_classifier(device)
    model.to(device)
    model.train()
    train_data, eval_data = random_split(mm_ds,
                                         [round(0.8 * mm_ds.__len__()),
                                          mm_ds.__len__() - round(0.8 * mm_ds.__len__())],
                                         generator=torch.Generator().manual_seed(42))
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)
    eval_loader = DataLoader(eval_data, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=False)

    optimizer = optim.Adam(model.parameters(), lr=0.000001)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        print("rounds" + str(epoch))
        train_loss, train_avg_acc, train_avg_rec, train_report = fit(model,
                                                                     train_loader, optimizer, criterion, device,
                                                                     batch_size,
                                                                     train=True)
        eval_loss, eval_avg_acc, eval_avg_rec, eval_report = fit(model,
                                                                 eval_loader, optimizer, criterion, device, batch_size,
                                                                 train=False)
        message = ('\n\tEpoch:' + str(epoch + 1) +
                   '\n\tTrain_loss: %.4f' % train_loss + '|Train_acc: %.4f' % train_avg_acc + '|Recall: %.4f' % train_avg_rec +
                   '\n\tTrain_report' + train_report +
                   '\n\teval_loss: %.4f' % eval_loss + '|eval_acc: %.4f' % eval_avg_acc + '|Recall: %.4f' % eval_avg_rec +
                   '\n\teval_report' + eval_report
                   )
        print(message)
        fd = open(r'./exist_model/' + type_name + '_log.txt', 'a+')
        fd.write(message)
        fd.close()
        torch.save(model.state_dict(), "./exist_model/" + '_' + type_name + '_' + str(epoch) + '.pth')


if __name__ == '__main__':
    weibo16_path = './data/weibo16/'
    weibo21_path = './data/weibo21/'

    ext = '123'
    print("begin")
    ext = input("训练即将开始，请输入字段以判明哪次训练：")
    data_select = input("weibo16 or weibo 21:")
    if data_select == 'weibo16':
        datapath = weibo16_path
    else:
        datapath = weibo21_path

    train_process(ext, device=device, epochs=100, batch_size=250)