#coding:utf-8
import torch
import time
import os
from torch import nn, optim
from dataset_MMFN import DataSetMMFN_semiChs, data_get_binEng, data_get_binEng_dev, data_get_binEng_test
from torch.utils.data import random_split, DataLoader
from tqdm import tqdm
from sklearn.metrics import recall_score, classification_report, precision_score, precision_recall_fscore_support, \
    accuracy_score
from MMFN_model import MMFN_semi_Texture_Branch, MMFN_semi_Visual_Branch, CTCoAttentionTransformer, MMFN_classifier
from MMFN_config import MMFN_config
import numpy as np

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

    all_label = []
    all_pred = []
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

        loss = criterion(detector_decode.to(device), batchLabel.to(device).view(-1))
        running_loss += loss

        all_label.append(batchLabel.view(-1).cpu().numpy())
        all_pred.append(pred.view(-1).cpu().numpy())
        # precision, recall, f1, support = precision_recall_fscore_support(batchLabel.view(-1).cpu().numpy(), pred.view(-1).cpu().numpy())
        # accuracy = accuracy_score(batchLabel.view(-1).cpu().numpy(), pred.view(-1).cpu().numpy())
        # acc += (pred.view(-1).data.cpu() == batchLabel.view(-1).data).sum()
        # report = classification_report(batchLabel.view(-1).cpu().numpy(), pred.view(-1).cpu().numpy())

        if train:
            optimizer.zero_grad()  # 清除上一轮的梯度，防止累积.cpu().numpy()
            loss.backward()
            optimizer.step()

        running_loss = float(running_loss) / step
        # if train:
        #   scheduler.step()
        # if step > 3:
        #     break

    precision, recall, f1, support = precision_recall_fscore_support(np.hstack(all_label), np.hstack(all_pred))
    accuracy = accuracy_score(np.hstack(all_label), np.hstack(all_pred))
    report = classification_report(np.hstack(all_label), np.hstack(all_pred))
    return running_loss, accuracy, precision, recall, f1, support, report


def train_process(ext, device='cpu', epochs=10, batch_size=32, datapath = './data/mediaeval/'):
    type_name = 'MMFN_semiEng_' + ext


    # for file in os.listdir(datapath):
    #     if '.pt' in file and 'single' in file:
    #         p = datapath + file
    #         paths.append(p)

    # datas = data_get_binChs(paths)
    # mm_ds = DataSetMMFN_semiChs(datas, datas[0].shape[0])

    model_save_path = './exist_model/' + type_name
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)
    paths = []
    datapath = './data/mediaeval_sub/'
    for file in os.listdir('./data/mediaeval_sub/'):
        if '.pt' in file:
            p = datapath + file
            paths.append(p)
    datas_train = data_get_binEng_dev(paths)
    mm_ds_train = DataSetMMFN_semiChs(datas_train, datas_train[0].shape[0])
    datas_test = data_get_binEng_test(paths)
    mm_ds_test = DataSetMMFN_semiChs(datas_test, datas_test[0].shape[0])
    train_loader = DataLoader(mm_ds_train, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)
    eval_loader = DataLoader(mm_ds_test, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=False)

    train_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    fd = open(model_save_path + '/' + type_name + '_log.txt', 'a+')
    fd.write(train_time)
    fd.close()

    model = MMFN_classifier(device)
    model.to(device)
    model.train()
    # train_data, eval_data = random_split(mm_ds,
    #                                      [round(0.8 * mm_ds.__len__()),
    #                                       mm_ds.__len__() - round(0.8 * mm_ds.__len__())],
    #                                      generator=torch.Generator().manual_seed(42))
    # train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)
    # eval_loader = DataLoader(eval_data, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=False)

    optimizer = optim.Adam(model.parameters(), lr=0.000001)
    criterion = nn.CrossEntropyLoss()

    train_results = {
        'running_loss' :[],
        'accuracy' :[],
        'precision': [],
        'recall': [],
        'f1': [],
        'support': []
    }
    eval_results = {
        'running_loss' :[],
        'accuracy': [],
        'precision': [],
        'recall': [],
        'f1': [],
        'support': []
    }

    for epoch in range(epochs):
        print("rounds" + str(epoch))
        train_loss, train_accuracy, train_precision, train_recall, train_f1, train_support, train_report = fit(model,
                                                                     train_loader, optimizer, criterion, device,
                                                                     batch_size,
                                                                     train=True)
        train_results['running_loss'].append(train_loss)
        train_results['accuracy'].append(train_accuracy)
        train_results['precision'].append(train_precision)
        train_results['recall'].append(train_recall)
        train_results['f1'].append(train_f1)
        train_results['support'].append(train_support)

        eval_loss, eval_accuracy, eval_precision, eval_recall, eval_f1, eval_support, eval_report = fit(model,
                                                                 eval_loader, optimizer, criterion, device, batch_size,
                                                                 train=False)

        eval_results['running_loss'].append(eval_loss)
        eval_results['accuracy'].append(eval_accuracy)
        eval_results['precision'].append(eval_precision)
        eval_results['recall'].append(eval_recall)
        eval_results['f1'].append(eval_f1)
        eval_results['support'].append(eval_support)

        message = ('\n\tEpoch:' + str(epoch + 1) +
                   '\n\tTrain_loss: %.4f' % train_loss + '|Train_acc: %.4f' % train_accuracy +
                   '\n\tTrain_report' + train_report +
                   '\n\teval_loss: %.4f' % eval_loss + '|eval_acc: %.4f' % eval_accuracy +
                   '\n\teval_report' + eval_report
                   )
        print(message)
        fd = open(model_save_path + '/' + type_name + '_log.txt', 'a+')
        fd.write(message)
        fd.close()
        torch.save(model.state_dict(), model_save_path + '/' + type_name + '_' + str(epoch) + '.pth')

    # 访问加载的数据参考
    # loaded_data = np.load('results.npz')
    # loaded_precision = loaded_data['precision']
    # loaded_recall = loaded_data['recall']
    # loaded_f1 = loaded_data['f1']
    # loaded_support = loaded_data['support']

    np.savez(model_save_path + '/' + type_name + '_' + 'train_results.npz', **train_results)
    np.savez(model_save_path + '/' + type_name + '_' + 'eval_results.npz', **eval_results)


if __name__ == '__main__':


    ext = '123'
    print("begin")
    ext = input("训练即将开始，请输入字段以判明哪次训练：")


    train_process(ext, device=device, epochs=100, batch_size=250)