import os
import torch
import numpy as np
import time
def data_get_binChs(paths):
    mark = False
    for path in paths:
        if 'nonrumor' in path or 'real' in path:
            continue
        if 'xlnet' in path:
            xlnet = np.fromfile(path, dtype=np.float32)
            xlnet = xlnet.reshape(-1, 144, 768)
        # elif 'swin' in path:
        #     swin = np.fromfile(path, dtype=np.float32)
        #     swin = swin.reshape(-1, 144, 1024)
        # elif 'text' in path:
        #     clip_txt = np.fromfile(path, dtype=np.float16)
        #     clip_txt = clip_txt.reshape(-1, 512)
        # elif 'image' in path:
        #     clip_img = np.fromfile(path, dtype=np.float16)
        #     clip_img = clip_img.reshape(-1, 512)
    label = torch.full((xlnet.shape[0],), 0, dtype=torch.int64)

    mark = True
    for path in paths:
        if '_rumor_' in path or 'fake' in path:
            continue
        if 'xlnet' in path:
            xlnet2 = np.fromfile(path, dtype=np.float32)
            xlnet2 = xlnet2.reshape(-1, 144, 768)
        # elif 'swin' in path:
        #     swin2 = np.fromfile(path, dtype=np.float32)
        #     swin2 = swin2.reshape(-1, 144, 1024)
        # elif 'text' in path:
        #     clip_txt2 = np.fromfile(path, dtype=np.float16)
        #     clip_txt2 = clip_txt2.reshape(-1, 512)
        # elif 'image' in path:
        #     clip_img2 = np.fromfile(path, dtype=np.float16)
        #     clip_img2 = clip_img2.reshape(-1, 512)

    label = torch.cat((label, torch.full((xlnet.shape[0],), 1, dtype=torch.int64)))

    # 指定文件路径
    file_path = 'tensor_data.pt'
    #
    # # 使用 torch.save() 将张量保存到二进制文件中
    # torch.save(tensor_data, file_path)




if __name__ == '__main__':
    abs_path = './data/weibo16/'
    ext = input("合并weibo16即将开始，请输入字段以判明合并哪片数据：")
    # ext = 'bert'
    paths = []
    for file in os.listdir(abs_path):
        if ext in file and file.endswith('.bin'):
            p = abs_path + file
            paths.append(p)
        # p = abs_path + file
        # paths.append(p)
    paths.sort()
    print(paths)

    saving_data = []
    label = []
    mark = 0
    for path in paths:
        if 'xlnet' in path or 'bert' in path:
            saving_data.append(torch.tensor(np.fromfile(path, dtype=np.float32).reshape(-1, 144, 768)))
        elif 'swin' in path:
            saving_data.append(torch.tensor(np.fromfile(path, dtype=np.float32).reshape(-1, 144, 1024)))
        elif 'text' in path:
            saving_data.append(torch.tensor(np.fromfile(path, dtype=np.float16).reshape(-1, 512)))
            mark = 1
            if 'nonrumor' in path or 'real' in path:
                label.append(torch.full((torch.tensor(np.fromfile(path, dtype=np.float16).reshape(-1, 512)).shape[0],), 1, dtype=torch.int64))
            else:
                label.append(torch.full((torch.tensor(np.fromfile(path, dtype=np.float16).reshape(-1, 512)).shape[0],), 0, dtype=torch.int64))
        elif 'image' in path:
            saving_data.append(torch.tensor(np.fromfile(path, dtype=np.float16).reshape(-1, 512)))
        elif 'vgg' in path:
            saving_data.append(torch.tensor(np.fromfile(path, dtype=np.float32).reshape(-1, 4096)))
    # data_get_binChs(paths)
    save_data = torch.cat(saving_data)
    saving_data = None
    print(save_data.shape)
    time.sleep(5)

    torch.save(save_data, './data/weibo16/' + ext + '16.pt')
    if mark == 1:
        label = torch.cat(label)
        print(label.shape)
        torch.save(label, './data/weibo16/label16.pt')

    print('end')