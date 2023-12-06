import os
import torch
import numpy as np

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
    paths = []
    for file in os.listdir(abs_path):
        if 'text' in file:
            p = abs_path + file
            paths.append(p)

    data_get_binChs(paths)
    print('end')