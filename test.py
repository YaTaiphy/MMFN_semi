from MMFN_model import MMFN_classifier
import torch
import numpy as np
import sys
import os
def test20230912():
    device = "cuda:1" if torch.cuda.is_available() else "cpu"
    model_path = './exist_model/_MMFN_semiChs_20230910_99.pth'
    MMFN_semi_model = MMFN_classifier(device)
    MMFN_semi_model.load_state_dict(torch.load(model_path))
    MMFN_semi_model.to(device)
    MMFN_semi_model.eval()

    # xlnet_feature = np.fromfile('./data/weibo16/select_test_rumor_xlnet_features.bin', dtype=np.float32).reshape(-1, 144, 768)
    # swin_feature = np.fromfile('./data/weibo16/select_test_rumor_swinT_features.bin', dtype=np.float32).reshape(-1, 144, 1024)
    # clip_txt_feature = np.fromfile('./data/weibo16/select_test_rumor_clip_features_text.bin', dtype=np.float16).reshape(-1, 512)
    # clip_img_feature = np.fromfile('./data/weibo16/select_test_rumor_clip_features_image.bin', dtype=np.float16).reshape(-1, 512)

    xlnet_feature = np.fromfile('./data/weibo21/select_real_release_all_xlnet_features.bin', dtype=np.float32).reshape(-1, 144, 768)
    swin_feature = np.fromfile('./data/weibo21/select_real_release_all_swinT_features.bin', dtype=np.float32).reshape(-1, 144, 1024)
    clip_txt_feature = np.fromfile('./data/weibo21/select_real_release_all_clip_features_text.bin', dtype=np.float16).reshape(-1, 512)
    clip_img_feature = np.fromfile('./data/weibo21/select_real_release_all_clip_features_image.bin', dtype=np.float16).reshape(-1, 512)

    lengh = xlnet_feature.shape[0]

    xlnet_feature = torch.tensor(xlnet_feature)
    swin_feature = torch.tensor(swin_feature)
    clip_txt_feature = torch.tensor(clip_txt_feature)
    clip_img_feature = torch.tensor(clip_img_feature)

    j = 0
    for i in range(lengh):
        sys.stdout.write("\r")
        sys.stdout.flush()
        _, pred = MMFN_semi_model(xlnet_feature[i].unsqueeze(0).to(device), swin_feature[i].unsqueeze(0).to(device), clip_txt_feature[i].to(torch.float32).unsqueeze(0).to(device), clip_img_feature[i].to(torch.float32).unsqueeze(0).to(device)).topk(1)
        if pred[0][0] == 0:
            j = j + 1
        sys.stdout.write("Exceeding %d/%d" % (j, i))
    print("res %d/%d" % (j, lengh))

def conbineEngF20231031():
    datapath = './data/mediaeval/'

    clipFeaturesTextPaths = []
    clipFeaturesImagePaths = []
    swinTFeaturesPaths = []
    xlnetFeaturesPaths = []
    labelFeaturesPaths = []
    for file in os.listdir(datapath):
        if 'clip_features_text' in file:
            p = datapath + file
            clipFeaturesTextPaths.append(p)
        elif 'clip_features_image' in file:
            p = datapath + file
            clipFeaturesImagePaths.append(p)
        elif 'swinT_features' in file:
            p = datapath + file
            swinTFeaturesPaths.append(p)
        elif 'xlnet_features' in file:
            p = datapath + file
            xlnetFeaturesPaths.append(p)
        elif 'label_features' in file:
            p = datapath + file
            labelFeaturesPaths.append(p)
    clipFeaturesTextPaths.sort()
    clipFeaturesImagePaths.sort()
    swinTFeaturesPaths.sort()
    xlnetFeaturesPaths.sort()
    labelFeaturesPaths.sort()
    clipFeaturesText = []
    clipFeaturesImage = []
    swinTFeatures = []
    xlnetFeatures = []
    labelFeatures = []
    for i in range(len(clipFeaturesTextPaths)):
        if i == 0:
            clipFeaturesText = torch.load(clipFeaturesTextPaths[i])
            clipFeaturesImage = torch.load(clipFeaturesImagePaths[i])
            swinTFeatures = torch.load(swinTFeaturesPaths[i])
            xlnetFeatures = torch.load(xlnetFeaturesPaths[i])
            labelFeatures = torch.load(labelFeaturesPaths[i])
        else:
            clipFeaturesText = torch.cat((clipFeaturesText, torch.load(clipFeaturesTextPaths[i])), dim=0)
            clipFeaturesImage = torch.cat((clipFeaturesImage, torch.load(clipFeaturesImagePaths[i])), dim=0)
            swinTFeatures = torch.cat((swinTFeatures, torch.load(swinTFeaturesPaths[i])), dim=0)
            xlnetFeatures = torch.cat((xlnetFeatures, torch.load(xlnetFeaturesPaths[i])), dim=0)
            labelFeatures = torch.cat((labelFeatures, torch.load(labelFeaturesPaths[i])), dim=0)

    torch.save(clipFeaturesText, datapath + 'clip_features_text_All.pt')
    torch.save(clipFeaturesImage, datapath + 'clip_features_image_All.pt')
    torch.save(swinTFeatures, datapath + 'swinT_features_All.pt')
    torch.save(xlnetFeatures, datapath + 'xlnet_features_All.pt')
    torch.save(labelFeatures, datapath + 'label_features_All.pt')


def conbineEngF20231031_dev():
    datapath = './data/mediaeval/'

    clipFeaturesTextPaths = []
    clipFeaturesImagePaths = []
    swinTFeaturesPaths = []
    xlnetFeaturesPaths = []
    labelFeaturesPaths = []
    for file in os.listdir(datapath):
        if 'test' in file or 'All' in file or 'no2' in file:
            continue
        elif 'clip_features_text' in file:
            p = datapath + file
            clipFeaturesTextPaths.append(p)
        elif 'clip_features_image' in file:
            p = datapath + file
            clipFeaturesImagePaths.append(p)
        elif 'swinT_features' in file:
            p = datapath + file
            swinTFeaturesPaths.append(p)
        elif 'xlnet_features' in file:
            p = datapath + file
            xlnetFeaturesPaths.append(p)
        elif 'label_features' in file:
            p = datapath + file
            labelFeaturesPaths.append(p)
    clipFeaturesTextPaths.sort()
    clipFeaturesImagePaths.sort()
    swinTFeaturesPaths.sort()
    xlnetFeaturesPaths.sort()
    labelFeaturesPaths.sort()
    clipFeaturesText = []
    clipFeaturesImage = []
    swinTFeatures = []
    xlnetFeatures = []
    labelFeatures = []
    for i in range(len(clipFeaturesTextPaths)):
        if i == 0:
            clipFeaturesText = torch.load(clipFeaturesTextPaths[i])
            clipFeaturesImage = torch.load(clipFeaturesImagePaths[i])
            swinTFeatures = torch.load(swinTFeaturesPaths[i])
            xlnetFeatures = torch.load(xlnetFeaturesPaths[i])
            labelFeatures = torch.load(labelFeaturesPaths[i])
        else:
            clipFeaturesText = torch.cat((clipFeaturesText, torch.load(clipFeaturesTextPaths[i])), dim=0)
            clipFeaturesImage = torch.cat((clipFeaturesImage, torch.load(clipFeaturesImagePaths[i])), dim=0)
            swinTFeatures = torch.cat((swinTFeatures, torch.load(swinTFeaturesPaths[i])), dim=0)
            xlnetFeatures = torch.cat((xlnetFeatures, torch.load(xlnetFeaturesPaths[i])), dim=0)
            labelFeatures = torch.cat((labelFeatures, torch.load(labelFeaturesPaths[i])), dim=0)



    torch.save(clipFeaturesText, datapath + 'clip_features_text_dev.pt')
    torch.save(clipFeaturesImage, datapath + 'clip_features_image_dev.pt')
    torch.save(swinTFeatures, datapath + 'swinT_features_dev.pt')
    torch.save(xlnetFeatures, datapath + 'xlnet_features_dev.pt')
    torch.save(labelFeatures, datapath + 'label_features_dev.pt')


def conbineEngF20231031_test():
    datapath = './data/mediaeval/'

    clipFeaturesTextPaths = []
    clipFeaturesImagePaths = []
    swinTFeaturesPaths = []
    xlnetFeaturesPaths = []
    labelFeaturesPaths = []
    for file in os.listdir(datapath):
        if 'dev' in file or 'All' in file:
            continue
        elif 'clip_features_text' in file:
            p = datapath + file
            clipFeaturesTextPaths.append(p)
        elif 'clip_features_image' in file:
            p = datapath + file
            clipFeaturesImagePaths.append(p)
        elif 'swinT_features' in file:
            p = datapath + file
            swinTFeaturesPaths.append(p)
        elif 'xlnet_features' in file:
            p = datapath + file
            xlnetFeaturesPaths.append(p)
        elif 'label_features' in file:
            p = datapath + file
            labelFeaturesPaths.append(p)
    clipFeaturesTextPaths.sort()
    clipFeaturesImagePaths.sort()
    swinTFeaturesPaths.sort()
    xlnetFeaturesPaths.sort()
    labelFeaturesPaths.sort()
    clipFeaturesText = []
    clipFeaturesImage = []
    swinTFeatures = []
    xlnetFeatures = []
    labelFeatures = []
    for i in range(len(clipFeaturesTextPaths)):
        if i == 0:
            clipFeaturesText = torch.load(clipFeaturesTextPaths[i])
            clipFeaturesImage = torch.load(clipFeaturesImagePaths[i])
            swinTFeatures = torch.load(swinTFeaturesPaths[i])
            xlnetFeatures = torch.load(xlnetFeaturesPaths[i])
            labelFeatures = torch.load(labelFeaturesPaths[i])
        else:
            clipFeaturesText = torch.cat((clipFeaturesText, torch.load(clipFeaturesTextPaths[i])), dim=0)
            clipFeaturesImage = torch.cat((clipFeaturesImage, torch.load(clipFeaturesImagePaths[i])), dim=0)
            swinTFeatures = torch.cat((swinTFeatures, torch.load(swinTFeaturesPaths[i])), dim=0)
            xlnetFeatures = torch.cat((xlnetFeatures, torch.load(xlnetFeaturesPaths[i])), dim=0)
            labelFeatures = torch.cat((labelFeatures, torch.load(labelFeaturesPaths[i])), dim=0)

    torch.save(clipFeaturesText, datapath + 'clip_features_text_test.pt')
    torch.save(clipFeaturesImage, datapath + 'clip_features_image_test.pt')
    torch.save(swinTFeatures, datapath + 'swinT_features_test.pt')
    torch.save(xlnetFeatures, datapath + 'xlnet_features_test.pt')
    torch.save(labelFeatures, datapath + 'label_features_test.pt')

if __name__ == '__main__':
    # test20230912()
    conbineEngF20231031_dev()
    conbineEngF20231031_test()