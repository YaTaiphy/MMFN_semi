from MMFN_model import MMFN_classifier
import torch
import numpy as np
import sys
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

if __name__ == '__main__':
    # test20230912()
    # import torch
    # import torchvision.transforms as transforms
    # from torchvision.models import resnet50
    # from PIL import Image
    # ResNet_model = resnet50(pretrained=False)
    # ResNet_model.load_state_dict(torch.load("./pre-trained-model/resnet50-19c8e357.pth"))
    # ResNet_model.eval()
    
    # ResNet_model_2048 = torch.nn.Sequential(*(list(ResNet_model.children())[:-1]))
    
    # transform = transforms.Compose([
    #     transforms.Resize((224, 224)),
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    # ])
    
    # image_path = "./temp/pigFish_01.jpg"
    # image = Image.open(image_path).convert('RGB')
    # input_image = transform(image).unsqueeze(0)

    # # 6. 使用模型进行推理
    # with torch.no_grad():
    #     output = ResNet_model_2048(input_image)

    # # 7. 提取特征向量
    # feature_vector = output.squeeze().numpy()

    # # 输出特征向量的形状
    # print("Feature vector shape:", feature_vector.shape)
    a = torch.load("./data/weibo16/label_single_16.pt")
    b = a.numpy()
    mark = 1
    cout = 0
    for each in b:
        if each == mark:
            cout = cout + 1
        else:
            mark = abs(abs(mark) - 1)
            print(cout)
            cout = 1
    print(cout)