import torch
from PIL import Image
from transformers import SwinModel, AutoFeatureExtractor
import clip
import numpy as np
import cn_clip.clip as cnClip
from cn_clip.clip import load_from_name
import langid
class swin_base_patch4_window12_384_model(torch.nn.Module):
    def __init__(self, device):
        super(swin_base_patch4_window12_384_model, self).__init__()
        self.device = device
        self.pre_train_model = SwinModel.from_pretrained('./pre-trained-model/swin-base-patch4-window12-384-in22k')
        self.pre_train_model.to(device)
        for param in self.pre_train_model.parameters():
            param.requires_grad = False
        self.pre_train_model.eval()

        self.feature_extractor = AutoFeatureExtractor.from_pretrained('../co-attention/swin-base-patch4-window12-384-in22k')

    def forward(self, inputs):
        return self.pre_train_model(inputs)

    def get_feature(self, image):
        inputs = self.feature_extractor(images=image, return_tensors="pt", device=self.device)
        outputs = self.pre_train_model(inputs)
        return outputs.last_hidden_state

class CLIP_model(object):
    device = "cuda:1" if torch.cuda.is_available() else "cpu"
    clipEng, clipPreprocessEng = clip.load("./pre-trained-model/ViT-B-32", device=device)
    clipChs, ClipPreprocessChs = load_from_name("./pre-trained-model/ViT-B-16", device=device, download_root='./pre-trained-model/')
    def __int__(self):
        super(CLIP_model, self).__init__()

    def get_clipEng(self, sentence, imagePath):
        with torch.no_grad():
            text_encoded = self.clipEng.encode_text(clip.tokenize(sentence, 77, True)).to(self.device)
            # text_encoded = myclip.encode_text(clip.tokenize(sentence).to(device))
            text_encoded /= text_encoded.norm(dim=-1, keepdim=True)
        # feat_t = text_encoded[0].cpu().numpy()
        feat_t = text_encoded.cpu().numpy()
        with torch.no_grad():
            image = torch.stack([self.clipPreprocessEng(Image.open(image)) for image in imagePath]).to(self.device)
            # image = preprocess(Image.open(imagePath)).unsqueeze(0).to(device)
            image_feature = self.clipEng.encode_image(image)
            image_feature /= image_feature.norm(dim=-1, keepdim=True)
            # return image_feature.cpu().numpy()
        # feat_m = image_feature[0].cpu().numpy()
        feat_m = image_feature.cpu().numpy()
        # similar = myclip.compute_sim(feat_m, feat_t)
        ## another_similar is official github use
        another_similar = (100.0 * feat_m @ feat_t.T)

        return text_encoded, image_feature, another_similar

    def clipChs(self,sentence, imagePath):
        # model, preprocess = load_from_name("ViT-B-16", device=device, download_root='./')
        # image = preprocess(Image.open(imagePath)).unsqueeze(0).to(device)
        image = torch.stack([self.ClipPreprocessChs(Image.open(image)) for image in imagePath]).to(self.device)
        # text = clip.tokenize(["杰尼龟", "妙蛙种子", "小火龙", "皮卡丘"]).to(device)
        text = cnClip.tokenize(sentence).to(self.device)

        with torch.no_grad():
            image_features = self.clipChs.encode_image(image)
            text_features = self.clipChs.encode_text(text)
            # 对特征进行归一化，请使用归一化后的图文特征用于下游任务
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)

            logits_per_image, logits_per_text = self.clipChs.get_similarity(image, text)
            # probs = logits_per_image.softmax(dim=-1).cpu().numpy()

        # print(logits_per_image)
        # print(logits_per_text)  # [[1.268734e-03 5.436878e-02 6.795761e-04 9.436829e-01]]
        # return logits_per_image.cpu().numpy()
        return image_features, text_features, logits_per_image

if __name__ == "__main__":
    # device = "cuda:1" if torch.cuda.is_available() else "cpu"
    # sentence = ["it's a pigfish", "it's a pig", "it's a fish", "it's a dog smiling"]
    # # sentence = ["一条猪鱼", "一只猪", "一条鱼", "一只狗在笑"]
    # image = ["./pigFish_01.jpg", "./a dog.jpg"]
    # model = CLIP_model(device)
    # text_encoded, image_feature, similar = model.get_clipEng(sentence, image)
    # print('abc')

    device = "cuda:1" if torch.cuda.is_available() else "cpu"
    image = Image.open('../co-attention/FJhl5eHVcAEPqbQ.jpg')
    model = swin_base_patch4_window12_384_model(device)
    print(model.get_feature(image).shape)