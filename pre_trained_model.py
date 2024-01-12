### 该预训练模型包含以下部分：BERT，CLIP，SWIN-T；可直接使用。
import os

import torch
from PIL import Image
from transformers import SwinModel, AutoFeatureExtractor, XLNetTokenizer, XLNetModel, XLNetConfig
import clip
import numpy as np
import cn_clip.clip as cnClip
from cn_clip.clip import load_from_name
import langid
import json
class swin_base_patch4_window12_384_model(torch.nn.Module):
    def __init__(self, device):
        super(swin_base_patch4_window12_384_model, self).__init__()
        self.device = device
        self.pre_train_model = SwinModel.from_pretrained('./pre-trained-model/swin-base-patch4-window12-384-in22k')
        self.pre_train_model.to(device)
        for param in self.pre_train_model.parameters():
            param.requires_grad = False
        self.pre_train_model.eval()

        self.feature_extractor = AutoFeatureExtractor.from_pretrained('./pre-trained-model/swin-base-patch4-window12-384-in22k')

    def forward(self, inputs):
        return self.pre_train_model(**inputs)

    def get_feature_batch(self, image):
        inputs = self.feature_extractor(images=image, return_tensors="pt")
        inputs = inputs.to(self.device)
        return inputs

    def get_feature(self, images):
        inputs = self.feature_extractor(images=images, return_tensors="pt")
        inputs = inputs.to(self.device)
        outputs = self.pre_train_model(**inputs)
        return outputs


class xlnet_base_cased_model(torch.nn.Module):
    def __init__(self, max_length, device):
        super(xlnet_base_cased_model, self).__init__()

        self.max_length = max_length

        self.device = device
        self.model_config = XLNetConfig.from_pretrained('./pre-trained-model/xlnet-base-cased')
        self.pre_train_model = XLNetModel.from_pretrained('./pre-trained-model/xlnet-base-cased', config = self.model_config)
        self.pre_train_model.to(device)
        for param in self.pre_train_model.parameters():
            param.requires_grad = False
        self.pre_train_model.eval()

        self.tokenizer = XLNetTokenizer.from_pretrained('./pre-trained-model/xlnet-base-cased')

    def forward(self, inputs):
        return self.pre_train_model(inputs)

    def get_feature(self, sentence):
        sentence = '<cls>' + sentence
        tokenized_text = self.tokenizer.tokenize(sentence)[:self.max_length]
        if(len(tokenized_text) < self.max_length):
            tokenized_text = tokenized_text + ['<pad>'] * (self.max_length - len(tokenized_text))

        outputs = self.pre_train_model(torch.tensor([self.tokenizer.convert_tokens_to_ids(tokenized_text)]).to(self.device).view(1, -1))
        return outputs

    def get_feature_batch(self, sentence):
        sentence = '<cls>' + sentence
        tokenized_text = self.tokenizer.tokenize(sentence)[:self.max_length]
        if(len(tokenized_text) < self.max_length):
            tokenized_text = tokenized_text + ['<pad>'] * (self.max_length - len(tokenized_text))

        return torch.tensor([self.tokenizer.convert_tokens_to_ids(tokenized_text)]).to(self.device).view(1, -1)

class chinese_xlnet_mid_model(torch.nn.Module):
    def __init__(self, max_length, device):
        super(chinese_xlnet_mid_model, self).__init__()

        self.max_length = max_length

        self.device = device
        self.model_config = XLNetConfig.from_pretrained('./pre-trained-model/chinese-xlnet-mid')
        self.pre_train_model = XLNetModel.from_pretrained('./pre-trained-model/chinese-xlnet-mid', config = self.model_config)
        self.pre_train_model.to(device)
        for param in self.pre_train_model.parameters():
            param.requires_grad = False
        self.pre_train_model.eval()

        self.tokenizer = XLNetTokenizer.from_pretrained('./pre-trained-model/chinese-xlnet-mid')

    def forward(self, inputs):
        return self.pre_train_model(inputs)

    def get_feature(self, sentence):
        sentence = '<cls>' + sentence
        tokenized_text = self.tokenizer.tokenize(sentence)[:self.max_length]
        if(len(tokenized_text) < self.max_length):
            tokenized_text = tokenized_text + ['<pad>'] * (self.max_length - len(tokenized_text))

        outputs = self.pre_train_model(torch.tensor([self.tokenizer.convert_tokens_to_ids(tokenized_text)]).to(self.device).view(1, -1))
        return outputs

    def get_feature_batch(self, sentence):
        sentence = '<cls>' + sentence
        tokenized_text = self.tokenizer.tokenize(sentence)[:self.max_length]
        if(len(tokenized_text) < self.max_length):
            tokenized_text = tokenized_text + ['<pad>'] * (self.max_length - len(tokenized_text))

        return torch.tensor([self.tokenizer.convert_tokens_to_ids(tokenized_text)]).to(self.device).view(1, -1)


def get_clipEng(sentence, imagePath, device):
    clipEng, clipPreprocessEng = clip.load("./pre-trained-model/ViT-B-32.pt", device=device)

    with torch.no_grad():
        text_encoded = clipEng.encode_text(clip.tokenize(sentence, 77, True).to(device)).to(device)
        # text_encoded = myclip.encode_text(clip.tokenize(sentence).to(device))
        text_encoded /= text_encoded.norm(dim=-1, keepdim=True)
    # feat_t = text_encoded[0].cpu().numpy()
    feat_t = text_encoded.cpu().numpy()
    with torch.no_grad():
        image = torch.stack([clipPreprocessEng(Image.open(image)) for image in imagePath]).to(device)
        # image = preprocess(Image.open(imagePath)).unsqueeze(0).to(device)
        image_feature = clipEng.encode_image(image)
        image_feature /= image_feature.norm(dim=-1, keepdim=True)
        # return image_feature.cpu().numpy()
    # feat_m = image_feature[0].cpu().numpy()
    feat_m = image_feature.cpu().numpy()
    # similar = myclip.compute_sim(feat_m, feat_t)
    ## another_similar is official github use
    another_similar = (100.0 * feat_m @ feat_t.T)

    return image_feature, text_encoded, torch.matmul(image_feature, text_encoded.t())

device = "cuda:1" if torch.cuda.is_available() else "cpu"
clipChs, ClipPreprocessChs = load_from_name("ViT-B-16", device=device,
                                                          download_root='./pre-trained-model/')
def get_clipChs(sentence, imagePath, device):
    # model, preprocess = load_from_name("ViT-B-16", device=device, download_root='./')
    # image = preprocess(Image.open(imagePath)).unsqueeze(0).to(device)
    image = torch.stack([ClipPreprocessChs(Image.open(image)) for image in imagePath]).to(device)
    # text = clip.tokenize(["杰尼龟", "妙蛙种子", "小火龙", "皮卡丘"]).to(device)
    text = cnClip.tokenize(sentence).to(device)

    with torch.no_grad():
        image_features = clipChs.encode_image(image)
        text_features = clipChs.encode_text(text)
        # 对特征进行归一化，请使用归一化后的图文特征用于下游任务
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        logits_per_image, logits_per_text = clipChs.get_similarity(image, text)
        # probs = logits_per_image.softmax(dim=-1).cpu().numpy()

    # print(logits_per_image)
    # print(logits_per_text)  # [[1.268734e-03 5.436878e-02 6.795761e-04 9.436829e-01]]
    # return logits_per_image.cpu().numpy()
    return image_features, text_features, logits_per_image

def spliceJudge(items1, items2):
    column1 = float((abs(items1[0][1] - items1[3][1]) + abs(items1[1][1] - items1[3][1])))/2
    column2 = float((abs(items2[0][1] - items2[3][1]) + abs(items2[1][1] - items2[3][1])))/2
    column_X = float((abs(items2[0][1] - items1[3][1]) + abs(items2[1][1] - items1[3][1])))/2

    if max(column2, column1)/min(column2, column1) < 1.3 and column_X < (column1 + column2)*1.5:
        return True
    return False

def ocr_word(image, reader):
    result = reader.readtext(image)

    words = ''
    pres = ''
    for items in result:
        ## 行距测验，判断处理
        if(isinstance(pres, str)):
            words = words + ' ' + items[1]
            pres = items
            continue
        if(spliceJudge(items[0], pres[0])):
            words = words + items[1]
        else:
            words = words + '。' + items[1]
        pres = items

    return words

if __name__ == "__main__":

    # device = "cuda:1" if torch.cuda.is_available() else "cpu"
    # sentence = ["it's a pigfish", "it's a pig", "it's a fish", "it's a dog smiling"]
    # sentence = ["一条猪鱼", "一只猪", "一条鱼", "一只狗在笑"]
    # image = ["./temp/pigFish_01.jpg", "./temp/a dog.jpg"]
    # text_encoded, image_feature, similar = get_clipChs(sentence, image, device)
    # print('abc')

    # device = "cuda:1" if torch.cuda.is_available() else "cpu"
    # image = Image.open('../co-attention/FJhl5eHVcAEPqbQ.jpg')
    # model = swin_base_patch4_window12_384_model(device)
    # print(model.get_feature(image))


    # sentence = ["it's a pigfish", "it's a pig", "it's a fish", "it's a dog smiling"]
    # sentence = ["一条猪鱼", "一只猪", "一条鱼", "一只狗在笑"]
    # device = "cuda:1" if torch.cuda.is_available() else "cpu"
    # model = chinese_xlnet_mid_model(77, device)
    # print(model.get_feature_batch(sentence))

    # weibo16 = open('./data/weibo16.txt', 'r', encoding='utf-8')
    # weibo16PathRealPic = '/home/lxy/FakeNews_data/new/weibo16/nonrumor_images/'
    # weibo16PathFakePic = '/home/lxy/FakeNews_data/new/weibo16/rumor_images/'
    # weibo21PathPic = '/home/lxy/FakeNews_data/new/weibo21/pic'
    #
    # PathAllPic = []
    # PathAllPic.append(weibo16PathRealPic)
    # PathAllPic.append(weibo16PathFakePic)
    # PathAllPic.append(weibo21PathPic)
    #
    #
    # import easyocr
    # reader = easyocr.Reader(['ch_sim'])
    # for sub_path in PathAllPic:
    #     for file in os.listdir(sub_path):
    #         if file.endswith('.jpg'):
    #             image_path = os.path.join(sub_path, file)
    #
    #             txt = ocr_word(image_path, reader)
    #
    #             save_path = os.path.join(sub_path, file.split('.')[0] + '.txt')
    #
    #             print(save_path)
    #             with open(save_path, 'w', encoding='utf-8') as f:
    #                 f.write(txt)
    #
    #             f.close()


    weibo16Path = '/home/lxy/FakeNews_data/new/weibo16/'
    weibo16PathSave = './data/weibo16/'
    weibo21Path = '/home/lxy/FakeNews_data/new/weibo21/'
    weibo21PathSave = './data/weibo21/'

    save_path = './data/'

    ### read json file and save feature
    ### and here is a bad code, but I don't want to change it
    ### however, the only thing I want to say is that FUCK ***
    def read_json_and_save_feature(json_file, save_base_path):
        f = open(json_file, 'r', encoding='utf-8')
        lines = f.readlines()
        f.close()

        clip_features_text = []
        clip_features_image = []

        xlnet_features = []
        swinT_features = []

        device = "cuda:1" if torch.cuda.is_available() else "cpu"
        swinTmodel = swin_base_patch4_window12_384_model(device)
        xlnetModel = chinese_xlnet_mid_model(144, device)


        for line in lines:
            all_data = json.loads(line)

            text = all_data["content"]
            images = all_data["piclists"]
            char_to_remove = "null"
            # 使用循环迭代列表并删除包含指定字符的元素
            images = [item for item in images if char_to_remove not in item]

            ## get features
            for image in images:
                try:
                    sentence = []
                    imageP = []
                    sentence.append(text)
                    imageP.append(os.path.split(json_file)[0]+'/pic/' + image)
                    text_encoded, image_feature, similar = get_clipChs(sentence, imageP, device)

                    image = Image.open(os.path.split(json_file)[0]+'/pic/' + image)
                    swinT_feature = swinTmodel.get_feature(image).last_hidden_state

                    xlnet_feature = xlnetModel.get_feature(text).last_hidden_state

                    clip_features_text.append(text_encoded.to('cpu').detach().numpy().reshape(-1))
                    clip_features_image.append(image_feature.to('cpu').detach().numpy().reshape(-1))
                    swinT_features.append(swinT_feature.to('cpu').detach().numpy().reshape(-1))
                    xlnet_features.append(xlnet_feature.to('cpu').detach().numpy().reshape(-1))
                except:
                    continue
            break

        # np.array(AAAI2vector_sentences).tofile('./AAAI_xlnet_larger_cls_01.bin')
        np.array(clip_features_text).tofile(save_base_path + '/' + os.path.splitext(os.path.split(json_file)[1])[0] + '_clip_features_text.bin')
        np.array(clip_features_image).tofile(save_base_path + '/' + os.path.splitext(os.path.split(json_file)[1])[0] + '_clip_features_image.bin')
        np.array(swinT_features).tofile(save_base_path + '/' + os.path.splitext(os.path.split(json_file)[1])[0] + '_swinT_features.bin')
        np.array(xlnet_features).tofile(save_base_path + '/' + os.path.splitext(os.path.split(json_file)[1])[0] + '_xlnet_features.bin')

    ### get json file
    def get_json_file(path):
        json_file = []
        for file in os.listdir(path):
            if file.endswith('.json') and 'select' in file:
                json_file.append(file)
        return json_file


    def execute_path(file_path, save_base_path):
        json_files = get_json_file(file_path)

        for json_file in json_files:
            print(json_file)
            read_json_and_save_feature(os.path.join(file_path, json_file), save_base_path)

    execute_path(weibo16Path, weibo16PathSave)
    execute_path(weibo21Path, weibo21PathSave)
