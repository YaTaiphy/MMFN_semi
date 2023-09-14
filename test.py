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

def testBertChs():
    max_length = 144
    from transformers import BertTokenizer, BertModel, BertConfig
    tokenizer = BertTokenizer.from_pretrained('./pre-trained-model/bert-base-chinese')
    model_config = BertConfig.from_pretrained('./pre-trained-model/bert-base-chinese')
    model = BertModel.from_pretrained('./pre-trained-model/bert-base-chinese', config = model_config)

    sentence = '一条小狗'
    sentence = tokenizer.tokenize(sentence)
    sentence = ['<CLS>'] + sentence
    tokenized_text = sentence[:max_length]
    if (len(tokenized_text) < max_length):
        tokenized_text = tokenized_text + ['<PAD>'] * (max_length - len(tokenized_text))

    input_ids = torch.tensor([tokenizer.convert_tokens_to_ids(tokenized_text)]).view(1,-1)  # Batch size 1
    outputs = model(input_ids)
    last_hidden_states = outputs[0]  # The last hidden-state is the first element of the output tuple
    print(last_hidden_states.shape)


if __name__ == '__main__':
    # test20230912()
    testBertChs()