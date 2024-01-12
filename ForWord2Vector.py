import os
import json
import gensim
from gensim.models import Word2Vec
import re
import jieba
import zhconv
import numpy as np

# -i https://pypi.tuna.tsinghua.edu.cn/simple
## 中文使用jieba进行分词，仅对url进行去除
def clear_sentence_chs(text):       #对单句文本的清理
    # text = re.sub(r"(回复)?(//)?\s*@\S*?\s*(:| |$)", " ", text)  # 去除正文中的@和回复/转发中的用户名
    results = re.compile(r'https?://[a-zA-Z0-9.?/&=:]*', re.S)  # 对数据进行预处理，去除url
    text = results.sub("", text)
    
    text = text.strip()  #去前后的空格
    #干掉所有麻烦字符的终极武器：非中文、英文、表情的都干掉
    sen_text = re.compile(u'[\u4E00-\u9FA5|\s\w]').findall(text)
    text = "".join(sen_text)

    
    text = zhconv.convert(text, 'zh-cn')  # 中文繁体转简体
    regex2 = jieba.cut(text, cut_all=False)
    regex2 = ' '.join(regex2)
    regex2 = regex2.split()
    return regex2


weibo16Path = '/home/lxy/FakeNews_data/new/weibo16/'
weibo16PathSave = './data/weibo16/'
weibo21Path = '/home/lxy/FakeNews_data/new/weibo21/'
weibo21PathSave = './data/weibo21/'

save_path = './data/'



def read_json_and_save_feature(json_file, save_base_path):
    f = open(json_file, 'r', encoding='utf-8')
    lines = f.readlines()
    f.close()

    for line in lines:
        all_data = json.loads(line)

        text = all_data["content"]
        # images = all_data["piclists"]
        # char_to_remove = "null"
        # # 使用循环迭代列表并删除包含指定字符的元素
        # images = [item for item in images if char_to_remove not in item]

            

### get json file
def get_json_file(path):
    json_file = []
    for file in os.listdir(path):
        if file.endswith('.json') and 'select' in file:
            json_file.append(file)
    return json_file



def read_weibo():
    ## 数据类型构造如下：文字，图片链接，label 1 true 0 false
    
    weibo_datasets = []
    information=[["<unk>"]]
    max_length=0
    json_files = get_json_file(weibo16Path)
    for json_file in json_files:
        f = open(os.path.join(weibo16Path, json_file), 'r', encoding='utf-8')
        lines = f.readlines()
        f.close()
        for line in lines:
            all_data = json.loads(line)
            text = all_data["content"]
            information.append(clear_sentence_chs(text))
            if(len(clear_sentence_chs(text)) > max_length):
                max_length = len(clear_sentence_chs(text))



    # def weibo2vector():
    #     weibo2vector_sentences = [["<unk>"]]
    #     datasets = read_weibo()
    #     for m in datasets:
    #         weibo2vector_sentences.append(m[0])
    # sentences = [["<unk>"], ["cat", "say", "meow"], ["dog", "say", "woof"]]
    model = Word2Vec(information, min_count=1, vector_size=32)
    model.wv.save_word2vec_format('./model/word2vec_weibo.bin', binary=True)
    print(max_length)


def text_process(text):
    model = gensim.models.KeyedVectors.load_word2vec_format('./model/word2vec_weibo.bin', binary=True)
    max_length=120

    text_vector=[]
    cout = 0
    for word in text:
        if(cout >= max_length):
            break
        try:
            b = model[word]
        except Exception:
            b = model["<unk>"]
        text_vector.append(b)
        cout=cout+1
    while cout < max_length:
        b = model["<unk>"]
        text_vector.append(b)
        cout=cout+1
        
    text_vector = np.array(text_vector)
    return text_vector

if __name__ == "__main__":
    # read_weibo()
    text_process("上海二手房均价一千元人民币")