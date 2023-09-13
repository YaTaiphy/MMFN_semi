import torch
from torch import nn
from PIL import Image
from transformers import SwinModel, AutoFeatureExtractor

from MMFN_config import MMFN_config


class model_test(nn.Module):
    def __init__(self, deivce):
        super(model_test, self).__init__()
        self.pre_train_model = SwinModel.from_pretrained('../co-attention/swin-base-patch4-window12-384-in22k')
        self.pre_train_model.to(device)
        for param in self.pre_train_model.parameters():
            param.requires_grad = False

    def forward(self, inputs):
        return self.pre_train_model(inputs)

class CTCoAttentionTransformer(nn.Module):
    def __init__(self):
        super(CTCoAttentionTransformer, self).__init__()
        self.d_model = MMFN_config["d_model"]
        self.k_dim = MMFN_config["k_dim"]
        self.v_dim = MMFN_config["v_dim"]
        self.num_heads = MMFN_config["num_heads"]

        self.mmAttention = nn.MultiheadAttention(self.k_dim, self.num_heads, batch_first=True)

        self.W_k = nn.Parameter(torch.randn(self.d_model, self.k_dim))
        self.W_v = nn.Parameter(torch.randn(self.d_model, self.v_dim))
        self.W_q = nn.Parameter(torch.randn(self.d_model, self.k_dim))

        self.norm1 = nn.LayerNorm(self.k_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(self.k_dim, self.k_dim),
            nn.ReLU(),
            nn.Linear(self.k_dim, self.k_dim)
        )
        self.norm2 = nn.LayerNorm(self.k_dim)


    def forward(self, A, B):
        Q = torch.matmul(A, self.W_q)
        K = torch.matmul(B, self.W_k)
        V = torch.matmul(B, self.W_v)

        attn_output, attn_output_weights = self.mmAttention(Q, K, V)

        x = self.norm1(Q + attn_output)
        ff_output = self.feed_forward(x)
        x = self.norm2(x + ff_output)
        return x, attn_output, attn_output_weights

if __name__ == '__main__':
    device = "cuda:1" if torch.cuda.is_available() else "cpu"
    # feature_extractor = AutoFeatureExtractor.from_pretrained('../co-attention/swin-base-patch4-window12-384-in22k')
    #
    # image = Image.open('../co-attention/FJhl5eHVcAEPqbQ.jpg')
    # inputs = feature_extractor(images=image, return_tensors="pt", device=device)
    # model = model_test(device)
    #
    # for param in model.parameters():
    #     print(param.requires_grad)
    #
    # # inputs = torch.randn(1, 3, 384, 384)
    # # inputs.to(device)
    # outputs = model(inputs)
    # print(outputs.last_hidden_state.shape)


    # model = CTCoAttentionTransformer()
    # input01 = torch.rand(32, 144, 512)
    # input02 = torch.rand(32, 144, 512)
    # a, b, c = model(input01, input02)
    # print(a.shape)

    # input1 = torch.randn(100, 128)
    # input2 = torch.randn(100, 128)
    # cos = nn.CosineSimilarity(dim=1, eps=1e-6)
    # output = cos(input1, input2)

    # 创建一维数组和二维数组
    # array1d = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
    array1d = torch.tensor([1.0, 2.0])
    array2d = torch.tensor([[4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0],
                            [14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0],
                            # ... (更多行)
                            ])

    # 执行逐元素乘法
    # result = array1d.unsqueeze(0) * array2d
    result = array1d.unsqueeze(1) * array2d
    print("Result:", result)
    print("Result shape:", result.shape)  # 输出为 (10, 20)