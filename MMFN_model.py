import torch

from torch import nn

from MMFN_config import MMFN_config
from MMoE_model import MMoE_Expert_Gate


class MMFN_semi_Texture_Branch(torch.nn.Module):
    def __init__(self, device):
        super(MMFN_semi_Texture_Branch, self).__init__()
        self.device = device

        self.projectionHead = nn.Sequential(
            # nn.Linear(MMFN_config["XLNET_size"] + MMFN_config["CLIP_size"], MMFN_config["XLNET_size"] + MMFN_config["CLIP_size"]),
            nn.Linear(MMFN_config["expert_dim"]*2, MMFN_config["expert_dim"]*2),
            # nn.ReLU(),
            # nn.Linear(MMFN_config["XLNET_size"] + MMFN_config["CLIP_size"], 512),
            nn.Linear(MMFN_config["expert_dim"]*2, 512),
            # nn.ReLU(),
            nn.Linear(512, 256),
            # nn.ReLU(),
        )

    def forward(self, inputs_lstm, inputs_clip):
        avg_inputs_lstm = nn.functional.avg_pool1d(inputs_lstm.permute(0, 2, 1), kernel_size=MMFN_config["w2v_length"])
        avg_inputs_lstm = avg_inputs_lstm.reshape(inputs_lstm.shape[0], -1)

        combine_feature = torch.cat((avg_inputs_lstm, inputs_clip), dim=1)
        return self.projectionHead(combine_feature)

class MMFN_semi_Visual_Branch(torch.nn.Module):
    def __init__(self, device):
        super(MMFN_semi_Visual_Branch, self).__init__()
        self.device = device

        self.projectionHead = nn.Sequential(
            # nn.Linear(MMFN_config["SWIN_size"] + MMFN_config["CLIP_size"], MMFN_config["SWIN_size"] + MMFN_config["CLIP_size"]),
            nn.Linear(MMFN_config["expert_dim"]*2, MMFN_config["expert_dim"]*2),
            # nn.ReLU(),
            # nn.Linear(MMFN_config["SWIN_size"] + MMFN_config["CLIP_size"], 512),
            nn.Linear(MMFN_config["expert_dim"]*2, 512),
            # nn.ReLU(),
            nn.Linear(512, 256),
            # nn.ReLU(),
        )

    def forward(self, inputs_swin, inputs_clip):
        avg_inputs_swin = nn.functional.avg_pool1d(inputs_swin.permute(0, 2, 1), kernel_size=MMFN_config["SWIN_max_length"])
        avg_inputs_swin = avg_inputs_swin.reshape(inputs_swin.shape[0], -1)

        combine_feature = torch.cat((avg_inputs_swin, inputs_clip), dim=1)
        return self.projectionHead(combine_feature)

class CTCoAttentionTransformer(nn.Module):
    def __init__(self, device):
        super(CTCoAttentionTransformer, self).__init__()
        self.device = device
        self.d_model = MMFN_config["d_model"]
        self.k_dim = MMFN_config["k_dim"]
        self.v_dim = MMFN_config["v_dim"]
        self.num_heads = MMFN_config["num_heads"]

        # pytorch 2.1和 1.8写法区别·
        # self.mmAttention = nn.MultiheadAttention(embed_dim=self.k_dim, num_heads=self.num_heads, batch_first=True)
        self.mmAttention = nn.MultiheadAttention(embed_dim=self.k_dim, num_heads=self.num_heads)
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


# 定义一个简单的 LSTM 模型
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(LSTMModel, self).__init__()

        # LSTM 层
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

        # 全连接层
        # self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # LSTM 前向传播
        out, _ = self.lstm(x)

        # 提取最后一个时间步的输出，作为 LSTM 的表示
        # out = out[:, -1, :]

        # # 全连接层
        # out = self.fc(out)

        return out


class Multi_grained_feature_fusion(torch.nn.Module):
    def __init__(self, device):
        super(Multi_grained_feature_fusion, self).__init__()
        self.device = device

        self.k_dim = MMFN_config["k_dim"]
        # self.LinearTexture = nn.Linear(MMFN_config["XLNET_size"], MMFN_config["d_model"])
        # self.LinearImage = nn.Linear(MMFN_config["SWIN_size"], MMFN_config["d_model"])
        self.LinearTexture = nn.Linear(MMFN_config["expert_dim"], MMFN_config["d_model"])
        self.LinearImage = nn.Linear(MMFN_config["expert_dim"], MMFN_config["d_model"])

        self.CoAttentionTI = CTCoAttentionTransformer(device)
        self.CoAttentionIT = CTCoAttentionTransformer(device)

        self.feed_forward01 = nn.Sequential(
            nn.Linear(self.k_dim * 2, self.k_dim * 2),
            nn.ReLU(),
            nn.Linear(self.k_dim * 2, self.k_dim * 2)
        )

        self.feed_forward02 = nn.Sequential(
            # nn.Linear(MMFN_config["CLIP_size"] * 2, MMFN_config["CLIP_size"] * 2),
            nn.Linear(MMFN_config["expert_dim"] * 2, MMFN_config["expert_dim"] * 2),
            nn.ReLU(),
            # nn.Linear(MMFN_config["CLIP_size"] * 2, MMFN_config["CLIP_size"] * 2)
            nn.Linear(MMFN_config["expert_dim"] * 2, MMFN_config["expert_dim"] * 2),
        )

        self.projectionHead = nn.Sequential(
            # nn.Linear(self.k_dim * 2 + MMFN_config["CLIP_size"] * 2, 1024),
            nn.Linear(self.k_dim * 2 + MMFN_config["expert_dim"] * 2, 1024),
            nn.Linear(1024, 512),
            nn.Linear(512, 512)
        )

        self.cos_clip = nn.CosineSimilarity(dim=1, eps=1e-6)

    def forward(self, inputs_lstm, inputs_swin, inputs_clip_text, inputs_clip_img):
        inputs_lstm = self.LinearTexture(inputs_lstm)
        inputs_swin = self.LinearImage(inputs_swin)

        output_lstm, _, _ = self.CoAttentionTI(inputs_lstm, inputs_swin)
        output_swin, _, _ = self.CoAttentionIT(inputs_swin, inputs_lstm)

        output_lstm = nn.functional.avg_pool1d(output_lstm.permute(0, 2, 1), kernel_size=MMFN_config["w2v_length"])
        output_lstm = output_lstm.reshape(output_lstm.shape[0], -1)

        output_swin = nn.functional.avg_pool1d(output_swin.permute(0, 2, 1), kernel_size=MMFN_config["SWIN_max_length"])
        output_swin = output_swin.reshape(output_swin.shape[0], -1)

        output_transformer = torch.cat((output_lstm, output_swin), dim=1)

        output_transformer = self.feed_forward01(output_transformer)


        Cos_Clip = self.cos_clip(inputs_clip_text, inputs_clip_img)

        output_clip = torch.cat((inputs_clip_text, inputs_clip_img), dim=1)
        output_clip = self.feed_forward02(output_clip)

        combine_features = torch.cat((output_transformer, output_clip), dim=1)

        combine_features = self.projectionHead(combine_features)

        multi_model_output = Cos_Clip.unsqueeze(1) * combine_features

        return multi_model_output


class MMFN_classifier(torch.nn.Module):
    def __init__(self, device):
        super(MMFN_classifier, self).__init__()
        self.device = device

        self.lstm = LSTMModel(
            input_size = MMFN_config["w2v_size"],
            hidden_size = MMFN_config["hidden_size"],
            num_layers = MMFN_config["num_layers"]
        )

        self.modelMoE_lstm = MMoE_Expert_Gate(
            feature_dim = MMFN_config["hidden_size"],
            expert_dim = MMFN_config['expert_dim'],
            n_expert = MMFN_config['n_expert'],
            n_task = 2
            )
        
        self.modelMoE_Swin = MMoE_Expert_Gate(
            feature_dim = MMFN_config['SWIN_size'],
            expert_dim = MMFN_config['expert_dim'],
            n_expert = MMFN_config['n_expert'],
            n_task = 2
            )
        
        self.modelMoE_CLIPT = MMoE_Expert_Gate(
            feature_dim = MMFN_config['CLIP_size'],
            expert_dim = MMFN_config['expert_dim'],
            n_expert = MMFN_config['n_expert'],
            n_task = 2
            )
        
        self.modelMoE_CLIPV = MMoE_Expert_Gate(
            feature_dim = MMFN_config['CLIP_size'],
            expert_dim = MMFN_config['expert_dim'],
            n_expert = MMFN_config['n_expert'],
            n_task = 2
            )
        
        self.modelTB = MMFN_semi_Texture_Branch(device)
        self.modelVB = MMFN_semi_Visual_Branch(device)
        self.modelMFF = Multi_grained_feature_fusion(device)
        
        self.Linear = nn.Linear(1024, 2)

    def forward(self, inputs_lstm, inputs_swin, inputs_clip_text, inputs_clip_img):
        batch_size = inputs_lstm.shape[0]
        
        inputs_lstm = self.lstm(inputs_lstm)
        
        TA_lstm = self.modelMoE_lstm(inputs_lstm.contiguous().view(-1, MMFN_config["hidden_size"]))
        TA_lstm = [single.view(batch_size, MMFN_config["w2v_length"], -1) for single in TA_lstm]
        
        TA_swin = self.modelMoE_Swin(inputs_swin.view(-1, MMFN_config["SWIN_size"]))
        TA_swin = [single.view(batch_size, MMFN_config["SWIN_max_length"], -1) for single in TA_swin]
        
        TA_clipT = self.modelMoE_CLIPT(inputs_clip_text)
        # TA_clipT = [single.view(batch_size, MMFN_config["CLIP_size"], -1) for single in TA_clipT]
        
        TA_clipV = self.modelMoE_CLIPV(inputs_clip_img)
        # TA_clipV = [single.view(batch_size, MMFN_config["CLIP_size"], -1) for single in TA_clipV]
        
        
        output_TB = self.modelTB(TA_lstm[0], TA_clipT[0])
        output_VB = self.modelVB(TA_swin[0], TA_clipV[0])
        output_MFF = self.modelMFF(TA_lstm[1], TA_swin[1], TA_clipT[1], TA_clipV[1])

        output = torch.cat((output_TB, output_VB, output_MFF), dim=1)

        output = self.Linear(output)

        output = torch.softmax(output, dim=1)

        return output

if __name__ == '__main__':
    # import torch
    # import torch.nn.functional as F
    #
    # # 创建一个示例输入张量（维度为 1x77x768）
    # input_tensor = torch.randn(1, 77, 768)
    #
    # # 在第二维进行 1D 平均池化操作，将输入转换为 1x1x768 形式
    # output_tensor = F.avg_pool1d(input_tensor.permute(0, 2, 1), kernel_size=77)
    #
    # output = output_tensor.reshape(input_tensor.shape[0], -1)
    # # 打印输出张量
    # print(output_tensor)
    # print(output_tensor.shape)

    # model = MMFN_semi_Texture_Branch("cuda:1")
    # input_tensor = torch.randn(32, 77, 768)
    # input_tensor2 = torch.randn(32, 512)
    # output = model(input_tensor, input_tensor2)
    # print(output.shape)

    # model = Multi_grained_feature_fusion("cuda:1")
    # input_xlnet = torch.randn(32, 144, 768)
    # input_swin = torch.randn(32, 144, 1024)
    # input_clip_text = torch.randn(32, 512)
    # input_clip_img = torch.randn(32, 512)
    # output = model(input_xlnet, input_swin, input_clip_text, input_clip_img)
    # print(output.shape)

    device = "cuda:1" if torch.cuda.is_available() else "cpu"
    model = MMFN_classifier("cuda:1")
    model.to("cuda:1")
    input_lstm = torch.randn(32, 144, 32).to(device)
    input_swin = torch.randn(32, 144, 1024).to(device)
    input_clip_text = torch.randn(32, 512).to(device)
    input_clip_img = torch.randn(32, 512).to(device)
    output = model(input_lstm, input_swin, input_clip_text, input_clip_img)
    print(output.shape)