### following code is from https://blog.csdn.net/cyj972628089/article/details/127633915

import torch
import torch.nn as nn
import numpy as np


class Expert(nn.Module):
    def __init__(self,input_dim,output_dim): #input_dim代表输入维度，output_dim代表输出维度
        super(Expert, self).__init__()
        
        p=0
        expert_hidden_layers = [64,32]
        self.expert_layer = nn.Sequential(
                            nn.Linear(input_dim, expert_hidden_layers[0]),
                            nn.ReLU(),
                            nn.Dropout(p),
                            nn.Linear(expert_hidden_layers[0], expert_hidden_layers[1]),
                            nn.ReLU(),
                            nn.Dropout(p),
                            nn.Linear(expert_hidden_layers[1],output_dim),
                            nn.ReLU(),
                            nn.Dropout(p)
                            )  

    def forward(self, x):
        out = self.expert_layer(x)
        return out

class Expert_Gate(nn.Module):
    def __init__(self,feature_dim,expert_dim,n_expert,n_task,use_gate=True): #feature_dim:输入数据的维数  expert_dim:每个神经元输出的维数  n_expert:专家数量  n_task:任务数(gate数)  use_gate：是否使用门控，如果不使用则各个专家取平均
        super(Expert_Gate, self).__init__()
        self.n_task = n_task
        self.use_gate = use_gate
        
        
        ## setattr作用：
        ## getattr作用：
        '''专家网络'''
        for i in range(n_expert):
            setattr(self, "expert_layer"+str(i+1), Expert(feature_dim,expert_dim)) 
        self.expert_layers = [getattr(self,"expert_layer"+str(i+1)) for i in range(n_expert)]#为每个expert创建一个DNN
        
        '''门控网络'''
        for i in range(n_task):
            setattr(self, "gate_layer"+str(i+1), nn.Sequential(nn.Linear(feature_dim, n_expert),
                                        					   nn.Softmax(dim=1))) 
        self.gate_layers = [getattr(self,"gate_layer"+str(i+1)) for i in range(n_task)]#为每个gate创建一个lr+softmax
        
    def forward(self, x):
        if self.use_gate:
            # 构建多个专家网络
            E_net = [expert(x) for expert in self.expert_layers]
            E_net = torch.cat(([e[:,np.newaxis,:] for e in E_net]),dim = 1) # 维度 (bs,n_expert,expert_dim)

            # 构建多个门网络
            gate_net = [gate(x) for gate in self.gate_layers]     # 维度 n_task个(bs,n_expert)

            # towers计算：对应的门网络乘上所有的专家网络
            towers = []
            for i in range(self.n_task):
                g = gate_net[i].unsqueeze(2)  # 维度(bs,n_expert,1)
                tower = torch.matmul(E_net.transpose(1,2),g)# 维度 (bs,expert_dim,1)
                towers.append(tower.transpose(1,2).squeeze(1))           # 维度(bs,expert_dim)
        else:
            E_net = [expert(x) for expert in self.expert_layers]
            towers = sum(E_net)/len(E_net)
        return towers

class MMoE(nn.Module):
	#feature_dim:输入数据的维数  expert_dim:每个神经元输出的维数  n_expert:专家数量  n_task:任务数(gate数)
    def __init__(self,feature_dim,expert_dim,n_expert,n_task,use_gate=True): 
        super(MMoE, self).__init__()
        
        self.use_gate = use_gate
        self.Expert_Gate = Expert_Gate(feature_dim=feature_dim,expert_dim=expert_dim,n_expert=n_expert,n_task=n_task,use_gate=use_gate)
        
        '''Tower1'''
        p1 = 0 
        hidden_layer1 = [64,32] #[64,32] 
        self.tower1 = nn.Sequential(
            nn.Linear(expert_dim, hidden_layer1[0]),
            nn.ReLU(),
            nn.Dropout(p1),
            nn.Linear(hidden_layer1[0], hidden_layer1[1]),
            nn.ReLU(),
            nn.Dropout(p1),
            nn.Linear(hidden_layer1[1], 1))
        '''Tower2'''
        p2 = 0
        hidden_layer2 = [64,32]
        self.tower2 = nn.Sequential(
            nn.Linear(expert_dim, hidden_layer2[0]),
            nn.ReLU(),
            nn.Dropout(p2),
            nn.Linear(hidden_layer2[0], hidden_layer2[1]),
            nn.ReLU(),
            nn.Dropout(p2),
            nn.Linear(hidden_layer2[1], 1))
        
    def forward(self, x):
        ## lxy:模型大概只需要Expert_Gate这一部分的结果，如需控制expert输出维度，见Expert类中的expert_hidden_layers部分
        towers = self.Expert_Gate(x)
        if self.use_gate:            
            out1 = self.tower1(towers[0])
            out2 = self.tower2(towers[1]) 
        else:
            out1 = self.tower1(towers)
            out2 = self.tower2(towers)
        
        return out1,out2
    
if __name__ == '__main__':
    Model = MMoE(feature_dim=128,expert_dim=32,n_expert=4,n_task=4,use_gate=True)

    input_test = torch.randn(32,128)
    output = Model(input_test)
    nParams = sum([p.nelement() for p in Model.parameters()])
    print('* number of parameters: %d' % nParams)
