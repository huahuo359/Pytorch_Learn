import torch 
import torchvision 
import torchvision.transforms as transforms
import matplotlib.pyplot as plt 
import numpy as np 
import torch.nn as nn 
import torch.nn.functional as F
import torch.optim as optim


# idea: 由多个块作为组件构建出深度神经网络  

class MLP(nn.Module): 
    
    def __init__(self):  
        super().__init__() 
        self.hidden = nn.Linear(20, 256)
        self.out = nn.Linear(256, 10)  
        
    def forward(self, X):  
        return self.out(F.relu(self.hidden(X))) 
    
    
class MySequential(nn.Module):  
    """将多个 block 串连在一起 

    Args:
        nn (_type_): _description_
    """
    def __init__(self, *args): 
        super().__init__() 
        for index, module in enumerate(args): 
            # use map to store NetWork Block 
            self._modules[str(index)] = module  
            
    def forward(self, X):  
        for block in self._modules.values(): 
            X = block(X)  
        
        return X 
   
   
class FixedHiddenMLP(nn.Module):
    # 可以在 farward 的过程中执行自己的操作
    # 这个例子在实际中或许不会被用到 
    def __init__(self): 
        super().__init__()
        # 不计算梯度的随机权重参数。因此其在训练期间保持不变
        self.rand_weight = torch.rand((20, 20), requires_grad=False)
        self.linear = nn.Linear(20, 20)
        
    def forward(self, X):
        X = self.linear(X)
        # 使⽤创建的常量参数以及relu和mm函数
        X = F.relu(torch.mm(X, self.rand_weight) + 1)
        # 复⽤全连接层。这相当于两个全连接层共享参数
        X = self.linear(X)
        # 控制流
        while X.abs().sum() > 1:
            X /= 2
        return X.sum()
 
    
if __name__ == '__main__':  
    X = torch.rand(2, 20)
    net = MySequential(nn.Linear(20, 256), nn.ReLU(), nn.Linear(256, 10))
    chimera = nn.Sequential(nn.Linear(16, 20), FixedHiddenMLP())
    print(net(X))
   