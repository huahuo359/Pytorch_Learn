import torch 
import torch.nn.functional as F  
from torch import nn 

# Make a layer without args  

class CenteredLayer(nn.Module): 
    def __init__(self): 
        super().__init__()  
        
        
    def forward(self, X): 
        return X - X.mean()     
    

# Mat a layer with args 
class MyLinear(nn.Module): 
    def __init__(self, in_units, units):  
        super().__init__()
        self.weight = nn.Parameter(torch.randn(in_units, units))     
        self.bias = nn.Parameter(torch.randn(units, )) 
        
    def forward(self, X):  
        linear = torch.matmul(X, self.weight.data) + self.bias.data     
        return F.relu(linear)


    
if __name__ == '__main__':  
    net = nn.Sequential(nn.Linear(8, 128), CenteredLayer())
    # print(net(torch.rand(4, 8)))
    linear = MyLinear(5, 3) 
    print(linear.weight) 
    print(linear.bias)
    print(linear(torch.rand(2, 5)))
    print(torch.cuda.device_count())