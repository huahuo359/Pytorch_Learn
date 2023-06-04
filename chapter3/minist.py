import torch
import torchvision
from torch.utils import data
from torchvision import transforms
import matplotlib.pyplot as plt 
import torch.nn as nn 
import torch.nn.functional as F
import torch.optim as optim
# from d2l import torch as d2l

# d2l.use_svg_display()

trans = transforms.ToTensor() 
mnist_train = torchvision.datasets.FashionMNIST(
    root="../data", train=True, transform=trans, download=True
)

mnist_test = torchvision.datasets.FashionMNIST(
    root="../data", train=False, transform=trans, download=True
)


def get_fashion_mnist_labels(labels):  #@save
    """返回Fashion-MNIST数据集的文本标签"""
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                   'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]


# 可视化图像 
def show_images(imgs, num_rows, num_cols, titles=None, scale=1.5): 
    figsize = (num_cols * scale, num_rows * scale)
    _, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
    axes = axes.flatten()
    for i, (ax, img) in enumerate(zip(axes, imgs)): 
        if torch.is_tensor(img): 
            ax.imshow(img.numpy())
        else: 
            ax.imshow(img)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        if titles:
            ax.set_title(titles[i])
    return axes


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        # 分别是输入通道，输出通道，kernel_size(a, b)
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(256, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
      
        x = self.pool(F.relu(self.conv1(x)))
       
        x = self.pool(F.relu(self.conv2(x)))
        
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        
        x = F.relu(self.fc1(x))
        
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
if __name__ == '__main__':
    # X,y = next(iter(data.DataLoader(mnist_train, batch_size=18)))
    # show_images(X.reshape(18, 28, 28), 2, 9, titles=get_fashion_mnist_labels(y))
    # plt.show()
    
    train_loader = data.DataLoader(mnist_train, batch_size=36)
    test_loader = data.DataLoader(mnist_test, batch_size=18)
    
    # 训练过程
    net = Net()
        
        
# Step3: Define a Loss function and optimizer

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001, betas=(0.9, 0.99))



# Step4: Train the NetWork

    for epoch in range(2):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
        # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

        # zero the parameter gradients
            optimizer.zero_grad()

        # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                running_loss = 0.0

    print('Finished Training')
    dataiter = iter(test_loader)
    images, labels = next(dataiter)
    outputs = net(images)
    print(labels)
    # print(outputs)
    print(torch.argmax(outputs, 1))
    show_images(images.reshape(18, 28, 28), 2, 9, titles=get_fashion_mnist_labels(torch.argmax(outputs, 1)))
    plt.show()
    
    
    
        
    
    