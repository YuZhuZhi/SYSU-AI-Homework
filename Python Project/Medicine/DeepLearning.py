import torch
from tqdm import *
import torch.nn.functional as F
from torchvision.datasets import ImageFolder
from torch import nn
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

class NeuralNet(nn.Module):
    def __init__(self) -> None:
        super(NeuralNet, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels = 3,
            out_channels = 6,
            kernel_size = 5,
            stride = 1,
        )
        self.conv2 = nn.Conv2d(
            in_channels = 6,
            out_channels = 16,
            kernel_size = 5,
            stride = 1,
        )
        self.fc1 = nn.Linear(16 * 53 * 53, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 5)
        self.pool = nn.MaxPool2d(kernel_size = 2)
        
    def forward(self, x: torch.tensor) -> torch.tensor:
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(-1, 16 * 53 * 53)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
        
class MedicineClassify:
    def __init__(self) -> None:
        transformation = transforms.Compose([
            transforms.Resize([224, 224]), # 图片裁剪为224 * 224
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225])
        ])
        self.iteration = 100
        self.loss_per_iteration, self.accuracy_per_iteration = [], []
        self.train_set = ImageFolder("./train", transform = transformation)
        self.test_set = ImageFolder("./test", transform = transformation)
        self.train_loader = torch.utils.data.DataLoader(self.train_set, batch_size = 4, shuffle = True, num_workers = 0)
        self.test_loader = torch.utils.data.DataLoader(self.test_set, batch_size = 4, shuffle = True, num_workers = 0)
        self.NeuralNet = NeuralNet()
        self.Loss = nn.CrossEntropyLoss()
        self.Optimizer = torch.optim.SGD(self.NeuralNet.parameters(), lr = 1e-3)
    
    def Train(self) -> None:
        """训练函数"""
        self.NeuralNet.train() # 训练模式
        for batch, (X, y) in enumerate(self.train_loader):
            loss = self.Loss(self.NeuralNet(X), y) # 计算损失
            self.Optimizer.zero_grad() # 梯度清零
            loss.backward() # 反向传播
            self.Optimizer.step() # 优化器更新
    
    def Test(self) -> None:
        """测试函数"""
        self.NeuralNet.eval() # 评估模式
        size = len(self.test_loader.dataset)
        num_batches = len(self.test_loader)
        test_loss, correct = 0.0, 0.0
        with torch.no_grad():
            for X, y in self.test_loader:
                pred = self.NeuralNet(X)
                test_loss = test_loss + self.Loss(pred, y).item() # 计算损失
                correct = correct + (pred.argmax(1) == y).type(torch.float).sum().item() # 累加正确分类数
        self.loss_per_iteration.append(test_loss / num_batches)
        self.accuracy_per_iteration.append(correct / size)
        
    def Draw(self) -> None:
        """绘制函数"""
        plt.subplots(1, 2, figsize = (20, 5))
        plt.subplot(1, 2, 1)
        plt.plot(self.loss_per_iteration)
        plt.xlabel("迭代次数")
        plt.ylabel("损失")
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        _ = plt.title("神经网络每次迭代中的损失")
        plt.grid(True)
        plt.subplot(1, 2, 2)
        plt.plot(self.accuracy_per_iteration)
        plt.xlabel("迭代次数")
        plt.ylabel("准确率")
        _ = plt.title("神经网络每次迭代中的准确率")
        plt.grid(True)
        plt.show()
        
    def Solve(self) -> None:
        """对外的解决方法"""
        for i in trange(0, self.iteration):
            self.Train()
            self.Test()
        self.Draw()
        
medicineClassify = MedicineClassify()
medicineClassify.Solve()