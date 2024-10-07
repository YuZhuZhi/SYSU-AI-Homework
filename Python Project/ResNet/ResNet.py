import gzip
from tqdm import *
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn.functional as F
import torchvision.datasets as datasets
from torchvision.datasets import ImageFolder
from torch import nn
import torchvision.transforms as transforms

#************************************************************************************************************************************#

"""常量定义区"""
DATA_SCALE = 1000 # 训练数据集的规模
TEST_SCALE = 500 # 测试数据集的规模
ITERATION = 150 # 迭代的次数

#************************************************************************************************************************************#

class DataProcess(torch.utils.data.dataset.Dataset):
    """生成数据集"""
    # 为了从大规模数据集中截取一部分
    def __init__(self, data_addr: str, label_addr: str, transformation: transforms.Compose):
        self.data = self.parseMNIST(data_addr)
        self.label = self.parseMNIST(label_addr)
        self.transform = transformation
    
    def __getitem__(self, index: int) -> tuple:
        img, target = self.transform(Image.fromarray(self.data[index])), int(self.label[index])
        return (img, target)
    
    def __len__(self) -> int:
        return len(self.label)
    
    def parseMNIST(self, file_addr: str) -> np.ndarray:
        """解析MNIST文件"""
        minst_file_name = os.path.basename(file_addr)  # 根据地址获取MNIST文件名字
        with gzip.open(filename = file_addr, mode = "rb") as minst_file:
            mnist_file_content = minst_file.read()
            if (minst_file_name.find("label") != -1):  # 若传入的为标签二进制编码文件地址
                data = np.frombuffer(buffer = mnist_file_content, dtype = np.uint8, offset = 8)  # MNIST标签文件的前8个字节为描述性内容
            else:  # 若传入的为图片二进制编码文件地址
                data = np.frombuffer(buffer = mnist_file_content, dtype = np.uint8, offset = 16)  # MNIST图片文件的前16个字节为描述性内容
                data = data.reshape(-1, 28, 28)
        # 截取其中一部分返回
        if (minst_file_name.find("t10k") != -1): return data[: TEST_SCALE]
        else: return data[: DATA_SCALE]

#************************************************************************************************************************************#

class ResNet18(nn.Module):
    class ConvNormAct(nn.Module):
        """生成Convolution-Normalize-(Activate)序列"""
        def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int, padding: int,
                    bias=False, activation=True) -> object:
            super(ResNet18.ConvNormAct, self).__init__()
            layers = [nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),  # 卷积
                        nn.BatchNorm2d(out_channels)]  # 归一
            if (activation): layers.append(nn.ReLU(inplace=True))  # 激活
            self.seq = nn.Sequential(*layers)

        def forward(self, x: torch.tensor) -> torch.tensor:
            return self.seq(x)
    
    #--------------------------------------------------------------------#

    class BasicBlock(nn.Module):
        """基于ConvNormAct生成ResNet中的BasicBlock"""
        def __init__(self, in_channels: int, out_channels: int, strides: int) -> None:
            super(ResNet18.BasicBlock, self).__init__()
            self.conv1 = ResNet18.ConvNormAct(in_channels, out_channels, 3, stride=strides, padding=1, bias=False)  # same padding
            self.conv2 = ResNet18.ConvNormAct(out_channels, out_channels, 3, stride=1, padding=1, bias=False, activation=False)

            self.short_cut = nn.Sequential()
            if (strides != 1):  # 如果是虚线旁路
                self.short_cut = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, 1, stride=strides, padding=0, bias=False),
                    nn.BatchNorm2d(out_channels)
                )

        def forward(self, x: torch.tensor) -> torch.tensor:
            out = self.conv1(x)
            out = self.conv2(out)
            out = out + self.short_cut(x)
            return F.relu(out)
        
    #--------------------------------------------------------------------#

    def __init__(self, block_type: object=BasicBlock, groups: list[int]=[2, 2, 2, 2], num_classes=10) -> object:
        """基于BasicBlock生成ResNet"""
        super(ResNet18, self).__init__()
        self.channels = 64  # 这个参数会在创建convi_x时更新，用来记录下一次BasicBlock的输入通道数
        self.block_type = block_type

        self.conv1 = nn.Conv2d(1, self.channels, kernel_size=7, stride=2, padding=3, bias=False)
        self.batch_norm = nn.BatchNorm2d(self.channels)
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv2_x = self.ConvX(channels=64, blocks=groups[0], strides=1, index=2)
        self.conv3_x = self.ConvX(channels=128, blocks=groups[1], strides=2, index=3)
        self.conv4_x = self.ConvX(channels=256, blocks=groups[2], strides=2, index=4)
        self.conv5_x = self.ConvX(channels=512, blocks=groups[3], strides=2, index=5)
        self.average_pool = nn.AvgPool2d(7)
        self.fc = nn.Linear(512, num_classes)  # 分类数为num_classes

    def ConvX(self, channels: int, blocks: int, strides: int, index: int) -> nn.Sequential:
        """生成conv_x层(基于BasicBlock)"""
        strides_list = [strides] + [1] * (blocks - 1)  # 对于一个conv_x，第一个卷积的stride为2，其余为1
        conv_x = nn.Sequential()
        for i in range(len(strides_list)):
            layer_name = str("block_%d_%d" % (index, i))  # add_module要求名字不同
            conv_x.add_module(layer_name, self.block_type(self.channels, channels, strides_list[i]))
            self.channels = channels  # 更新为下一BasicBlock的输入通道数
        return conv_x

    def forward(self, x: torch.tensor) -> torch.tensor:
        out = self.conv1(x)
        out = F.relu(self.batch_norm(out))
        out = self.max_pool(out)
        out = self.conv2_x(out)
        out = self.conv3_x(out)
        out = self.conv4_x(out)
        out = self.conv5_x(out)
        out = self.average_pool(out)
        out = out.view(out.size(0), -1)
        out = F.softmax(self.fc(out), dim=1)
        return out

#************************************************************************************************************************************#

class Classify:
    def __init__(self, iteration: int = 100) -> None:
        self.iteration = iteration
        self.loss_per_iteration, self.accuracy_per_iteration = [], []
        self.__init_data__()
        self.ResNet = ResNet18()
        self.Loss = nn.CrossEntropyLoss()
        self.Optimizer = torch.optim.SGD(self.ResNet.parameters(), lr = 1e-3)
        
    def __init_data__(self) -> None:
        """加载训练数据集和测试数据集"""
        transformation = transforms.Compose([
            transforms.Resize([224, 224]),
            transforms.ToTensor(),
            transforms.Normalize([0.5],[0.5])]
        )
        # train_dataset = datasets.MNIST(root = './MNIST', train = True, transform = transformation, download = True)
        # test_dataset = datasets.MNIST(root = './MNIST', train = False, transform = transformation, download = True)
        train_dataset = DataProcess("./MNIST/MNIST/raw/train-images-idx3-ubyte.gz", "./MNIST/MNIST/raw/train-labels-idx1-ubyte.gz", transformation)
        test_dataset = DataProcess("./MNIST/MNIST/raw/t10k-images-idx3-ubyte.gz", "./MNIST/MNIST/raw/t10k-labels-idx1-ubyte.gz", transformation)
        self.train_loader = torch.utils.data.DataLoader(dataset = train_dataset, batch_size=5, shuffle=True, num_workers=0)
        self.test_loader = torch.utils.data.DataLoader(dataset = test_dataset, batch_size=5, shuffle=True, num_workers=0)

    def Train(self) -> None:
        """训练函数"""
        self.ResNet.train() # 训练模式
        for batch, (X, y) in (enumerate(self.train_loader)):
            loss = self.Loss(self.ResNet(X), y) # 计算损失
            self.Optimizer.zero_grad() # 梯度清零
            loss.backward() # 反向传播
            self.Optimizer.step() # 优化器更新

    def Test(self) -> None:
        """测试函数"""
        self.ResNet.eval() # 评估模式
        size = len(self.test_loader.dataset)
        num_batches = len(self.test_loader)
        test_loss, correct = 0.0, 0.0
        with torch.no_grad():
            for X, y in self.test_loader:
                pred = self.ResNet(X)
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
        _ = plt.title("ResNet18每次迭代中的损失")
        plt.grid(True)
        plt.subplot(1, 2, 2)
        plt.plot(self.accuracy_per_iteration)
        plt.xlabel("迭代次数")
        plt.ylabel("准确率")
        _ = plt.title("ResNet18每次迭代中的准确率")
        plt.grid(True)
        plt.show()
        
    def Solve(self) -> None:
        """对外的解决方法"""
        for i in trange(0, self.iteration):
            self.Train()
            self.Test()
        self.Draw()

#************************************************************************************************************************************#

if __name__ == "__main__":
    classify = Classify(ITERATION)
    classify.Solve()
