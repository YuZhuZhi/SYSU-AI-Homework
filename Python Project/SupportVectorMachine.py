import math
from tqdm import *
from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

class SupportVectorMachine:
    def __init__(self) -> None:
        self.iteration = 1000
        self.iris = datasets.load_iris()
        self.scale = len(self.iris.data)
        self.iris_length, self.iris_width = self.iris.data[:,2], self.iris.data[:,3]
        self.iris_target = np.zeros(self.scale, dtype = int)
        for i in range(0, self.scale):
            self.iris_target[i] = -1 if (self.iris.target[i] == 0) else 1
        self.multiplier = self.__init_multiplier__() # 拉格朗日乘子取符合约束条件的随机值
        self.nonezero_multiplier_index = []
        for i in range(0, self.scale):
            if (self.multiplier[i] != 0): self.nonezero_multiplier_index.append(i)
        
    def __init_multiplier__(self) -> np.ndarray:
        """初始化拉格朗日乘子"""
        multiplier = np.random.rand(self.scale)
        sum = self.__Constraint__(multiplier)
        while (sum < 0):
            multiplier = np.random.rand(self.scale)
            sum = self.__Constraint__(multiplier)
        for i in range(0, self.scale):
            if (self.iris_target[i] == -1):
                multiplier[i] = sum + multiplier[i]
                return multiplier
    
    def __Constraint__(self, array: np.ndarray) -> float:
        """计算乘子约束条件"""
        sum = 0.0
        for i in range(0, self.scale): sum = sum + array[i] * self.iris_target[i]
        return sum
        
    def SeqMinOptimize(self, index1: int, index2: int) -> None:
        """更新其中两个拉格朗日乘子"""
        if (index1 == index2): return
        sum = 0.0
        for i in range(0, self.scale):
            if (i == index1): continue
            temp = self.iris_length[index1] * self.iris_length[i] + self.iris_width[index1] * self.iris_width[i]
            sum = sum + self.multiplier[i] * self.iris_target[i] * temp
        temp = self.iris_length[index1] * self.iris_length[index2] + self.iris_width[index1] * self.iris_width[index2]
        ykyl = self.iris_target[index1] * self.iris_target[index2]
        temp = temp * ykyl * self.multiplier[index2]
        new_multi_i = (4 - self.iris_target[index1] * sum - temp) / (2 * (self.iris_length[index1] * self.iris_length[index1] + self.iris_width[index1] * self.iris_width[index1]))
        if (ykyl < 0):
            if (new_multi_i < max(0, self.multiplier[index2] - self.multiplier[index1])):
                new_multi_i = max(0, self.multiplier[index2] - self.multiplier[index1])
        else:
            if (new_multi_i < 0): new_multi_i = 0
            elif (new_multi_i > self.multiplier[index2] + self.multiplier[index1]):
                new_multi_i = self.multiplier[index2] + self.multiplier[index1]
        new_multi_j = (self.multiplier[index1] - new_multi_i) * ykyl + self.multiplier[index2]
        self.multiplier[index1], self.multiplier[index2] = new_multi_i, new_multi_j
    
    def Weight(self) -> "tuple[float, float, float]":
        """返回分类直线的系数"""
        index = 0 # 记录拉格朗日乘子不为0的项的下标
        w_0, w_1, w_2 = 0.0, 0.0, 0.0
        for i in range(0, self.scale):
            if (self.multiplier[i] != 0): index = i
            w_1 = w_1 + self.multiplier[i] * self.iris_length[i] * self.iris_target[i]
            w_2 = w_2 + self.multiplier[i] * self.iris_width[i] * self.iris_target[i]
        w_1, w_2 = w_1 / 2, w_2 / 2
        w_0 = self.iris_target[index] - w_1 * self.iris_length[index] - w_2 * self.iris_width[index]
        return [w_0, w_1, w_2]
    
    def Accuracy(self, weight: tuple[float, float, float]) -> float:
        """返回分类准确率"""
        count = 0
        for i in range(0, self.scale):
            temp = weight[0] + weight[1] * self.iris_length[i] + weight[2] * self.iris_width[i]
            if (temp > 0) and (self.iris_target[i] == 1): count = count + 1
            if (temp < 0) and (self.iris_target[i] == -1): count = count + 1 
        return (count / self.scale)
    
    def Draw(self) -> None:
        """绘制结果"""
        plt.figure()
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        # 绘制数据集
        plt.title("支持向量机鸢尾花分类", size = 14)
        plt.scatter(self.iris_length, self.iris_width, c = self.iris_target)
        plt.xlabel("萼片长度")
        plt.ylabel("萼片宽度")
        # 绘制分类直线
        weight = self.Weight()
        x = np.arange(1.5, 3, 0.01)
        y = -(weight[0] + weight[1] * x) / weight[2]
        plt.plot(x, y)
        plt.show()
    
    def Solve(self) -> "tuple[float, float, float]":
        """对外的解决方法"""
        for i in trange(self.iteration):
            index1, index2 = self.nonezero_multiplier_index[np.random.randint(0, len(self.nonezero_multiplier_index))], self.nonezero_multiplier_index[np.random.randint(0, len(self.nonezero_multiplier_index))]
            while (index1 == index2): index2 = self.nonezero_multiplier_index[np.random.randint(0, len(self.nonezero_multiplier_index))]
            self.nonezero_multiplier_index.remove(index1)
            self.nonezero_multiplier_index.remove(index2)
            self.SeqMinOptimize(index1, index2)
            if (self.multiplier[index1] != 0): self.nonezero_multiplier_index.append(index1)
            if (self.multiplier[index2] != 0): self.nonezero_multiplier_index.append(index2)
        weight = self.Weight()
        print("支持向量机分类准确率：", self.Accuracy(weight))
        self.Draw()
        return weight

svm = SupportVectorMachine()
weight = svm.Solve()