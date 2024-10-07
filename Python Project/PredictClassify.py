import math
from tqdm import *
import numpy as np
import matplotlib.pyplot as plt
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

class LogicRegression:
    """逻辑回归算法"""
    def __init__(self, filename: str) -> None:
        age, estimateSalary, purchased, self.loss_per_iteration = [], [], [], []
        with open(filename) as file_obj:
            for line in file_obj.readlines():
                if (line[0].isalpha()): continue
                nums = line.split(',')
                age.append(int(nums[0]))
                estimateSalary.append(int(nums[1]))
                purchased.append(int(nums[2]))
        self.capacity = len(age)
        self.age, self.estimateSalary, self.purchased = np.array(age), np.array(estimateSalary), np.array(purchased)
        self.__normalize_age__ = LogicRegression.__Normalize__(self.age)
        self.__normalize_estimateSalary__ = LogicRegression.__Normalize__(self.estimateSalary)
        self.weight = np.random.rand(3)
        # self.weight = np.zeros(3, dtype = float)
        self.probabilities = np.zeros(self.capacity, dtype = float)
        self.learningRate = 0.5
        self.iteration = 3000
        
    def Probability(self, index: int) -> float:
        """计算给定样本取值为1的概率"""
        intermediate = self.weight[0] + self.__normalize_age__[index] * self.weight[1] + self.__normalize_estimateSalary__[index] * self.weight[2]
        return LogicRegression.Sigmoid(intermediate)
    
    def Loss(self) -> float:
        """使用交叉熵函数衡量损失"""
        loss = 0.0
        for index in range(0, self.capacity):
            self.probabilities[index] = self.Probability(index)
            loss = loss + self.purchased[index] * np.log(self.probabilities[index])
            loss = loss + (1 - self.purchased[index]) * np.log(1 - self.probabilities[index])
        return (-loss / self.capacity)
    
    def Gradient(self) -> "tuple[float, float, float]":
        """求梯度"""
        grad_w0, grad_w1, grad_w2 = 0.0, 0.0, 0.0
        for index in range(0, self.capacity):
            temp = self.purchased[index] - self.probabilities[index]
            grad_w0 = grad_w0 + temp
            grad_w1 = grad_w1 + temp * self.__normalize_age__[index]
            grad_w2 = grad_w2 + temp * self.__normalize_estimateSalary__[index]
        return [grad_w0 / self.capacity, grad_w1 / self.capacity, grad_w2 / self.capacity]
    
    def Accuracy(self) -> float:
        """计算预测准确率"""
        count = 0
        for index in range(0, self.capacity):
            if (self.probabilities[index] >= 0.5) and (self.purchased[index] == 1): count = count + 1
            elif (self.probabilities[index] < 0.5) and (self.purchased[index] == 0): count = count + 1
        return count / self.capacity
    
    def Draw(self) -> None:
        """绘制"""
        # 绘制预测分类图
        plt.scatter(self.age, self.estimateSalary, c = self.purchased)
        plt.xlabel("年龄")
        plt.ylabel("收入")
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        _ = plt.title("逻辑回归预测分类")
        x = np.arange(20, 60, 0.1)
        y = (-self.weight[1] * (x - self.age.min()) / (self.age.max() - self.age.min()) - self.weight[0]) * (self.estimateSalary.max() - self.estimateSalary.min()) / self.weight[2] + self.estimateSalary.min()
        plt.plot(x, y)
        plt.show()
        # 绘制损失函数图
        plt.plot(self.loss_per_iteration)
        plt.xlabel("迭代次数")
        plt.ylabel("损失")
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        _ = plt.title("逻辑回归每次迭代中的损失")
        plt.show()
    
    def Solve(self) -> np.ndarray:
        """对外的解决方法"""
        for i in trange(0, self.iteration):
            self.loss_per_iteration.append(self.Loss())
            gradient = self.Gradient()
            for index in range(0, len(self.weight)): self.weight[index] = self.weight[index] - self.learningRate * gradient[index]
        self.Draw()
        print("逻辑回归准确率：", self.Accuracy())
        return self.weight
    
    @staticmethod
    def Sigmoid(input: float) -> float:
        return (1 / (1 + np.e ** input))
    
    @staticmethod
    def __Normalize__(array: np.ndarray) -> np.ndarray:
        """将给定的数组归一化"""
        returner = np.zeros(array.size, float)
        max, min = array.max(), array.min()
        for i in range(0, returner.size): returner[i] = (array[i] - min) / (max - min)
        return returner

####################################################################################

class Perceptron:
    def __init__(self, filename: str) -> None:
        age, estimateSalary, purchased, self.misclassified , self.loss_per_iteration = [], [], [], [], []
        with open(filename) as file_obj:
            for line in file_obj.readlines():
                if (line[0].isalpha()): continue
                nums = line.split(',')
                age.append(int(nums[0]))
                estimateSalary.append(int(nums[1]))
                purchased.append(int(nums[2]))
        self.capacity = len(age)
        self.age, self.estimateSalary, self.purchased = np.array(age), np.array(estimateSalary), np.array(purchased)
        self.__normalize_age__ = LogicRegression.__Normalize__(self.age)
        self.__normalize_estimateSalary__ = LogicRegression.__Normalize__(self.estimateSalary)
        self.weight = np.random.rand(3)
        self.learningRate = 0.0005
        self.iteration = 3000
        
    def Loss(self) -> float:
        """使用距离衡量损失"""
        loss, self.misclassified = 0.0, []
        for index in range(0, self.capacity):
            intermediate = self.weight[0] + self.weight[1] * self.__normalize_age__[index] + self.weight[2] * self.__normalize_estimateSalary__[index]
            if (intermediate > 0) and (self.purchased[index] == 0):
                self.misclassified.append(index)
                loss = loss + intermediate
            elif (intermediate < 0) and (self.purchased[index] == 1):
                self.misclassified.append(index)
                loss = loss - intermediate
        return loss / math.sqrt(self.weight[1] * self.weight[1] + self.weight[2] * self.weight[2])
    
    def Gradient(self) -> "tuple[float, float, float]":
        """求梯度"""
        chosen = self.misclassified[np.random.randint(0, len(self.misclassified))] # 随机选择一个误分类数据点(的下标)
        intermediate = self.weight[0] + self.weight[1] * self.__normalize_age__[chosen] + self.weight[2] * self.__normalize_estimateSalary__[chosen]
        grad_w0 = self.purchased[chosen] - Perceptron.Sign(intermediate)
        grad_w1, grad_w2 = grad_w0 * self.__normalize_age__[chosen], grad_w0 * self.__normalize_estimateSalary__[chosen]
        return [grad_w0, grad_w1, grad_w2]
    
    def Accuracy(self) -> float:
        """计算预测准确率"""
        return 1 - (len(self.misclassified) / self.capacity)
    
    def Draw(self) -> None:
        """绘制"""
        # 绘制预测分类图
        plt.scatter(self.age, self.estimateSalary, c = self.purchased)
        plt.xlabel("年龄")
        plt.ylabel("收入")
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        _ = plt.title("感知机预测分类")
        x = np.arange(20, 60, 0.1)
        y = (-self.weight[1] * (x - self.age.min()) / (self.age.max() - self.age.min()) - self.weight[0]) * (self.estimateSalary.max() - self.estimateSalary.min()) / self.weight[2] + self.estimateSalary.min()
        plt.plot(x, y)
        plt.show()
        # 绘制损失函数图
        plt.plot(self.loss_per_iteration)
        plt.xlabel("迭代次数")
        plt.ylabel("损失")
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        _ = plt.title("感知机每次迭代中的损失")
        plt.show()
        
    def Solve(self) -> np.ndarray:
        """对外的解决方法"""
        for i in trange(0, self.iteration):
            self.loss_per_iteration.append(self.Loss())
            gradient = self.Gradient()
            for index in range(0, len(self.weight)): self.weight[index] = self.weight[index] + self.learningRate * gradient[index]
        self.Draw()
        print("感知机准确率：", self.Accuracy())
        return self.weight
    
    @staticmethod
    def Sign(input: float) -> int:
        return (1) if (input >= 0) else (-1)
    
    @staticmethod
    def __Normalize__(array: np.ndarray) -> np.ndarray:
        """将给定的数组归一化"""
        returner = np.zeros(array.size, float)
        max, min = array.max(), array.min()
        for i in range(0, returner.size): returner[i] = (array[i] - min) / (max - min)
        return returner

logic_regression = LogicRegression("data.csv")
logic_regression.Solve()

perceptron = Perceptron("data.csv")
perceptron.Solve()