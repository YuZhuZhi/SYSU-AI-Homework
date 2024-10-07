import math
import copy
import re
from tqdm import *
import numpy as np
import matplotlib.pyplot as plt
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

class GeneticAlgTSP:
    def __init__(self, filename: str) -> None:
        """初始化"""
        self.max_iteration = 10000 # 最大迭代次数
        self.max_population = 100 # 种群最大容量
        self.cross_prob = 0.6 # 优秀染色体交叉概率
        self.meta_prob = 0.01 # 染色体变异概率
        self.survive_rate = 0.5 # 生存概率
        self.cities_amount = (int)(re.findall(r"\d+", filename)[0]) # 问题涉及城市数
        self.cities: np.ndarray = np.array([[0.0 for i in range(0, 2)]for j in range(0, self.cities_amount + 1)]) # 存储城市坐标
        with open(filename) as file_obj:
            for line in file_obj.readlines():
                if (line[0] >= 'A' and line[0] <= 'Z'): continue
                nums = line.split(' ')
                self.cities[(int)(nums[0])][0] = (float)(nums[1])
                self.cities[(int)(nums[0])][1] = (float)(nums[2])
        self.population = self.__init_poplulation__() # 初始化种群
        self.best_gene_per_generation: "list[np.ndarray]" = [] # 存储每一代中最优的个体，即最优路径
    
    def iterate(self) -> np.ndarray:
        """解问题的对外方法，返回所得的最优路径"""
        for i in trange(0, self.max_iteration):
            selection = self.Select() # 选择种群中的一部分染色体
            self.Cross(selection) # 将选出的染色体作为亲本产生子代
            self.Meta() # 将产生的子代变异
            # self.DrawPresent()
        self.Select() # 最后做一次选择，实际是为了选出最后一次迭代后的最优个体
        self.DrawResult() # 绘制结果
        return self.best_gene_per_generation[-1]
        
    def DrawPresent(self) -> None:
        """绘制当前迭代所得最优路径图"""
        # 这个函数会导致严重性能问题，故实际并不使用
        x1, y1 = np.zeros(self.cities_amount), np.zeros(self.cities_amount)
        for i in range(0, self.cities_amount):
            x1[i] = self.cities[self.best_gene_per_generation[-1][i]][0]
            y1[i] = self.cities[self.best_gene_per_generation[-1][i]][1]
        plt.plot(x1, y1)
        plt.show()
        plt.pause(0.5)
        plt.close()
    
    def DrawResult(self) -> None:
        """绘制迭代过程中最优路程的变化，以及迭代前的初始路径图、与迭代之后的最优路径图"""
        figure = np.zeros(self.max_iteration)
        for i in range(0, self.max_iteration): figure[i] = self.DistanceTotal(self.best_gene_per_generation[i])
        plt.plot(figure)
        plt.xlabel("迭代次数")
        plt.ylabel("最优路程")
        plt.rcParams['font.sans-serif'] = ['SimHei']
        _ = plt.title("每次迭代中的最优路程")
        plt.show()
        begin_path = self.best_gene_per_generation[1]
        x1, y1 = np.zeros(self.cities_amount), np.zeros(self.cities_amount)
        for i in range(0, self.cities_amount):
            x1[i] = self.cities[begin_path[i]][0]
            y1[i] = self.cities[begin_path[i]][1]
        plt.scatter(x1, y1, c = 'r', s = 20)
        plt.plot(x1, y1)
        plt.rcParams['font.sans-serif'] = ['SimHei']
        _ = plt.title("初始路径图")
        plt.show()
        best_path = self.best_gene_per_generation[-1]
        x2, y2 = np.zeros(self.cities_amount), np.zeros(self.cities_amount)
        for i in range(0, self.cities_amount):
            x2[i] = self.cities[best_path[i]][0]
            y2[i] = self.cities[best_path[i]][1]
        plt.scatter(x2, y2, c = 'r', s = 20)
        plt.plot(x2, y2)
        plt.rcParams['font.sans-serif'] = ['SimHei']
        _ = plt.title("解得路径图")
        plt.show()
    
    def __init_poplulation__(self) -> "list[np.ndarray]":
        """初始化种群"""
        population = []
        path = np.arange(1, self.cities_amount + 1)
        for i in range(0, self.max_population): population.append(np.random.permutation(path))
        return population
    
    def __SinglePointCross__(self, gene1: np.ndarray, gene2: np.ndarray) -> "tuple[np.ndarray]":
        """两个染色体间单点交叉"""
        crosspoint = np.random.randint(0, self.cities_amount) # 选择交叉点
        gene1_list, gene2_list = gene1.tolist(), gene2.tolist()
        child1, child2 = copy.deepcopy(gene1_list), copy.deepcopy(gene2_list)
        for city in gene1_list[crosspoint:]: child2.remove(city) # 剔除保留部分中在交换部分中已有的基因
        for city in gene2_list[crosspoint:]: child1.remove(city) # 剔除保留部分中在交换部分中已有的基因
        child1.extend(gene2_list[crosspoint:])
        child2.extend(gene1_list[crosspoint:])
        return np.array(child1), np.array(child2)
    
    def __TwoPointsCross__(self, gene1: np.ndarray, gene2: np.ndarray) -> "tuple[np.ndarray]":
        """两个染色体间双点交叉"""
        crosspoint1 = np.random.randint(0, self.cities_amount // 2)
        crosspoint2 = np.random.randint(self.cities_amount // 2, self.cities_amount)
        gene1_list, gene2_list = gene1.tolist(), gene2.tolist()
        child1_front, child2_front = copy.deepcopy(gene1_list[0: crosspoint1]), copy.deepcopy(gene2_list[0: crosspoint1])
        child1_back, child2_back = copy.deepcopy(gene1_list[crosspoint1:]), copy.deepcopy(gene2_list[crosspoint1:])
        for city in gene1_list[crosspoint1: crosspoint2]:
            if city in child2_front: child2_front.remove(city) # 剔除保留部分中在交换部分中已有的基因
            if city in child2_back: child2_back.remove(city)
        for city in gene2_list[crosspoint1: crosspoint2]:
            if city in child1_front: child1_front.remove(city) # 剔除保留部分中在交换部分中已有的基因
            if city in child1_back: child1_back.remove(city)
        child1_front.extend(gene2_list[crosspoint1: crosspoint2])
        child1_front.extend(child1_back)
        child2_front.extend(gene1_list[crosspoint1: crosspoint2])
        child2_front.extend(child2_back)
        return np.array(child1_front), np.array(child2_front)
    
    def CrossBetween(self, gene1: np.ndarray, gene2: np.ndarray) -> "tuple[np.ndarray]":
        """两个染色体间交叉"""
        if (np.random.rand() < 0.5): return self.__SinglePointCross__(gene1, gene2) # 五成概率单点交叉
        else: return self.__TwoPointsCross__(gene1, gene2) # 五成概率双点交叉
    
    def Cross(self, selection: "list[int]") -> None:
        """在选择亲本染色体后，按概率生成子代染色体"""
        # 这里是直接将被淘汰者直接替换为新的子代，直到全部替换为止
        replace_index = len(selection) + 1
        for path_index in selection:
            if (np.random.rand() > self.cross_prob): continue
            if (replace_index >= self.max_population): break
            other = np.random.randint(0, len(selection))
            self.population[replace_index - 1], self.population[replace_index] = self.CrossBetween(self.population[path_index], self.population[other])
            replace_index = replace_index + 2
    
    def __PositionExchangeMeta__(self, gene: np.ndarray) -> None:
        """仅针对两个位的互换变异"""
        point1, point2 = np.random.randint(0, self.cities_amount), np.random.randint(0, self.cities_amount)
        gene[point1], gene[point2] = gene[point2], gene[point1]
    
    def __SegmentExchangeMeta__(self, gene: np.ndarray) -> None:
        """针对两个段的互换变异"""
        point1, point2 = np.random.randint(0, self.cities_amount // 2), np.random.randint(self.cities_amount // 2, self.cities_amount)
        lenth = np.random.randint(0, min(point2 - point1, self.cities_amount - point2))
        for i in range(0, lenth): gene[point1 + i], gene[point2 + i] = gene[point2 + i], gene[point1 + i]
    
    def Meta(self) -> None:
        """对种群中所有子代按概率变异"""
        for i in range((int)(self.max_population * self.survive_rate), self.max_population): # 只对子代部分变异
            if (np.random.rand() > self.meta_prob): continue
            else:
                if (np.random.rand() < 0.5): self.__PositionExchangeMeta__(self.population[i]) # 五成概率单位交换变异
                else: self.__SegmentExchangeMeta__(self.population[i]) # 五成概率段交换变异
        # """对种群中所有染色体按概率变异"""
        # 原代码是对所有染色体变异，会造成最优路程的起伏
        # for path in self.population:
        #     if (np.random.rand() > self.meta_prob): continue
        #     else:
        #         if (np.random.rand() < 0.5): self.__PositionExchangeMeta__(path)
        #         else: self.__SegmentExchangeMeta__(path)
        
    def Select(self) -> list:
        """选择亲本染色体"""
        self.population.sort(key = lambda x: self.DistanceTotal(x)) # 按照路程大小升序排序
        self.best_gene_per_generation.append(self.population[0]) # 因此排序之后第一个染色体必是最优者，记录之
        fitness, totalfitness = self.TotalFitness() # 为选择做准备。fitness是各路径的适应度的数组。
        return np.random.choice(np.arange(0, self.max_population), size = (int)(self.max_population * self.survive_rate), replace = False, p = fitness / totalfitness)
    
    def DistanceBetween(self, city1: int, city2: int) -> float:
        """计算两个城市间的距离"""
        delta1 = self.cities[city1][0] - self.cities[city2][0]
        delta2 = self.cities[city1][1] - self.cities[city2][1]
        return math.sqrt(delta1 * delta1 + delta2 * delta2)
    
    def DistanceTotal(self, path: np.ndarray) -> float:
        """计算一个路径中的总距离"""
        distance = self.DistanceBetween(path[self.cities_amount - 1], path[0])
        for i in range(1, self.cities_amount): distance = distance + self.DistanceBetween(path[i - 1], path[i])
        return distance
    
    def TotalFitness(self) -> tuple:
        """计算当前种群的各染色体适应度与总适应度"""
        fitness, totalfitness = [], 0.0
        for path in self.population:
            fitness.append(self.Fitness(path))
            totalfitness = totalfitness + fitness[-1]
        return np.array(fitness), totalfitness
    
    def Fitness(self, path: np.ndarray) -> float:
        """计算一个路径的适应度"""
        return (1000 / self.DistanceTotal(path))

# https://www.math.uwaterloo.ca/tsp/world/countries.html
def main(choice: int):
    if (choice == 1): map = GeneticAlgTSP("dj38.tsp")
    elif (choice == 2): map = GeneticAlgTSP("qa194.tsp")
    elif (choice == 3): map = GeneticAlgTSP("mu1979.tsp")
    elif (choice == 4): map = GeneticAlgTSP("ja9847.tsp")
    elif (choice == 5): map = GeneticAlgTSP("ch71009.tsp")
    print(map.iterate())

main(1)
