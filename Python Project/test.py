import numpy as np
from sklearn import datasets
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))


iris = datasets.load_iris()
scale = len(iris.data)
iris_length, iris_width = iris.data[:,2], iris.data[:,3]
iris_target = np.zeros(scale, dtype = int)
for i in range(0, scale):
    iris_target[i] = 0 if (iris.target[i] == 0) else 1
    
with open("data2.csv", 'w') as file:
    for i in range(0, scale):
        file.write(str(iris_length[i]) + ',' + str(iris_width[i]) + ',' + str(iris_target[i]) + '\n')