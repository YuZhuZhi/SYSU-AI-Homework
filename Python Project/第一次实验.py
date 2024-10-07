
# 作业一：二分查找函数

def BinarySearch(nums: list, target: int):
    pre, back = 0, len(nums) - 1 #双指针法，首先将前后指针分别指向最前和最后元素
    while (pre <= back): #在不越界的情况下
        mid = (pre + back) // 2 #访问区间中间元素
        if (nums[mid] < target): pre = mid + 1 #如果比期望小，说明期望在后半区间
        elif (nums[mid] > target): back = mid - 1 #如果比期望大，说明期望在前半区间
        else: return mid #恰好等于，直接返回
    return -1 #越界，未找到

#------------------------------#

# 作业二：矩阵加乘法
##这里多设计了两个函数：向量数乘VectorMul和矩阵转置MatrixTrans
##这是为了从数学层面更加清晰体现矩阵乘法的意义，即：
##新矩阵的r行c列元素是第一矩阵r行向量与第二矩阵c列向量的数乘
##由于多了矩阵转置的操作，相比起直接按照对原矩阵计算的写法，必然导致时间和空间性能损失
##但考虑到Cache的存在，将第二矩阵列向量换为行向量，可能能提高缓存命中率

import copy

def VectorMul(A: list, B: list):
    sum = 0
    for i in range(0, len(A)): sum = sum + A[i] * B[i] #sum自增两个向量对应坐标值之积
    return sum

def MatrixTrans(A: list[list]):
    returner = copy.deepcopy(A) #创建返回对象，初始化为原矩阵之深复制
    for r in range(0, len(A)):
        for c in range(0, len(A[r])): returner[c][r] = A[r][c] #新矩阵的c行r列元素是原矩阵r行c列元素
    return returner

def MatrixAdd(A: list[list], B: list[list]):
    returner = [] #创建返回对象
    for r in range(0, len(A)):
        temp = []
        for c in range(0, len(A[r])): temp.append(A[r][c] + B[r][c]) #新矩阵[i][j]的元素是第一第二矩阵[i][j]元素之和
        returner.append(temp)
    return returner

def MatrixMul(A: list[list], B: list[list]):
    returner = []
    BTrans = MatrixTrans(B)
    for r in range(0, len(A)):
        temp = []
        for c in range(0, len(A[r])): temp.append(VectorMul(A[r], BTrans[c])) #新矩阵[i][j]的元素是第一矩阵i行向量与第二矩阵j列向量的数乘
        returner.append(temp)
    return returner

#------------------------------#

# 作业三：字典遍历

def ReverseKeyValue(diction: dict):
    returner = {}
    for key, value in diction.items(): returner[value] = key #从前往后遍历交换键和值
    return returner

#------------------------------#

# 附：测试数据：

nums = [1,3,5,7,9,11,33,44,55,66,77,88] #二分查找用例
matrix1 = [[1,2,3],[2,3,4],[3,4,5]] #矩阵加乘用例
matrix2 = [[1,2,3],[4,5,6],[7,8,9]] #矩阵加乘用例
diction = {1:'one', 2:'two', 3:'three', 4:'four'} #字典遍历用例

print("在nums列表中寻找数值0~11:")
for i in range(0, 12): print(i, BinarySearch(nums, i)) #二分查找验证
print("矩阵加法与乘法结果:")
Answer1 = MatrixAdd(matrix1, matrix2)
Answer2 = MatrixMul(matrix1, matrix2)
print(Answer1) #矩阵加法验证
print(Answer2) #矩阵乘法验证
print("原字典与字典遍历结果:")
Answer3 = ReverseKeyValue(diction)
print(diction) #原字典
print(Answer3) #字典遍历验证
