import numpy as np
import matplotlib.pyplot as plt
import math
import pandas as pd
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签

plt.close('all')
# 读取训练集
traindata = pd.read_csv("traindata.csv") #直接将csv表中的第一行当作表头
traindata = traindata[['height','weight']]
traindata = np.array(traindata)

male = traindata[0:50, :]
female = traindata[50:100, :]

male_height = traindata[0:50, 0]  # 读取男性身高
female_height = traindata[50:100, 0]  # 读取女性身高
male_weight = traindata[0:50, 1]  #读取男性体重
female_weight = traindata[50:100, 1]  # 读取女性体重

# 画训练集分布图
plt.scatter(male[:,0], male[:,1], alpha = 0.6,s=100,c = 'y') #alpha 透明度 s 大小 c颜色
plt.scatter(female[:,0], female[:,1], alpha = 0.6,s=100,c = 'r')
plt.xlabel('身高')
plt.ylabel('体重')
plt.legend(['男性','女性'])
plt.title('traindata.csv的样本分布')
plt.show()

# 读取测试集
testdata = pd.read_csv("testdata.csv") #直接将csv表中的第一行当作表头
testdata = testdata[['height','weight']]
testdata = np.array(testdata)
# 画测试集分布图
male1 = testdata[0:50, :]
female1 = testdata[50:100, :]
plt.scatter(male1[:,0], male1[:,1], alpha = 0.6, s=100,c = 'y')
plt.scatter(female1[:,0], female1[:,1], alpha = 0.6, s=100,c = 'c')
plt.xlabel('身高')
plt.ylabel('体重')
plt.legend(['男性','女性'])
plt.title('testdata.csv的样本分布')
plt.show()

## 极大似然估计
print('最大似然结果:')
# 身高
male_u_height = male_height.mean() #均值
male_sig_height = np.sqrt(((male_height - male_u_height)**2).mean()) #方差
female_u_height = female_height.mean()
female_sig_height = np.sqrt(((female_height - female_u_height)**2).mean())
print('身高的参数:男性均值{},方差{},女性均值{},方差{}'.format(male_u_height, male_sig_height, female_u_height, female_sig_height))

# 体重
male_u_weight = male_weight.mean()
male_sig_weight = np.sqrt(((male_weight - male_u_weight)**2).mean())
female_u_weight = female_weight.mean()
female_sig_weight = np.sqrt(((female_weight - female_u_weight)**2).mean())
print('体重的参数:男性均值{},方差{},女性均值{},方差{}'.format(male_u_weight, male_sig_weight, female_u_weight, female_sig_weight))

#贝叶斯估计
print('\n*********************\n')
print('贝叶斯估计结果:')
u0 = 0
sig0 = 1
male_N = len(male_height)
male_u_height_bayes = (1/(male_N + male_sig_height**2))*(male_height.sum()) # 参考贝叶斯估计
female_N = len(female_height)
female_u_height_bayes = (1/(female_N + female_sig_height**2))*(female_height.sum())
print('身高的参数:男性均值{},女性均值{}'.format(male_u_height_bayes, female_u_height_bayes))

## 决策面

# 先验概率
P_male = len(male)/(len(male) + len(female))
P_famale = 1 - P_male

# 协方差矩阵
sig_male = np.cov(male.T)
sig_female = np.cov(female.T)

# 均值
mean_male = np.array([male_u_height, male_u_weight]).reshape(-1,1) # 列向量
mean_female = np.array([female_u_height, female_u_weight]).reshape(-1,1)

# 用训练集构建决策面
plt.scatter(male[:,0], male[:,1], alpha = 0.6,s=100,c = 'y') #alpha 透明度 s 大小 c颜色
plt.scatter(female[:,0], female[:,1], alpha = 0.6,s=100,c = 'r')
plt.xlabel('身高')
plt.ylabel('体重')
plt.legend(['男性','女性'])

sample_height = np.linspace(150,200,50) # 构建50*50的一个待检测区域
sample_weight = np.linspace(40,100,50)
sample = np.zeros((50, 50))
for i in range(50):
    for j in range(50):
        x = np.array([sample_height[i],sample_weight[j]]).reshape(-1,1)
        sample[i,j] = 0.5 * (np.dot(np.dot((x-mean_male).T,np.linalg.inv(sig_male)), (x-mean_male))-\
        np.dot(np.dot((x-mean_female).T,np.linalg.inv(sig_female)), (x-mean_female))) +\
        0.5 * math.log(np.linalg.det(sig_male)/np.linalg.det(sig_female)) - math.log(P_male/P_famale)

plt.contour(sample_height, sample_weight, sample, 0, colors = 'green',linewidths=2) #为训练出的决策面
plt.title('训练集训练出的决策面')
plt.show()

# 测试数据集
##在测试集上显示决策面
plt.scatter(male1[:,0], male1[:,1], alpha = 0.6, s=100,c = 'y')
plt.scatter(female1[:,0], female1[:,1], alpha = 0.6, s=100,c = 'c')
plt.xlabel('身高')
plt.ylabel('体重')
plt.legend(['男性','女性'])
plt.contour(sample_height, sample_weight, sample, 0, colors = 'green',linewidths=2)
plt.title('使用测试集的分类结果')
plt.show()

# 在测试集上画待区分的点
plt.scatter(male1[:,0], male1[:,1], alpha = 0.6, s=100,c = 'y')
plt.scatter(female1[:,0], female1[:,1], alpha = 0.6, s=100,c = 'c')
plt.xlabel('身高')
plt.ylabel('体重')
plt.scatter(180, 120, norm = 2, c = 'red', s=100,marker='s')
plt.contour(sample_height, sample_weight, sample, 0, colors = 'green',linewidths=2)
plt.legend(['男性','女性','待检测'])
plt.title('未知点的性别预测结果')
plt.show()
