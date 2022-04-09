

1.实验环境(开发工具、调用库等)



开发工具：pycharm



调用库：numpy、matplotlib.pyplot、math、pandas



2.  使用算法具体结构和参数



①　读取数据：traindata = pd.read_csv("traindata.csv")，traindata = pd.read_csv("testdata.csv")

②　绘制数据集样本分布图：plt.scatter()



③　极大似然估计：loc =
data.mean()，scale = np.sqrt(((data - loc)**2).mean())



④　计算先验概率：P_male =
len(male)/(len(male) + len(female))



⑤　用训练集构建决策面



⑥　用测试集计算错误率



⑦　输入待预测数据并分类：plt.scatter(180, 120, norm = 2, c = 'red', s=100,marker='s')








本项目通过建立贝叶斯分类器，运用极大似然估计，实现了根据身高和体重对人的性别进行分类和预测。由本文的实验结果可知，贝叶斯分类器的决策面可直观看出所预测数据的所属类别，此外，贝叶斯算法还有方法简单、学习效率高的优点。然而，由于算法以自变量间的独立性和连续变量的正态性假设为前提，会导致算法精度在某种程度上受影响。本实验未能实现自动计算错误率的功能，在接下来的学习中，会尝试实现它。



 



