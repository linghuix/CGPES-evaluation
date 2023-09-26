# -- coding: utf-8 --

import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import LeavePOut,ShuffleSplit,KFold
from sklearn.model_selection import cross_val_score
# from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier,MLPRegressor

import pickle                       # pickle模块

import scipy.io as sio

"""
  * @brief  模仿math.atan2(y,x)的-pi-+pi的坐标角度算法
  *
  * @param  y xy坐标轴上的y坐标
  *
  * @param  x xy坐标轴上的x坐标
  *
  * @retval v - xy坐标轴上,坐标（x,y)到原点的直线与x轴正向的夹角弧度值，范围为-pi-+pi
"""
import math
def myatan2(y, x):
    v=0
    if y==0 and x==0:
        print("error y = 0, x = 0")
        return 0.0
    if x>0:
        v = math.atan(y/x);
    if y>=0 and x<0:
        v = math.pi + math.atan(y/x);
    if y<0 and x<0:
        v = -math.pi + math.atan(y/x);
    if y>0 and x==0:
        v = math.pi/2;
    if y<0 and x==0:
        v = -math.pi/2;
    return v


# 读取数据包
data = sio.loadmat(r'test/hipgait.mat')
RH = data['data']['hipd']
RH = RH[0][0]
phase = data['data']['phase']
phase = phase[0][0]

MAXNUM = len(RH)
print('MAXNUM :',MAXNUM)
print('phase :',phase.shape)
print('RH :',RH.shape)


""" NN """
# 目标
coordination_x_y = np.hstack((np.cos(phase),np.sin(phase)))
print('coordination_x_y :',coordination_x_y.shape)

# 特征提取
for i in range(33536):
    if i < 33536-50:
        interval = RH[i:i+50]
        coord = coordination_x_y[i:i+50,:]
        mean = np.mean(interval)
        dev = np.var(interval, ddof = 1)
        maxium = max(interval)[0]
        minimum = min(interval)[0]
        firstval = interval[0][0]
        lastval = interval[-1][0]
        
        target = coord[-1,:]
        feature = [mean,dev,maxium,minimum,firstval,lastval]
        
        if i == 0:
            features = feature
            targets  = target
        else:
            features = np.vstack((features,feature))
            targets  = np.vstack((targets, target))
print('feature, target shape = ', features.shape, targets.shape)


# 训练集和测试集划分
SSlit = KFold(n_splits=2)

clf_NN = MLPRegressor(hidden_layer_sizes=(8,), activation='relu', solver='adam', alpha=0.0001, batch_size='auto', learning_rate='adaptive', max_iter=5000,tol=1e-10, verbose=True, warm_start=False, nesterovs_momentum=True)
    
if 1:
    for train_index, test_index in SSlit.split(features):
        print(features[train_index,:],targets[train_index,:])

    clf_NN.fit(features[train_index,:],targets[train_index,:])

    fw = open("trainedModel",'wb')
    pickle.dump(clf_NN, fw, -1)
    fw.close()
    print('训练模型保存成功！')
    
    
    # 预测的分类数据
    plt.subplot(212)
    results = clf_NN.predict(features)
    print(results)
    
    degree = list()
    row,col = results.shape
    for i in range(row):
        degree.append(myatan2(results[i,1],results[i,0]))
    plt.plot(degree,c='b', label=("NN predicted phase"))
    plt.legend()
    
    plt.subplot(212)
    plt.plot(phase[50:],c='r', label=("manual marked phase"))
    plt.legend()

    plt.subplot(211)
    plt.plot(RH[50:], label=("Right Hip degree curve"))
    plt.scatter(train_index, np.zeros((len(train_index),1)), label=("training set"))
    plt.legend()
    

plt.show()
