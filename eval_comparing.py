# -- coding: utf-8 --


import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import LeavePOut,ShuffleSplit,KFold
from sklearn.model_selection import cross_val_score
from sklearn.neural_network import MLPClassifier,MLPRegressor
import scipy.io as sio



"""
brief - 计算每一步的误差
return - 返回每一步的最大误差, 平均值
"""
def MAX(estimated_phase, true_phase, event):

    err_ = estimated_phase-true_phase
    err_[err_>np.pi] = err_[err_>np.pi]-2*np.pi
    err_[err_<-np.pi] = err_[err_<-np.pi]+2*np.pi
    
    errstride = np.array([]);       # sum error in one stride
    errmaxstride = np.array([]);
    err = np.array([]);
    for i in range(len(event)):
        err = np.append(err, err_[i])
        if(event[i] == 1):
            errstride = np.append( errstride, np.sqrt(np.sum(err**2)/len(err)) )
            errmaxstride = np.append(errmaxstride, np.max(abs(err)))
            err = np.array([]);
        else:
            errstride = np.append(errstride, 0)
            errmaxstride = np.append(errmaxstride, 0)

    return errmaxstride, np.mean(errmaxstride[errmaxstride>0])



"""
brief - 计算均方根误差
return - 返回整个步行过程中的均方根误差
"""
def RSME(estimated_phase, true_phase):
    err = estimated_phase - true_phase
    # error is -pi ~ +pi
    err[err>np.pi] = err[err>np.pi]-2*np.pi
    err[err<-np.pi] = err[err<-np.pi]+2*np.pi

    rsme = np.sqrt( np.sum(err**2)/len(true_phase) )
    return rsme



def convert_GA(value):
    while ( value < 0):
        value  = value + np.pi*2
    while(value > np.pi*2)    :
        value =  value - np.pi*2
    return value
"""
brief - 计算步态相位时间偏移误差. phase 是没有2*pi分割的原始phase
return - 返回时间偏移误差 ATE align time error
"""
def AT_MAX(estimated_phase, true_phase, event):

    ate = np.array([])      # 相位时间偏移误差
    for i in range(len(estimated_phase)):
            
        wall = 0; num = 100;
        alignErr = 100000;
        j = i;      # 真实相位游标
        baseline = convert_GA(true_phase[i]-estimated_phase[i]);

        # 判断 GP 估计值和实际值是否相等
        while 0 <= j and j < len(true_phase) and ( (baseline-np.pi) * (convert_GA(true_phase[j]-estimated_phase[i])-np.pi) > 0 ) :  
            j = j-1;
        if (0 <= j and j < len(true_phase) ):
            alignErr = i-j;
        
        j = i;      # 真实相位游标
        # 判断 GP 估计值和实际值是否相等
        while 0 <= j and j < len(true_phase) and ( (baseline-np.pi) * (convert_GA(true_phase[j]-estimated_phase[i])-np.pi) > 0 ) : 
            j = j+1;
        if ( (0 <= j and j < len(true_phase) ) ) :
            if(abs(alignErr) > abs(i-j)):
                alignErr = i-j;
        
        if(i-num < 0 or i+num> len(estimated_phase)):
            wall = 1

        if(wall == 1):
            alignErr = 0
        
        ate = np.append(ate, alignErr)       # 记录 ，负表示滞后，正表示超前
    
    maxate = list()     # 一步内的最大相位时间偏移误差
    minate = list()
    ate_ = list()       # 一步内的相位时间偏移误差记录
    begin = 0
    for i in range(len(ate)):
        
        if event[i] == 1 and len(ate_) == 0:     # 从第一个peak开始计算
            begin = 1
        
        if begin == 1 :
            if event[i] == 1:                    # 一步完成计算该步内的最大最小值
                if len(ate_) > 0:
                    # print("ate_  ", ate_)
                    #minate.pop(-1)
                    minate.append(np.min(ate_))
                    #maxate.pop(-1)
                    maxate.append(np.max(ate_))
                    ate_.clear()
            else:
                ate_.append(ate[i]) 
        
    maxate = np.array(maxate)
    minate = np.array(minate)
    print(maxate[maxate>0] ,minate[minate<0])
    
    return np.mean(minate[minate<0]), np.mean(maxate[maxate>0])
    


"""
brief - 计算生成的相位的粗糙程度。通过一个步态周期内的相位曲线的斜率的变化速度来体现
return - 返回整个步行过程中每一步的粗糙度平均值
"""
def roughness(estimated_phase):

    index = np.array([0])               # 相位分割坐标.一开始的0主要用于第二个循环的实现
    Sum_k = np.array([0])               # 一步内的相邻斜率差之和
    
    for i in range(len(estimated_phase)-1):         # 分割从2pi到0的时刻,每一步的起点
        if estimated_phase[i] - estimated_phase[i+1] > 5 and (len(index)==1 or i-index[-1] > 20):
            # print(i)
            # 第一步不需要大于20，其次每一步间隔必须大于20
            index = np.append(index, i)

    # 对每一步进行批量操作
    for i in range(len(index)-1):                       
        sum_k_ = 0;  
        for j in range(index[i]+1, index[i+1]+1-2):     # 包含index[i]+1, 不包含index[i+1]+1
            k_1 = (estimated_phase[j+1] - estimated_phase[j])/0.005
            k_2 = (estimated_phase[j+2] - estimated_phase[j+1])/0.005
            sum_k_ = sum_k_ + abs(k_2-k_1)
        Sum_k = np.append(Sum_k, sum_k_)
        
    return np.mean(Sum_k[1:])       # 去除第一值
