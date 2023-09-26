#!/usr/bin/python3.6
#-*- coding: utf-8 -*-
import numpy as np
import scipy.io as sio              # 读取数据包
import matplotlib.pyplot as plt


win = 5;
vel = list()    # 角速度
th = list()     # 角度

k = 1           # 角速度范围调整
alpha = 0       # 角速度偏移
beta = 0        # 角度偏移

maxVel = 0;
maxTh = 0;
minVel = 0;
minTh = 0;
alpha_ = 0;
beta_ = 0


def advancedPO(velocity, theta):
    global k, alpha, beta, maxVel, maxTh, minVel, minTh, alpha_, beta_
    vel.append(velocity);
    th.append(theta);
    
    
    if len(vel) > 5:
        vel.pop(0);
        th.pop(0);
        flagmaxVel = maxDetector(vel)
        flagmaxTh = maxDetector(th)
        flagminVel = maxDetector(-np.array(vel))
        flagminTh = maxDetector(-np.array(th))
    
    
        if flagmaxVel == 1:
            maxVel = velocity;
        if flagmaxTh == 1:
            maxTh = theta;
        if flagminVel == 1:
            minVel = velocity;
        if flagminTh == 1:
            minTh = theta;
            
        if(maxVel != 0 and minVel != 0):
            alpha = (maxVel+minVel)/2
            alpha_ = (maxVel-minVel)
            maxVel = 0;
            minVel = 0;
        if(maxTh != 0 and minTh != 0):
            beta = (maxTh+minTh)/2
            beta_ = (maxTh-minTh)
            maxTh = 0;
            minTh = 0;
        if alpha_ != 0 and beta_!=0:
            k = beta_/alpha_
            
    fai = myatan2(k*(velocity-alpha), (theta-beta))
    return fai,alpha,beta


"""
  * @brief  maxDetector 用于检测输入的窗口中是否有峰值
  *
  * @param  maxDetector_window 窗口
  * 
  *
  * @retval 1 - 有峰值

"""

def maxDetector(maxDetector_window):
    win = len(maxDetector_window)
    mid =  maxDetector_window[round(win/2)];
    left = maxDetector_window[0];
    right = maxDetector_window[win-1];
        
    if(left < mid and right < mid):
        return 1
    else:
        return 0

    
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


"""
  * @brief  相交振荡器助力算法测试
  *
  * @param  velocity 助行器关节角速度
  *
  * @param  theta 助行器关节角速度
  *
  * @retval assive_torque - 助力力矩
  *         fai - 关节相位，范围为-pi～pi，从最小关节角度开始（初始值为-pi）
  *         velocity - 同输入值
  *         theta - 同输入值
"""

def test():
    dt = 0.002
    t = np.arange(0,100,dt)

    shiftAngle = 0.2*t  # mimic shift of IMU and change of walking posture
    neuralAngle = 5     # mimic shift of neutral postion of hip
    w = 0.01*t+0.5      # mimic change of walking speed
    y_hip = -10*np.cos(w*t)+shiftAngle+neuralAngle

    C = 1.0
    d = 2.0
    pre = y_hip[0];
    al = list()
    beta = list()
    FAI = list()
    V = list()

    for i in y_hip:
        v = (i-pre)/dt
        if(i==0):
            i = 0.00000001
        pre = i
        fai,a,b = advancedPO(v, i)
        FAI.append(fai)
        al.append(a)
        beta.append(b)
        V.append(v)
    
    plt.subplot(3,1,2)
    plt.plot(t, FAI, label="po phase")
    plt.legend()
    plt.subplot(3,1,1)
    plt.plot(t, y_hip, label="original signal")
    plt.legend()
    plt.subplot(3,1,3)
    plt.plot(t, al, label="alpha")
    plt.plot(t, beta, label="beta")
    plt.legend()
    plt.show()



if __name__ == "__main__":

    test()   
    