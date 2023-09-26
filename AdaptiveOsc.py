"""
#!/usr/bin/python3.6
#-*- coding: utf-8 -*-
# 2020-5-2
"""

import numpy as np

"""
  * @brief  振荡器类，包含了振荡器池的初始化(init)，迭代处理函数(Oscillators,input,multijoints,predict)，状态显示函数(show)
"""
class Oscillators:
    """
  * @brief  振荡器初始化函数
  * @param  迭代算法的学习参数 vw,va,vph - 对应某个倍频正弦分量的角频率，幅值，相位参数
  *         vw - 常数
  *         va - 常数
  *         vw - 常数
  * @param  振荡器池 order,ww,aa,pphh
  *         order - 振荡器池中的振荡器个数,注意不包含倍频为0的振荡器
  *         ww - 常数 输入为基频，内部按照order的阶数倍频乘以0,1,2...
  *         aa - array 输入为0,1,2,3,order 倍频振荡器对应的幅值初值
  *         pphh - array 输入为0,1,2,3,order 倍频振荡器对应的相位初值
  * @param  采样间隔 dt, 迭代导数更新中使用
  *         dt - 单位为秒
  * @param  是否同步振荡器池中的振荡器 sync
  *         sync - 0 不同步
  * @retval none
    """
    def __init__(self, vw, va, vph, order, dt, ww, aa, pphh, sync=0):
        self.va = va
        self.vw = vw
        self.vph = vph
        self.order = order
        self.aa = aa            # 振荡器当前幅值
        self.init_aa = aa
        self.ww = ww            # 振荡器当前频率
        self.init_ww = ww
        self.pphh = pphh        # 振荡器当前相位
        self.init_pphh = pphh
        self.dt = dt
        self.sync = sync
        print('inital amplify = ', aa)
        print('inital phase = ', pphh)
        print('inital w = ',ww)
        
        self.y_pre = list()         # 预测值状态
        #self.y_pre_old = list()    # self.step 前预测现在的值
        self.phase_pre = list()     # 相位预测值状态
        self.ph1 = list()           # 基频对应的相位
        
        self.IsPhaseGood = list();  # 相位估计是否正常的指示
        
        self.y_osci = list()        # 基于振荡器池的输入信号估计值
        self.a = list()             # 存储不同时刻震荡池中振荡器池的幅值
        self.w = list()             # 存储不同时刻震荡池中振荡器池的频率
        self.ph = list()            # 存储不同时刻震荡池中震荡器的相位
        #self.phase = np.copy(pphh)
        
        self.step = 5               # 默认的预测步长


    """
  * @brief  振荡器池迭代核心算法
  *
  * @param  a_now - 振荡器当前幅值 order+1个阶数
  *         order - 振荡器池中的振荡器个数,注意不包含倍频为0的振荡器
  *         w_now - 振荡器当前频率
  *         ph_now - 振荡器当前相位
  * @param  e - 外部输入-估计值, 迭代的目标值
  * @param  sync - 振荡器是否同步,注意同步的振荡器频率初值必须相同，否则同步就没有意义了；默认不同步
  *         vw_sync - 选择同步后的频率同步学习速率；默认为 0
  * 
  * @retval a_next - 迭代更新的幅值
  *         w_next - 迭代更新的角频率
  *         ph_next - 迭代更新的相位值
  *         test - 用于测试数据 (w_real 振荡器池的角频率,test_da 幅值变化率,test_dw 角频率变化值,test_dph 相位变化率)
    """
    def Oscillators(self, a_now, w_now, ph_now, e, sync=0, vw_sync=0):
        test = list()
        test_da = list()
        test_dw = list()
        test_dph = list()
        
        order = self.order
        va = self.va
        vw = self.vw
        vph = self.vph
        dt = self.dt

        w_real = np.array(range(order+1))*w_now
        a_next = np.copy(a_now)
        w_next = 0
        ph_next = np.copy(ph_now)

        for i in range(order+1):
            if(i == 0):                 # 针对w=0的特殊待遇
                Da = va*e
                Dph = 0
            else:
                Dph = w_real[i] + vph*e*np.cos(ph_now[i])/np.sum(a_now)
                ph_next[i] = dt*Dph + ph_now[i]
                Da = va*e*np.sin(ph_now[i])

            a_next[i] = dt*Da + a_now[i]
            
            if(i == 1 and sync == 0):   # w update. unsync mode.
                Dw = vw*e*np.cos(ph_now[i])/np.sum(a_now)
                #if( dt*Dw + w_now < 0):                             ####### avoid w < 0
                    #Dw = 0
                w_next = dt*Dw + w_now
                test_dw.append(Dw)
            elif(sync == 1):            # w update. multi oscillators freq sync mode.
                w_next = dt*vw_sync + w_now
            
            test_da.append(Da)
            test_dph.append(Dph)
        
        test = (w_real,test_da,test_dw,test_dph)
        
        return a_next, w_next, ph_next, test

    """
  * @brief  使用振荡器池进行预测未来时刻的预测数值
  *
  * @param  a - 振荡器pool的幅值参数
  *         w - 振荡器pool的基频
  *         ph - 振荡器pool的相位
  *
  * @param  t_now - 当前时刻. 与预测过程无关，未来可以去除
  *         delta_t - 预测未来与当前时刻的时间差,预测时长
  * 
  * @retval y_pre - 预测值
    """
    def output(self, a, w, ph,t_now):
        order = self.order
        w_real = np.array(range(order+1))*w
        #print(w_real)
        y_pre = 0
        #print("predict",a.round(2),w_real.round(2),ph.round(2))
        for i in range(order+1):
            if(w_real[i] == 0):
                y_pre = a[i]
            else:
                y_pre = y_pre + a[i]*np.sin(ph[i])
        return y_pre
        

        
    """
  * @brief  一个一个点的输入曲线，可以预测未来时刻的值，使用了上述振荡器迭代函数 Oscillators 和预测函数 predict 的功能
  *
  * @param  y_now - 曲线值
  *         t_now - time at y_now
  *
  * @param  predict_steps - 预测的采样间隔数量. 因此实际的预测长度为 predict_steps * dt
  *
  * @param  sync - 是否同步
  *         vw_sync - 选择同步后的频率同步速率变化率
  *
  * @retval y_next, y_osci_now
  *         y_next - 振荡器池预测的下一时刻的值
  *         y_osci_now - 振荡器池得到的当前的值
    """
    def input(self, y_now, t_now, predict_steps=0, sync=0, vw_sync=0):
        # easy to use. substitute variable name.
        aa = self.aa
        ww = self.ww
        pphh = self.pphh
        dt = self.dt
        self.step = predict_steps
        
        # update
        y_osci_now = self.output(aa,ww,pphh,t_now)
        e = y_now - y_osci_now
        aa,ww,pphh,test = self.Oscillators(aa,ww,pphh,e,sync,vw_sync)
        
        # keep phase in [-2pi, 2pi]
        phase = np.copy(pphh)
        #phase = np.zeros((1,len(aa)))[0]
        (w_real,unuse,unuse,unuse) = test
        for i in range(len(phase)):
            while(phase[i]>0):
	            phase[i] = phase[i]-2*np.pi
            while(phase[i]<0):
	            phase[i] = phase[i]+2*np.pi
        
        # change to original variable name.
        self.aa = aa
        self.ww = ww
        self.pphh = pphh
        self.phase = phase
        
        # save state data
        self.y_osci.append(y_osci_now)
        self.a.append(aa)
        self.w.append(ww)
        self.ph.append(phase)
        
        if(ww < 0):
            phase_base = np.pi-phase[1]
        else:
            phase_base = phase[1]
        self.ph1.append(phase_base)
        
        return y_osci_now
    
    
    """
  * @brief  绘制振荡器池中的振荡器所有状态(频率，相位，角频率)迭代曲线图
  *
  * @param  t - 采样点
  *         y - 采样值
  * @param  show - 1-显示曲线图 0-不显示曲线图，但是直接保存图片到本地
  *         
    """
    def show(self,t,y,show=1,ID=[]):
        font = {'family': 'sans-serif', 'weight': 'normal', 'size': 12}
        dt = self.dt
        step = self.step
        
        plt.subplot(211)
        plt.plot(t,y,label="y",linewidth=3)
        plt.plot(t,self.y_osci,label="osci output",linewidth=1)
        plt.legend()
        plt.title('inital amp='+str(self.init_aa.round(3))+'\n  '+'inital phase='+str(self.init_pphh.round(3))+'\ninital freq='+str(self.init_ww),font)
        
        plt.subplot(212)
        basePhase = np.array(self.ph1)
        plt.plot(t, basePhase, label="phase", linewidth=3)    # 一倍的基频对应的相位
        plt.legend()
        
          
        plt.legend()
        if show==1:
            plt.show()
        else:
            plt.savefig('./img/'+str(ID)+'w='+str(self.init_ww)+'.jpg')
        
        
        plt.subplot(311)
        plt.title('oscillator pool state')
        plt.plot(self.a,label="amplify")
        plt.legend()
        plt.subplot(312)
        plt.plot(self.w,label="frequent")
        plt.legend()
        plt.subplot(313)
        plt.plot(t, self.ph, label="phase")  # draw all the pahse curve
        plt.legend()
        if show==1:
            plt.show()
        
        
    """
  * @brief  返回振荡器重要状态(频率，相位，角频率)迭代曲线图
  *
  * @param  t - 采样点
  *         y - 采样值
  * @param  show - 1-显示曲线图 0-不显示曲线图，但是直接保存图片到本地
  *      
    """
    def states(self):
        phase_base = np.array(self.ph1)
        return phase_base, self.y_pre, self.phase_pre
        

def test_knee():

    # Oscillators parameters
    dt = 0.1;T = 51*dt  # gait cycle
    va = T * 0.2
    vw = va
    vph = np.sqrt(24.2*vw)
    order = 4
    aa = np.random.rand(order+1)*10
    pphh = np.random.rand(order+1)*2*np.pi
    ww = 4 * 2*np.pi/ T

    Osc_knee = Oscillators(vw=vw,va=va,vph=vph,order=order,dt=dt,ww=ww,aa=aa,pphh=pphh);


    # load knee dataset 
    data = sio.loadmat(r'./test/knee.mat')
    y_knee_onecycle = data['knee'][0]; y_knee = np.hstack((np.tile(y_knee_onecycle,20)))
    t = np.arange(0,len(y_knee)*dt,dt)
    
    # feed one by one
    for i in range(len(t)):
        Osc_knee.input(y_knee[i],dt*i)
    
    # plot results
    r = np.random.rand(1)[0].round(5)
    Osc_knee.show(t,y_knee,1,r)


def test_hip():
    from scipy.interpolate import interp1d
    
    # parameters
    dt = 0.1;T = 51*dt  # gait cycle
    va = T * 0.2
    vw = va
    vph = np.sqrt(24.2*vw)
    order = 2
    aa = np.random.rand(order+1)*10
    pphh = np.random.rand(order+1)*2*np.pi
    ww = 4 * 2*np.pi/ T

    Osc_hip = Oscillators(vw=vw,va=va,vph=vph,order=order,dt=dt,ww=ww,aa=aa,pphh=pphh);

    # data
    data = sio.loadmat(r'./test/hipflex.mat')
    y_hip_onecycle = data['hipflex'][0]

    t = np.linspace(0,100,num=len(y_hip_onecycle))
    f1 = interp1d( t,y_hip_onecycle,kind='cubic')
    
    y_hip = []
    
    t_pred =  np.linspace(0,100,num=round(2.5*len(y_hip_onecycle)))
    y_hip = f1(t_pred); y_hip = np.hstack((np.tile(y_hip_onecycle,20)))
    t = np.arange(0,len(y_hip)*dt,dt)
    
    # estmation

    for i in range(len(t)):
        Osc_hip.input(y_hip[i],dt*i)

    # results
    id = np.random.rand(1)[0].round(5)
    Osc_hip.show(t,y_hip,1,id)
    print("aa=",Osc_hip.aa.round(3),"ww=",Osc_hip.ww.round(3))


import scipy.io as sio              # 读取数据包
import matplotlib.pyplot as plt

if __name__ == "__main__":

    test_knee()
    #test_hip()