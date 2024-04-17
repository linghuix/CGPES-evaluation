import numpy as np

"""
  * @brief  振荡器类，包含了振荡器池的初始化(init)，迭代处理函数(Oscillators,input,multijoints,predict)，状态显示函数(show)
"""
class TBE:
    """
  * @brief              intialization for TBE 
  * @param  previous_Period_List      list to store previous gait cycle
  * estimatedPeriod     estimated gait cycle
  * lasteventTime       record last time when event happens
  * eventcounter        
  * preN                number of previous gait cycle for estimating current gait cycle
  * @retval none
    """
    def __init__(self):
        self.previous_Period_List = []
        self.estimatedPeriod = 0;
        self.lasteventTime = 0
        self.eventcounter = 0
        self.preN = 1
        
    """
  * @brief  
  *
  * @param  event - 事件触发信息
  *         index - 时间信息
  *
  *
  * @retval y_next, y_osci_now
  *         y_next - 振荡器池预测的下一时刻的值
  *         y_osci_now - 振荡器池得到的当前的值
    """
    def input(self, event, time_index):

        # if event happens
        if(event == 1 and time_index - self.lasteventTime > 20):      # event happened，如果有一连串事件，则取第一次的是真实的事件
            
            # save prevous gait cycle if gait cycle is rational
            if time_index-self.lasteventTime < 500:                   # 周期如果太大，不可能是一步
                self.previous_Period_List.append(time_index-self.lasteventTime)
            
            # keep previous_Period_List length
            if(len(self.previous_Period_List) > self.preN):                         # 窗口最多存储五个之前的步态周期
                self.previous_Period_List.pop(0)
                
            # estimate gait cycle from prevous gait cycles
            if(len(self.previous_Period_List) > 0):
                self.estimatedPeriod = sum(self.previous_Period_List)/len(self.previous_Period_List)
                
            # record event time
            self.lasteventTime = time_index;                          # 记录此次事件
            self.eventcounter = self.eventcounter+1                   # 事件计数器

            return 0
        
        # if event does not happen
        else:
            # deal with special case
            if(self.estimatedPeriod == 0 or (time_index-self.lasteventTime)>self.estimatedPeriod ):
                return 0
            # return estimated gait phase
            return (time_index-self.lasteventTime)/self.estimatedPeriod * 2*np.pi


import scipy.io as sio              # 读取数据包
import matplotlib.pyplot as plt
import copy



def test_tbe():

    tbe = TBE()

    data = sio.loadmat(r'./test/event.mat')
    data_event = data['peak'][0]

    # estimate gait phase
    fai = list()
    for i in range(len(data_event)):
        fai.append( tbe.input(data_event[i], i) )
 
    # plot
    plt.subplot(211); plt.plot(data_event)
    plt.subplot(211); plt.plot(fai)
    plt.show()

 


if __name__ == "__main__":
    test_tbe()
