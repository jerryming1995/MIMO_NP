# import tensorflow as tf
import numpy as np
# from Detector import Detector
import scipy.io as sio
from utils import *
from Dataset_Mu import *

# from CommonCom import *
np.random.seed(667)
## 备注：未开发完 有需要继续开发1024
class Mu_detector_np(object):
    """
    为方便dubug与结果观测 Mu_detector将会做成单数据检测形式而batch检测形式
    """
    def __init__(self, params):

        self.Nr = params['Nr']
        self.Nt = params['Nt']
        self.constellation = params['constellation']
        # self.M = np.shape(params['constellation'])[0]
        self.M = params['M']
        self.User = params['User']
        self.Ns = self.Nt * self.User
        self.SNR_dBs = np.arange(params['SNR_dB_min_test'], params['SNR_dB_max_test'], params['SNR_step_test'])
        # self.mode = params['mode']  # 有些算法可能本身带有多种模式可以选用
        self.mode= 0
        self.type = params['type']
        self.rho = params['rho']
        self.num = params.get('num',4)
        self.x_hat = 0  #用于存储当前估计的帧
        self.iteration = params['iteration']
        self.fc_layer = params.get('fc_layer',1)
        self.p_mode = params.get('p_mode','ml')
        self.i_mode = params.get('i_mode','ml')

    def ZF(self,y,H):

        HTH_inv = np.linalg.inv(np.matmul(np.conj(H).T, H))
        HT = np.conj(H).T
        Hi_pinv = np.matmul(HTH_inv, HT)  # 伪逆 (Nt,Nt)
        res = Hi_pinv@y
        res = np.expand_dims(np.array(res), axis=0)
        res = demodulate_np(res,self.constellation)
        self.x_hat = res
        return res

    def MMSE(self,y,H,SNR):
        sigma2 =  (self.Nt*self.User) / (np.power(10, SNR / 10) * self.Nr)
        HTH_inv = np.linalg.inv((np.matmul(np.conj(H).T, H)+ sigma2*np.eye(self.Nt*self.User)))
        Hty = np.conj(H).T@y
        res = HTH_inv@Hty
        res = np.expand_dims(np.array(res), axis=0)
        res = demodulate_np(res, self.constellation)
        self.x_hat = res
        return res

    def Soft_SIC(self, y, H, SNR):
        """

        :param y:
        :param H:
        :param SNR:
        :return:
        """

        sigma2 = (self.Nt * self.User) / (np.power(10, SNR / 10) * self.Nr)
        # sigma2 = sigma2 + sigma2*0.3*np.random.randn(1)
        # sigma2 = 1
        Nt = self.Nt * self.User
        Nr = self.Nr
        Iter = self.iteration
        # print("iter",Iter)
        M = self.M
        p_i =  (1/M-0)*np.ones((Nt,M),dtype=np.float)   #初始化概率密度函数
        p_i[:,1] = (1/M+0)*np.ones(Nt)

        cons = np.expand_dims(self.constellation,axis=0) # (1,M)



        y = np.expand_dims(y,axis=1)
        for i in range(Iter):
            e_i = np.expand_dims(np.sum(cons*p_i,axis=1),axis=1)  # Nt *1
            v_i = np.expand_dims(np.sum(np.square(np.abs(e_i-cons)) * p_i,axis=1),axis=1)
            # e_i = 2*p_i-1
            # v_i = 1 - np.square(e_i)



            z_total = y - H@e_i
            sigma_total= sigma2 * np.eye(Nr,dtype=np.complex)

            for  t  in range(Nt):
                sigma_total += v_i[t,0] * H[:,t:t+1]@np.conj(H[:,t:t+1]).T  #Nr*Nr   这边写的有问题，要改

            for j in range(Nt):
                z_cur =  z_total + e_i[j,0]* H[:,j:j+1]    #  (Nr,1)
                sigma_cur = sigma_total - v_i[j,0] * H[:,j:j+1]@np.conj(H[:,j:j+1]).T
                # sigma_cur = sigma2 * np.eye(Nr,dtype=np.float)
                sigma_inv = np.linalg.inv(sigma_cur)
                # sigma_inv = np.linalg.inv(sigma2 * np.eye(Nr,dtype=np.complex))  #测试用

                x_temp = z_cur - H[:,j:j+1]*np.tile(cons,(Nr,1)) #(Nr,M)

                # core = -0.5* x_temp.T@sigma_inv
                # core = core@x_temp
                # p_list = np.exp(core)

                p_list = np.exp(-0.5*np.conj(x_temp).T@sigma_inv@x_temp)



                # print("P_list:",np.trace(p_list))

                norm = np.trace(p_list)

                for k in range(M):
                    if norm>1e-100:
                        p_i[j,k] = p_list[k,k]/norm
        # print(p_i)
        # print(np.sum(p_i,axis=1))   #验证概率为1
        index= np.argmax(p_i,axis=1)
        self.x_hat = np.expand_dims(self.constellation[index],axis=0)
        # print(self.x_hat)
        return self.x_hat




    def partial_order(self,H):

        """
        :param H:
        :return:list H_order
        """


        order_list = []
        p_num = self.Ns // self.num
        find_list = [i for i in range(self.Ns)]
        H_prev = H
        for i in range(p_num):  # 一共只做p_num次求导即可
            if not order_list:
                H_raw = H_prev
            HTH_inv = np.linalg.inv(np.matmul(np.conj(H_raw).T, H_raw))
            HT = np.conj(H_raw).T
            Hi_pinv = np.matmul(HTH_inv, HT)  # 伪逆 (Nt,Nt)
            for j in range(self.num):
                k = np.argmin(np.sum((Hi_pinv * np.conj(Hi_pinv)), axis=1))  # 为了防止错误传播先检测噪声增益小的
                Hi_pinv = np.delete(Hi_pinv, k, axis=0)
                H_raw = np.delete(H_raw, k, axis=1)
                order_list.append(find_list[k])
                find_list.remove(find_list[k])

        order_list = order_list[::-1]  # 防止错误传播先解最正确的
        H_new = H[:, order_list]
        return order_list, H_new

    def partial_order_simplified(self,H):  #更简单的分组排序，以用户为单位进行排序
        """

        :param H:
        :return:
        """
        order_list = []
        p_num = self.User
        find_list = [i for i in range(self.User)]
        H_prev = H
        for i in range(p_num):  # 一共只做p_num次求导即可
            if not order_list:
                H_raw = H_prev
            HTH_inv = np.linalg.inv(np.matmul(np.conj(H_raw).T, H_raw))
            HT = np.conj(H_raw).T
            Hi_pinv = np.matmul(HTH_inv, HT)  # 伪逆 (Nt,Nt)
            Score = np.sum((Hi_pinv * np.conj(Hi_pinv)), axis=1)

            User_Score = np.array([np.sum(Score[j * self.Nt:(j + 1) * self.Nt]) for j in range(p_num - i)])

            k = np.argmin(User_Score)

            H_raw = np.delete(H_raw, [i for i in range(k, k + self.Nt)], axis=1)
            order_list += [k for k in range(find_list[k] * self.Nt, (find_list[k] + 1) * self.Nt)]
            find_list.remove(find_list[k])

        order_list = order_list[::-1]  # 防止错误传播先解最正确的
        H_new = H[:, order_list]
        return order_list, H_new

    def FC_SD(self,y,H):           #ZF版本的FC_SD不需要sigma2信息

        # 测试代码
        # H_order = H
        # order_list = np.array(range(self.Ns))

        # Ns_distribution =  [1]*(self.Ns/2) + [self.M] * (self.Ns/2)  #一般是1 一般是M
        # Ns_distribution = [1,1,1,1,1,self.M,self.M,self.M]

        # Ns_distribution = [self.M, self.M, self.M, self.M]

        # Ns_distribution = [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,self.M]

        Nr,Nt = H.shape[0],H.shape[1]


        #
        # if Nt!=self.Ns:   #测试用
        #     raise(ValueError("FCSD failed please check Nt"))


        Ns_distribution = [1]*(Nt-self.fc_layer) + [self.M] * (self.fc_layer)

        Ns_distribution = Ns_distribution[::-1]
        #
        order_list, H_order = self.FC_Order(H, Ns_distribution)

        # 不排序结果 (复杂度低)
        # order_list = [i for i in range(self.Ns)]
        # H_order = H
        # Ns_distribution = [1] * 15 + [self.M] * 1
        # Ns_distribution = Ns_distribution[::-1]


        Q,R = np.linalg.qr(H_order)

        y_eff = np.matmul(np.conj(Q).T,y)

        path = [0 for i in range(self.M)]   # 记录最小路径

        R_min = [float('inf')]              # 记录最小路径的路径损失值


        def FC_dfs(iter,temp):
            if iter == Nt:   #达到最深深度
                path_temp = np.array(temp[::-1])      # self.Ns = Nt
                y_hat = np.matmul(R,path_temp)
                distance = np.sum(np.square(np.abs(y_eff - y_hat)))
                if distance <= R_min[0]:
                    path[:] = temp[:]
                    R_min[0] = distance
                return

            infer = 0
            for i in range(iter):
                infer += R[Nt-1-iter,Nt-1-i] * temp[i]
            x = (y_eff[Nt-1-iter]-infer)/R[Nt-1-iter,Nt-1-iter]    # 算符号的估计

            Ns_i  =  Ns_distribution[iter]          # 本层的节点数
            #当前版本先设计 只有  1, Ns分布的情况
            if Ns_i==1:
                cons_dis = x - self.constellation
                distance = np.square(np.abs(cons_dis))
                k = np.argmin(distance)   # 硬判决只保留最小分支度量,直接硬判决符号
                FC_dfs(iter+1,temp+[self.constellation[k]])
            else:
                for i in range(self.M):
                    FC_dfs(iter+1,temp+[self.constellation[i]])  # 全搜索保留所有分支

        FC_dfs(0,[])

        # print(path)
        survive_path = path[::-1]   #从底下检测上来的需要还原回去

        res = np.zeros(Nt,dtype = np.complex)

        # print(order_list)
        for i,item in enumerate (order_list):
            res[item] =  survive_path[i]

        res = np.expand_dims(np.array(res),axis=0)
        self.x_hat = res
        return res

    def FC_Order(self,H,Ns_distribution):         #都拆成分块矩阵去做
        """
        :param H:
        :return:list H_order
        """
        Nt = len(Ns_distribution)
        order_list = []
        find_list = [i for i in range(Nt)]
        H_prev= H
        for i in range(Nt):
            if not order_list:
                H_raw = H_prev

            HTH_inv = np.linalg.inv(np.matmul(np.conj(H_raw).T,H_raw))
            HT = np.conj(H_raw).T
            Hi_pinv = np.matmul(HTH_inv,HT)       # 伪逆 (Nt,Nt)

            if(Ns_distribution[i]<self.M):
                k = np.argmin(np.sum((Hi_pinv * np.conj(Hi_pinv)),axis=1))    #按照(8)式  后检测的信号应当是噪声增益小的
            elif(Ns_distribution[i]==self.M):
                k = np.argmax(np.sum((Hi_pinv * np.conj(Hi_pinv)),axis=1))

            H_raw = np.delete(H_raw, k, axis=1)
            order_list.append(find_list[k])
            find_list.remove(find_list[k])

        order_list = order_list[::-1]
        H_new = H[:,order_list]

        return order_list,H_new



    def V_blast(self,y,H):
        order_list, H_order = self.V_blast_order(H)   #9h,9i
        #
        # order_list, H_order = self.general_order(H)

        # order_list, H_order = self.simplified_order(H)


        Q, R = np.linalg.qr(H_order)

        y_eff = np.matmul(np.conj(Q).T, y)

        x_hat = []

        # for i in range(self.Ns):   备注： 这个写法速度过慢
        #     infer = 0
        #     for j in range(i):
        #         infer += x_hat[j] * R[self.Ns-1-i,self.Ns-1-j]
        #
        #     temp =(y_eff[self.Ns-1-i]-infer)/R[self.Ns-1-i,self.Ns-1-i]  #9e 9d
        #     distance = np.square(np.abs(temp-self.constellation))
        #     x_hat_i = self.constellation[np.argmin(distance)]      # 9f
        #     x_hat.append(x_hat_i)

        for i in range(self.Ns):
            temp = y_eff[self.Ns-1-i]/R[self.Ns-1-i,self.Ns-1-i]
            distance = np.square(np.abs(temp-self.constellation))
            x_hat_i = self.constellation[np.argmin(distance)]      # 9f
            x_hat.append(x_hat_i)
            y_eff -=  x_hat_i * R[:,self.Ns-1-i]          # 减掉已经估出来的部分



        x_hat = x_hat[::-1]
        res = np.zeros(self.Ns, dtype=np.complex)
        # print(order_list)
        for i, item in enumerate(order_list):
            res[item] = x_hat[i]

        res = np.expand_dims(np.array(res), axis=0)
        self.x_hat = res
        return res

    def simplified_order(self,H):
        """
          :param H:
          :return:list H_order
        """
        order_list = []
        find_list = [i for i in range(self.Ns)]
        Q,R = np.linalg.qr(H)
        R_list = [(np.square(np.abs(R[i,i])),i) for i in range(self.Ns)]    # 按照主对角元素进行排序
        R_sort = sorted(R_list,key = lambda x: x[0])
        for (val,index) in R_sort:
            order_list.append(index)          #从小到大
        H_new = H[:, order_list]
        return order_list,H_new

    def V_blast_order(self,H):
        """
           :param H:
           :return:list H_order
           """
        order_list = []
        find_list = [i for i in range(self.Ns)]
        H_prev = H
        for i in range(self.Ns):
            if not order_list:
                H_raw = H_prev

            HTH_inv = np.linalg.inv(np.matmul(np.conj(H_raw).T, H_raw))
            HT = np.conj(H_raw).T
            Hi_pinv = np.matmul(HTH_inv, HT)  # 伪逆 (Nt,Nt)
            k = np.argmin(np.sum((Hi_pinv * np.conj(Hi_pinv)), axis=1))  # 为了防止错误传播先检测噪声增益效地
            H_raw = np.delete(H_raw, k, axis=1)
            order_list.append(find_list[k])
            find_list.remove(find_list[k])

        order_list = order_list[::-1]   #防止错误传播先解最正确的
        H_new = H[:, order_list]

        return order_list, H_new

    def general_order(self,H,mode=0):          #测试这种简化排序的性能，理论上说这种排序本应当和不断求逆的排序性能相同但实际上性能不一样是否为数值计算问题？？？
        """
        :param H:
        :param mode: 选择排序模式 模式0为简化的SNR准则排序效果不如删除求逆怀疑为数值问题
        :return:
        """
        if mode ==0:
            H_prev = H
            order_list = []
            if not order_list:
                H_raw = H_prev
            HTH_inv = np.linalg.inv(np.matmul(np.conj(H_raw).T, H_raw))
            HT = np.conj(H_raw).T
            Hi_pinv = np.matmul(HTH_inv, HT)  # 伪逆 (Nt,Nt)
            H_square = np.sum(np.square(np.abs(Hi_pinv)), axis=1)
            for i in range(self.Ns):
                k = np.argmin(H_square)
                order_list.append(k)
                H_square[k] = np.inf

            H_new = H[:, order_list]

        else:
            raise(ValueError("排序模式不支持"))

        return order_list, H_new


    def P_ML(self,y,H):
        """

        :param y:接收向量
        :param H: 接收矩阵
        :param num: 分组个数
        :return: 返回
        """

        # print("P_ML")
        # SNR准则V_blast排序
        order_list, H_order = self.V_blast_order(H)  # 使用SNR准则进行排序
        # 反向V_blast排序
        # order_list = order_list[::-1]; H_order = H[:,order_list]

        # FC_SD排序
        # Ns_distribution =  [1] * 14 + [self.M]*2
        # order_list, H_order = self.FC_Order(H,Ns_distribution)


        #partial_Order
        # order_list,H_order = self.partial_order(H)

        #simplified_partial_Order
        # order_list, H_order = self.partial_order_simplified(H)


        # 不排序结果 (复杂度低)
        # order_list = [i for i in range(self.Ns)]
        # H_order = H

        # 按R[i,i]简单排序
        # order_list,H_order = self.simplified_order(H)
        num = self.num
        p_num = self.Ns // num + self.Ns%num
        Q, R = np.linalg.qr(H_order)
        y_eff = np.matmul(np.conj(Q).T, y)

        if self.mode==0:                    # 默认使用均匀分组
            if self.Ns% num!=0:
                print("分组错误！")
                raise(ValueError("分组不是整数，请检查分组值num！！"))
            #自定义
            # mode_distribution = [1] * 1 + [0]*(p_num-1)

            # 一半ML 一半硬判
            # mode_distribution = [1]*(p_num//2) + [0]* (p_num-p_num//2)  # 一开始ML后面硬判

            # 分组内全ML
            mode_distribution = [1] * p_num

            # 分组内全硬判
            # mode_distribution = [0]* p_num
            ML_table= self.ML_table_gen(num)
            x_hat = []
            for i in range(p_num):
                y_i = y_eff[(p_num-1-i)*num:(p_num-i)*num]
                R_i = R[(p_num-1-i)*num:(p_num-i)*num,(p_num-1-i)*num:(p_num-i)*num]

                if self.p_mode=="ml":
                    x_hat_i = self.P_ML_detect(y_i,R_i,mode_distribution[i],ML_table)
                elif self.p_mode=="fcsd":
                    x_hat_i = self.P_FCSD_detect(y_i,R_i)
                else:
                    raise(ValueError("其他组内检测算法不支持"))

                x_hat.append(x_hat_i)

                for j in range(num):   # 检测完的部分的影响要在y_eff中扣除
                    y_eff -= x_hat_i[num-1-j] * R[:,(p_num-i-1)*num+j]
            x_hat = np.reshape(np.array(x_hat),[1,-1])
            x_hat = x_hat[0:1,::-1]
            res = np.zeros(self.Ns, dtype=np.complex)
            # print(order_list)
            for i, item in enumerate(order_list):
                res[item] = x_hat[0,i]
            res = np.expand_dims(np.array(res), axis=0)
            self.x_hat = res

        else:                      # 使用不均匀分组
            divide_list = [6,6,2,2]
            mode_list = [1,1,1,1]
            # divide_list = [4,4, 4, 4]
            # mode_list = [1, 1, 1, 1]
            if sum(divide_list)!=self.Ns:
                print("divide_list error!!")
                raise ValueError("check out the divdel_list and mode_list")
            if len(mode_list)!=len(divide_list):
                print("check out the divdel_list and mode_list")
                raise ValueError("check out the divdel_list and mode_list")
            x_hat = []
            pointer = self.Ns
            previous =-1
            for i,j in zip(divide_list,mode_list):
                if j==1 and i>1 and i!=previous:
                    ML_table = self.ML_table_gen(i)
                y_i = y_eff[pointer-i:pointer]
                R_i = R[pointer-i:pointer,pointer-i:pointer]
                x_hat_i = self.P_ML_detect(y_i, R_i, j, ML_table)
                x_hat.append(np.array(x_hat_i))
                for k in range(i):  # 检测完的部分的影响要在y_eff中扣除
                    y_eff -= x_hat_i[k] * R[:, (pointer-1-k)]
                previous = i
                pointer-= i
            x_hat = np.concatenate(x_hat,axis=0)
            x_hat = x_hat[::-1]
            res = np.zeros(self.Ns, dtype=np.complex)
            # print(order_list)
            for i, item in enumerate(order_list):
                res[item] = x_hat[i]
            res = np.expand_dims(np.array(res), axis=0)
            self.x_hat = res
        return res

    def ML_table_gen(self,num):
        ML_table = []

        def generate_ML_table(depth, path):
            if depth == num:
                ML_table.append(path[:])
                return
            for i in range(self.M):
                generate_ML_table(depth + 1, path + [self.constellation[i]])

        generate_ML_table(0, [])  # 输出ML_table的大小为(4^num, num)

        ML_table = np.array(ML_table).T

        return ML_table

    def P_ML_detect(self,y,R,mode,ML_table):

        x_hat = []

        num = R.shape[0]

        if mode==0:
            for i in range(num):
                temp = y[num-1-i]/R[num-1-i,num-1-i]
                index = np.argmin(np.square(np.abs(temp-self.constellation)))
                x_hat_i = self.constellation[index]  #直接硬判
                x_hat.append(x_hat_i)
                y -=x_hat_i*R[:,num-1-i]
            x_hat = np.array(x_hat)  # 1维向量

        elif mode==1:    #mode 1采用ML检测算法
            y = np.expand_dims(y, axis=1)
            distance = np.sum(np.square(np.abs(y - np.matmul(R,ML_table))),axis=0)
            index_k = np.argmin(distance)  #argmin |y-Hx|^2
            x_hat = ML_table[:,index_k]
            x_hat = x_hat[::-1]

        return x_hat

    def P_FCSD_detect(self,y,H):
        """

        :param y: y_eff_i
        :param H:
        :return:
        """
        x_hat = self.FC_SD(y,H)
        res = np.squeeze(x_hat,axis=0)[::-1]
        return res



    def Iteration_cancel(self,y,H):
        """

        :param y:
        :param H:
        :return:
        备注：
        需要处理的一些细节：1.是全部迭代完统一刷新还是边做边刷新2.迭代使用分组ML还是硬判决
        """
        # Standard Iteration
        x_hat = self.P_ML(y,H)
        ML_table = self.ML_table_gen(self.num)
        p_num = self.Ns // self.num + self.Ns % self.num
        # print("num",self.num)
        for i in range(self.iteration):
            x_temp = np.zeros(self.Ns,dtype=np.complex)
            for j in range(p_num):
                low = j*self.num
                high = (j+1)*self.num
                y_eff = y- H[:,:low]@x_hat[0,:low]-H[:,high:]@x_hat[0,high:]
                H_eff = H[:,low:high]
                if self.i_mode =="ml":
                    x_part = self.P_ML_detect(y_eff,H_eff,1,ML_table)
                elif self.i_mode == "fcsd":
                    x_part = self.P_FCSD_detect(y_eff,H_eff)
                else:
                    raise(ValueError("其他迭代干扰抵消算法不支持"))
                x_temp[low:high] = x_part[::-1]           #把当前迭代的所有结果算出来再更新
            x_hat[0,:] = x_temp[:]

        #  Enhanced Iteration
        # x_hat = self.P_ML(y, H)
        # ML_table = self.ML_table_gen(self.num)
        # p_num = self.Ns // self.num + self.Ns % self.num
        # for i in range(self.iteration):
        #     for j in range(p_num):
        #         low = j * self.num
        #         high = (j + 1) * self.num
        #         y_eff = y - H[:, :low] @ x_hat[0, :low] - H[:, high:] @ x_hat[0, high:]
        #         H_eff = H[:, low:high]
        #         x_part = self.P_ML_detect(y_eff, H_eff, 1, ML_table)
        #         x_hat[0,low:high] = x_part[::-1]             #算出来结果就实时的更新

        self.x_hat = x_hat
        return x_hat

    def Test_func(self,iterations,algorithm):
        ser_all=[]
        for i in range(self.SNR_dBs.shape[0]):
            ser = 0.
            print('======================正在仿真SNR:%ddB================================' % (self.SNR_dBs[i]))
            for j in range(iterations):
                # x,y,H = Gen_data(None,self.Nr,self.Nt,self.SNR_dBs[i],self.constellation,self.type,self.rho)
                x,y,H = Gen_data_mu(None,0,self.Nr,self.Nt,self.User,self.SNR_dBs[i],self.constellation,self.type,self.rho)
                # self.Soft_SIC(y,H,self.SNR_dBs[i])
                # x_hat = self.x_hat

                # sio.savemat('test_lab/H.mat',{'H':H})
                # x_hat = self.Iteration_cancel(y,H)
                # x_hat = self.P_ML(y, H)
                # x_hat = self.MMSE(y,H,self.SNR_dBs[i])
                # x_hat = self.FC_SD(y,H)


                #
                if algorithm =="MMSE" or algorithm=="Soft_SIC":
                    to_run = "self."+algorithm+"(y,H,self.SNR_dBs[i])"
                else:
                    to_run = "self." + algorithm+ "(y,H)"
                try:
                    exec(to_run)
                    x_hat = self.x_hat
                except:
                    print("algorithm error!!")
                    raise(ValueError("algorithm error"))

                ser += accuracy(x,x_hat)/iterations
            print("current_ser:",ser)
            ser_all.append(ser)

        return ser_all





if __name__ == "__main__":
    params = {
        # 二选一
        # 'dataset_dir': r'D:\Nr8Nt8batch_size500mod_nameQAM_4',  # 使用固定数据集
        # 'dataset_dir': "./H_data/H.mat",  # 程序运行时生成数据集
        'dataset_dir': None,  # 程序运行时生成数据集

        # ************************程序运行之前先检查下面的参数*****************
        # 仿真算法

        # 仿真参数
        # 'constellation': np.array([0.7071, -0.7071], dtype=np.float32),
        'constellation': np.array([0.7071 + 0.7071j, -0.7071 + 0.7071j, 0.7071 - 0.7071j, -0.7071 - 0.7071j]),
        'Read_constealltaion': False,
        'Nt': 16,  # Number of transmit antennas
        'Nr': 32,  # Number of receive antennas
        'User': 1,  # Number of Users
        'batch_size': 1,
        'M':4,

        # 训练网络时的信噪比，一般选择误符号率在1e-2左右
        'SNR_dB_train': 20,
        # 网络的学习速率，手动调节
        'learning_rate': 0.001,  # Learning rate
        # 测试检测算法的信噪比，一般误符号率到1e-4就行
        'SNR_dB_min_test': 8,  # Minimum SNR value in dB for simulation
        'SNR_dB_max_test': 10,  # Maximum SNR value in dB for simulation
        'SNR_step_test': 2,
        # 测试检测算法的迭代次数，一般保证最低误符号率的错误符号在1e2以上
        'mode': 0,          # 部分算法可能有多种工作模式，默认使用0模式
        'num': 4,          # 分组大小
        'rho':0,           # 现在需要支持有相关性的矩阵
        'iteration': 5,    # 大迭代算法需要提供迭代次数信息
        'type':"related",
        'test_iterations': 2000,
        'p_mode': "ml",
        'i_mode': "ml",
        'fc_layer': 1
    }
    det = Mu_detector_np(params)
    data_node = genData_Mu(params)
    iterations = 2000



    # ser_all = det.Test_func(iterations,"Soft_SIC")
    # print("Soft_SIC",ser_all)


    # ser_all = det.Test_func(iterations,"ZF")
    # print("ZF",ser_all)

    ser_all = det.Test_func(iterations,"MMSE")
    print("MMSE",ser_all)




    ser_all = det.Test_func(iterations, "FC_SD")
    print("FC_SD", ser_all)
    # ser_all = det.Test_func(iterations, "V_blast")
    # print("V_blast", ser_all)

    #
    ser_all = det.Test_func(iterations, "P_ML")
    print("P_ML", ser_all)

    ser_all = det.Test_func(iterations, "Iteration_cancel")
    print("Iteration_cancel", ser_all)








