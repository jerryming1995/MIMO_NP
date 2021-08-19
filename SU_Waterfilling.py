import numpy as np

import numpy as np
# from Detector import Detector
import scipy.io as sio
from utils import *
from Dataset_Mu import *
from Mu_detector_np import *
from Mu_detector_coding import *

# from CommonCom import *
np.random.seed(667)

from utils_LDPC import *

class Mu_precode(object):
    def __init__(self,params):


        self.Nr = params['Nr']   # 接收天线数    32
        self.Nt = params['Nt']  # 用户发送天线数  4
        self.User = params['User'] #用户数



        self.constellation = params['constellation']
        # self.M = np.shape(params['constellation'])[0]
        self.M = params['M']   #调制阶数


        #仿真参数
        self.Ns = self.Nt * self.User
        self.SNR_dBs = np.arange(params['SNR_dB_min_test'], params['SNR_dB_max_test'], params['SNR_step_test'])
        self.test_symbol = params['test_symbol']

    def Waterfill3(self, H, sigma2):
        """
        :param H:
        :param sigma2:
        :return:
        """
        # sigma2 = sigma2/self.Nt
        Rh = np.conj(H).T @ H
        V, lam, Vh = np.linalg.svd(Rh)
        # _,lam,_=np.linalg.svd(H)
        P = np.zeros(self.Nt)
        r = np.shape(lam)[0]
        ignore = 0
        sigma2 = 1/sigma2

        while ignore < self.Nt:
            left = self.Nt - ignore
            temp = np.sum(1 / lam[:left])
            mu = self.Nt / (r - ignore) * (1 + 1 / sigma2 * temp)
            for i in range(left):
                P[i] = mu - self.Nt / (sigma2 * lam[i])
            if P[left - 1] < 0:
                ignore += 1
                P[left - 1] = 0
            else:
                break

        L_send = self.Nt - ignore
        P_eff = P[:L_send]
        U_H, S_H, Vh_H = np.linalg.svd(H)
        S_eff = np.zeros((self.Nr, L_send))
        for i in range(L_send):
            S_eff[i, i] = np.sqrt(P_eff[i]) * S_H[i]

        H_eff = U_H @ S_eff

        return H_eff, L_send, U_H, S_eff



    def Waterfill2(self,H,sigma2):
        """
        :param H:
        :param sigma2:
        :return:
        """
        # sigma2 = sigma2/self.Nt
        Rh = np.conj(H).T@H
        V,lam,Vh = np.linalg.svd(Rh)
        # _,lam,_=np.linalg.svd(H)
        P = np.zeros(self.Nt)
        r = np.shape(lam)[0]
        ignore = 0

        while ignore<self.Nt:
            left = self.Nt-ignore
            temp = np.sum(1/lam[:left])
            mu = self.Nt/(r-ignore) * (1+1/sigma2* temp)
            for i in range(left):
                P[i] = mu - self.Nt/(sigma2*lam[i])
            if P[left-1] < 0:
                ignore+=1
                P[left-1]= 0
            else:
                break

        L_send = self.Nt - ignore
        P_eff = P[:L_send]
        U_H, S_H, Vh_H = np.linalg.svd(H)
        S_eff = np.zeros((self.Nr, L_send))
        for i in range(L_send):
            S_eff[i, i] = np.sqrt(P_eff[i]) * S_H[i]

        H_eff = U_H @ S_eff

        return H_eff, L_send,U_H, S_eff



    def Waterfill1(self,H,sigma2):
        """

        :param H:
        :param sigma2:
        :return:
        备注2021.6.5： 似乎有一些问题，注水过程重为何总功率改变？？？？？？
        """
        Rn = 1/sigma2 * np.eye(self.Nr)

        Rh = np.conj(H).T@Rn@H

        V,Lam,Vh = np.linalg.svd(Rh)

        P = np.zeros(self.Nt)

        flag = False

        sqrtrsum = np.sum(1 / np.sqrt(Lam))

        rsum = self.Nt + np.sum(1 / Lam)

        sqrtwater = sqrtrsum / rsum


        for j in range(self.Nt):
            P[j] = 1 / sqrtwater * 1 / np.sqrt(Lam[j]) - 1 / Lam[j]
            if P[j] < 0:
                flag = True
        ignore = 0
        while flag:
            flag = False
            sqrtrsum = sqrtrsum - 1 / np.sqrt(Lam[-1 - ignore])
            rsum = rsum - 1 - 1 / Lam[-1 - ignore]    # 带着ZF检测器一起算？

            sqrtwater = sqrtrsum / rsum
            for j in range(self.Nt):
                P[j] = 1 / sqrtwater * 1 / np.sqrt(Lam[j]) - 1 / Lam[j]
                if P[j] < 0:
                    P[j] = 0
                    if j < self.Nt - ignore - 1:
                        flag = True


            ignore = ignore + 1


        # 功率注水的对角矩阵：
        # P = np.matmul(V,P)
        L_send = self.Nt - ignore
        P_eff = P[self.Nt-L_send:self.Nt]
        U_H,S_H,Vt_H =  np.linalg.svd(H)

        S_eff = np.zeros((self.Nt,L_send))
        for i in range(L_send):
            S_eff[i,i] = np.sqrt(P_eff[i])*S_H[i]

        H_eff = U_H@S_eff
        return H_eff,L_send,


    def MMSE(self,y,H,sigma2):
        HTH = np.conj(H).T@H
        HTH_inv = np.linalg.inv((np.matmul(np.conj(H).T, H)+ sigma2*np.eye(HTH.shape[0])))
        Hty = np.conj(H).T@y
        res = HTH_inv@Hty
        # res = np.expand_dims(np.array(res), axis=0)
        res = res.T
        res = demodulate_np(res, self.constellation)
        self.x_hat = res
        return res

    def svd_solve(self,y,U,L_send,P_eff):
        y = np.squeeze(y)
        temp = np.conj(U).T@y
        # res = np.expand_dims(np.array(res), axis=0)
        # P_show = P_eff
        P_inv = np.linalg.pinv(P_eff)

        # ob = np.zeros((1,L_send),dtype=np.complex)
        # for i in range(L_send):
        #     ob[0,i] = temp[i]/P_eff[i,i]



        res = P_inv@temp
        # res =res.T
        res = np.expand_dims(res,axis=0)



        res = demodulate_np(res, self.constellation)

        return res


    def test_func_normal(self):
        ser_all = []
        for i in range(self.SNR_dBs.shape[0]):
            ser = 0.
            print('======================正在仿真SNR:%ddB================================' % (self.SNR_dBs[i]))
            for j in range(self.test_symbol):
                y = np.zeros((self.Nr, 1), dtype=np.complex)

                H_all_list = []
                x_all_list = []
                sigma2_all = self.Nt * self.User / (np.power(10, self.SNR_dBs[i] / 10) )

                for k in range(self.User):
                    ########### 仅测试用################
                    Hr = np.random.randn(self.Nr, self.Nt) / np.sqrt(2)
                    Hi = np.random.randn(self.Nr, self.Nt) / np.sqrt(2)
                    H = Hr + 1j * Hi
                    # H = np.eye(2) + 1j * np.eye(2)
                    ######################################

                    sigma2 = self.Nt / (np.power(10, self.SNR_dBs[
                        i] / 10))  # sigma2 = self.Nt / (np.power(10, snr / 10) * self.Nr)
                    H_eff, L_send = H, self.Nt
                    H_all_list.append(H_eff)

                    s_index = np.random.randint(low=0, high=len(self.constellation), size=[self.Nt, 1])
                    x = self.constellation[s_index]

                    noise = np.sqrt(sigma2 / 2) * np.random.randn(self.Nr, 1) + 1j * np.sqrt(
                        sigma2 / 2) * np.random.randn(self.Nr, 1)

                    y += H_eff @ x + noise

                    x_all_list.append(x)

                H_all = np.concatenate(H_all_list, axis=1)

                x_all = np.concatenate(x_all_list, axis=0)

                x_hat = self.MMSE(y, H_all, sigma2_all)

                x_hat = np.expand_dims(x_hat, axis=1)

                ser += accuracy(x_all, x_hat) / self.test_symbol
            # print("current_norm_ser:", ser)
            ser_all.append(ser)

        return ser_all



    def test_func(self):
        ser_all = []
        for i in range(self.SNR_dBs.shape[0]):
            ser = 0.
            print('======================正在仿真SNR:%ddB================================' % (self.SNR_dBs[i]))
            for j in range(self.test_symbol):
                y = np.zeros((self.Nr,1),dtype=np.complex)

                H_all_list = []
                x_all_list = []
                sigma2_all = self.Nt*self.User / (np.power(10, self.SNR_dBs[i] / 10)) # sigma2 = self.Nt / (np.power(10, snr / 10) * self.Nr) (不需要再除self.Nr)

                for k in range(self.User):
                    ########### 仅测试用################
                    Hr = np.random.randn(self.Nr, self.Nt)/np.sqrt(2)
                    Hi = np.random.randn(self.Nr, self.Nt)/np.sqrt(2)
                    H = Hr + 1j * Hi
                    # H = np.eye(2) + 1j * np.eye(2)
                    ######################################

                    sigma2 = self.Nt / (np.power(10,self.SNR_dBs[i] / 10))  # sigma2 = self.Nt / (np.power(10, snr / 10) * self.Nr)
                    H_eff,L_send,U,P_eff = self.Waterfill2(H,1/(sigma2/self.Nt))
                    # H_eff, L_send, U, P_eff = self.Waterfill2(H, sigma2 / self.Nt)
                    H_all_list.append(H_eff)

                    s_index = np.random.randint(low=0, high=len(self.constellation), size=[L_send, 1])
                    x = self.constellation[s_index]


                    noise = np.sqrt(sigma2 / 2) * np.random.randn(self.Nr,1) + 1j * np.sqrt(sigma2 / 2) * np.random.randn(self.Nr,1)

                    y += H_eff@x + noise


                    x_all_list.append(x)

                H_all = np.concatenate(H_all_list,axis=1)

                x_all = np.concatenate(x_all_list,axis=0)


                x_hat = self.MMSE(y,H_all,sigma2_all)
                # x_hat = self.svd_solve(y,U,L_send,P_eff)

                x_hat = np.expand_dims(x_hat,axis=1)


                ser += accuracy(x_all, x_hat) / self.test_symbol
            # print("current_ser:", ser)
            ser_all.append(ser)

        return ser_all







if __name__ == "__main__":
    params = {
        # 二选一
        # 'dataset_dir': r'D:\Nr8Nt8batch_size500mod_nameQAM_4',  # 使用固定数据集
        # 'dataset_dir': "./H_data/H.mat",  # 程序运行时生成数据集
        'dataset_dir': None,  # 程序运行时生成数据集

        # ************************程序运行之前先检查下面的参数*****************

        # 仿真参数
        # 'constellation': np.array([0.7071, -0.7071], dtype=np.float32),
        'constellation': sio.loadmat('QAM16.mat')['QAM_16'][0],
        'bitmap': {0.7071 + 0.7071j:[0,0],-0.7071 + 0.7071j:[0,1],0.7071 - 0.7071j:[1,0],-0.7071 - 0.7071j:[1,1]},   # 该映射也可以自动生成
        'Nt': 4,  # Number of transmit antennas
        'Nr': 32,  # Number of receive antennas
        'User': 1,  # Number of Users
        'M':16,


        # 测试检测算法的信噪比，一般误符号率到1e-4就行
        'SNR_dB_min_test':0,  # Minimum SNR value in dB for simulation
        'SNR_dB_max_test':9,  # Maximum SNR value in dB for simulation
        'SNR_step_test': 1,
        'test_symbol': 1000,

        #信道编译码新加字段
        'rate':1/3,
        'test_code':1000,
        'load_dir':"Tanner_R13_K120_Z12_BG2.mat",
        'Z':12

    }
    Mu_pre = Mu_precode(params)


    ser_all = Mu_pre.test_func()
    print("Water_fill", ser_all)

    ser_all_norm = Mu_pre.test_func_normal()
    print("MMSE_norm",ser_all_norm)

