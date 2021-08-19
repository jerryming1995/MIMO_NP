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

class Mu_precode_channelcoding(object):
    def __init__(self,params):


        self.Nr = params['Nr']   # 接收天线数    32
        self.Nt = params['Nt']  # 用户发送天线数  4
        self.User = params['User'] #用户数



        self.constellation = params['constellation']
        # self.M = np.shape(params['constellation'])[0]


        self.M = params['M']   #调制阶数
        self.order = np.log2(self.M).astype(np.int)  #调制比特数

        # 编码参数
        self.load_dir = params.get("loar_dir", "Tanner_R12_K120_Z12_BG2.mat")
        self.code_params = Parameter_Gen(self.load_dir)
        self.K = params['K']
        self.N = params['N']
        self.Z = params['Z']
        self.G = construct_generate_matrix(convert(self.code_params), self.code_params)


        #仿真参数
        self.Ns = self.Nt * self.User
        self.SNR_dBs = np.arange(params['SNR_dB_min_test'], params['SNR_dB_max_test'], params['SNR_step_test'])
        self.test_symbol = params['test_symbol']
        self.test_code = params['test_code']

    def generate_set(self):
        if len(self.constellation) == 4:  # QSPK
            zero_set = [[0, 1], [0, 2]]
            one_set = [[2, 3], [3, 1]]
        elif len(self.constellation)==16: #16QAM
            zero_set = [[0,1,2,3,4,5,6,7],
                        [0,1,2,3,8,9,10,11],
                        [0,1,4,5,8,9,12,13],
                        [0,2,4,6,8,10,12,14]]
            one_set = [[8,9,10,11,12,13,14,15],
                       [4,5,6,7,12,13,14,15],
                       [2,3,6,7,10,11,14,15],
                       [1,3,5,7,9,11,13,15]]
        else:
            raise(ValueError("暂不支持其他调制"))
        return zero_set,one_set

    def linear_soft_cacu(self,x_hat,sigma2,rou):
        """

        :param x_hat:  L_send_k,1
        :param sigma2:
        :param rou:   L_send_k, 1   归一化因子 高阶调制时重要！
        :return: llr_bit :  1,L_send_k * self.order
        """
        pi = 3.1415926
        llr_bit = np.zeros((1,x_hat.shape[0]*self.order))

        # x_hat = np.expand_dims(x_hat,axis=1)
        cons = np.expand_dims(self.constellation,axis=0)

        distance = np.square(np.abs(rou*((1/rou)*x_hat-cons)))

        zero_set,one_set = self.generate_set()

        for i in range(x_hat.shape[0]):
            for l in range(self.order):
                one_dis = distance[i,one_set[l]]
                zero_dis = distance[i, zero_set[l]]
                llr = 1/(2*sigma2)*(np.min(one_dis)-np.min(zero_dis))
                # llr = 1 / (2 * sigma2) * (np.min(zero_dis) - np.min(one_dis))
                llr_bit[:, i * self.order + l] = llr


        return llr_bit






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
        """

        :param y:   self.Nr, block_len
        :param H:   self.Nr, Lsend_total
        :param sigma2:  1
        :return:  x_hat: self.Nt, L_send_total
                  G: Lsend_total, self.Nr
        """

        HTH = np.conj(H).T@H
        HTH_inv = np.linalg.inv((np.matmul(np.conj(H).T, H)+ sigma2*np.eye(HTH.shape[0])))
        Hty = np.conj(H).T@y
        G= HTH_inv@np.conj(H).T
        res = HTH_inv@Hty
        # res = np.expand_dims(np.array(res), axis=0)

        return res,G

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
        print("仿真一般MMSE")
        bler_all = []
        for s in range(self.SNR_dBs.shape[0]):
            ber = 0.
            bler = 0.
            print('======================正在仿真SNR:%fdB================================' % (self.SNR_dBs[s]))

            count = 0

            while (count < self.test_code):

                # 一个码字内的信道矩阵不变

                H_all_list = []

                sigma2 = self.Nt / (
                    np.power(10, self.SNR_dBs[s] / 10))  # sigma2 = self.Nt / (np.power(10, snr / 10) * self.Nr)

                self.order = np.log2(self.M).astype(np.int)

                L_send = []
                modulation = []
                block_num = []
                trans_ldpc = []
                trans_punc = []

                # 映射过程
                for k in range(self.User):

                    # 此处改为读
                    Hr = np.random.randn(self.Nr, self.Nt) / np.sqrt(2)
                    Hi = np.random.randn(self.Nr, self.Nt) / np.sqrt(2)
                    H = Hr + 1j * Hi

                    H_eff_k, L_send_k = H, self.Nt

                    L_send.append(L_send_k)

                    H_all_list.append(H_eff_k)

                    send_k = np.random.randint(0, 2, (1, self.K))
                    trans_ldpc_k = get_ldpc_code(send_k, self.G)
                    trans_punc_k = trans_ldpc_k[:, 2 * self.Z:]

                    trans_ldpc.append(trans_ldpc_k)
                    trans_punc.append(trans_punc_k)

                    modu_len_k = int(np.shape(trans_punc_k)[1] / self.order)
                    modu_index_k = np.zeros((1, modu_len_k), dtype=np.int)

                    for m in range(modu_len_k):
                        for j in range(self.order):
                            modu_index_k[0, m] += int(
                                np.power(2, self.order - 1 - j) * trans_punc_k[0, self.order * m + j])

                    modulation_k = self.constellation[modu_index_k]

                    modulation.append(modulation_k)

                    block_num_k = np.shape(modulation_k)[1] // L_send_k

                    if np.shape(modulation_k)[1] % L_send_k == 0:
                        block_num.append(block_num_k)  # 每个用户需要发送多少个数据块
                    else:
                        raise (ValueError("不支持非整数分割"))

                H_all = np.concatenate(H_all_list, axis=1)  # (Nr, Lsend_total )

                # 发送与接收过程
                len_total = max(block_num)
                send_block = []
                noise_list = []
                y = np.zeros((self.Nr,), dtype=np.complex)
                for i in range(len_total):
                    noise_t = 0
                    send_block_k_list = []
                    for k in range(self.User):

                        if i < block_num[k]:
                            send_block_k_list.append(
                                modulation[k][:, i * L_send[k]:(i + 1) * L_send[k]].T)  # (L_send_k,1)

                        else:
                            # 长度不够随便发点什么
                            s_index = np.random.randint(low=0, high=len(self.constellation), size=[L_send[k], 1])
                            send_block_k_list.append(self.constellation[s_index])
                            # pass

                        noise_t += np.sqrt(sigma2 / 2) * np.random.randn(self.Nr, 1) + 1j * np.sqrt(
                            sigma2 / 2) * np.random.randn(
                            self.Nr, 1)  # self.Nr,1
                    noise_list.append(noise_t)  # len_total 个 self.Nr,1
                    send_block_k = np.concatenate(send_block_k_list, axis=0)  # (L_send_all, 1)
                    send_block.append(send_block_k)  # len_total 个  L_send_all, k

                noise_all = np.concatenate(noise_list, axis=1)  # Nr ,len_total
                send_all = np.concatenate(send_block, axis=1)  # Lsend_all, len_total

                # 过信道接收
                y = H_all @ send_all + noise_all
                sigma2_all = self.Nt * self.User / (np.power(10, self.SNR_dBs[s] / 10))
                x_hat, G = self.MMSE(y, H_all, sigma2_all)

                rou = np.zeros((H_all.shape[1], 1))  # 先统计所有流的情况
                for i in range(H_all.shape[1]):
                    rou[i, 0] = G[i, :] @ H_all[:, i]
                    # rou[i,0] = 1

                # 测试用代码
                # mse = np.sum(np.square(np.abs(x_hat-send_all)))
                # send_origin = np.reshape(modulation[0],(block_num[0],L_send[0])).T
                # mse = np.sum(np.square(np.abs(send_all-send_origin)))

                # 码块拆分+译码+统计误块率
                start = 0
                for k in range(self.User):

                    x_hat_k = x_hat[start:start + L_send[k], :block_num[k]]
                    # x_hat_k = send_all[start:start+L_send[k],:block_num[k]]
                    # x_hat_k = np.reshape(modulation[k],(block_num[k],L_send[k])).T          #直接用原数据是可以的
                    rou_k = rou[start:start + L_send[k], 0:1]
                    bit_llr = np.zeros((1, np.shape(trans_punc[k])[1]))
                    for i in range(block_num[k]):
                        llr_block = self.linear_soft_cacu(x_hat_k[:, i:i + 1], sigma2, rou_k)
                        bit_llr[:, i * L_send[k] * self.order:(i + 1) * L_send[k] * self.order] = llr_block

                    zero_llr = np.zeros((1, 2 * self.Z))
                    receive = np.concatenate((zero_llr, bit_llr), axis=1)
                    decode_bit = decode_algorithm_NMS(receive, 15, self.code_params)
                    decode_bit = np.expand_dims(decode_bit, axis=0)
                    ber += np.mean(abs(decode_bit - trans_ldpc[k])) / self.User
                    if np.mean(abs(decode_bit - trans_ldpc[k])) != 0:
                        bler += 1 / self.User

                    start += L_send[k]

                ber_c = ber / (count + 1)
                bler_c = bler / (count + 1)

                if count % 5 == 0:
                    print("已完成%d帧数据译码，码块误比特率为" % ((count + 1) * 4), ber_c)
                    print("已完成%d帧数据译码，码块误块率为" % ((count + 1) * 4), bler_c)

                count += 1
            bler_all.append(bler_c)

        return bler_all


    def test_func(self):
        print("仿真注水算法")
        bler_all = []
        for s in range(self.SNR_dBs.shape[0]):
            ber = 0.
            bler = 0.
            print('======================正在仿真SNR:%fdB================================' % (self.SNR_dBs[s]))


            count = 0



            while(count <self.test_code):

                # 一个码字内的信道矩阵不变

                H_all_list = []

                sigma2 = self.Nt / (
                    np.power(10, self.SNR_dBs[s] / 10))  # sigma2 = self.Nt / (np.power(10, snr / 10) * self.Nr)

                self.order = np.log2(self.M).astype(np.int)

                L_send = []
                modulation = []
                block_num = []
                trans_ldpc = []
                trans_punc = []

                # 映射过程
                for k in range(self.User):

                    # 此处改为读
                    Hr = np.random.randn(self.Nr, self.Nt) / np.sqrt(2)
                    Hi = np.random.randn(self.Nr, self.Nt) / np.sqrt(2)
                    H = Hr + 1j * Hi

                    H_eff_k, L_send_k, U, S_eff = self.Waterfill2(H, 1 / (sigma2 / self.Nt))

                    L_send.append(L_send_k)

                    H_all_list.append(H_eff_k)



                    send_k = np.random.randint(0,2,(1,self.K))
                    trans_ldpc_k = get_ldpc_code(send_k, self.G)
                    trans_punc_k = trans_ldpc_k[:, 2 * self.Z:]

                    trans_ldpc.append(trans_ldpc_k)
                    trans_punc.append(trans_punc_k)


                    modu_len_k = int(np.shape(trans_punc_k)[1] / self.order)
                    modu_index_k = np.zeros((1, modu_len_k), dtype=np.int)

                    for m in range(modu_len_k):
                        for j in range(self.order):
                            modu_index_k[0, m] += int(np.power(2, self.order - 1 - j) * trans_punc_k[0, self.order * m + j])

                    modulation_k = self.constellation[modu_index_k]

                    modulation.append(modulation_k)

                    block_num_k = np.shape(modulation_k)[1]//L_send_k

                    if np.shape(modulation_k)[1]%L_send_k == 0:
                        block_num.append(block_num_k)    #每个用户需要发送多少个数据块
                    else:
                        raise (ValueError("不支持非整数分割"))


                H_all = np.concatenate(H_all_list, axis=1)   #(Nr, Lsend_total )



                # 发送与接收过程
                len_total = max(block_num)
                send_block = []
                noise_list = []
                y = np.zeros((self.Nr,), dtype = np.complex)
                for i in range(len_total):
                    noise_t = 0
                    send_block_k_list = []
                    for k in range(self.User):

                        if i<block_num[k]:
                            send_block_k_list.append(modulation[k][:,i*L_send[k]:(i+1)*L_send[k]].T)  #(L_send_k,1)

                        else:
                            # 长度不够随便发点什么
                            s_index = np.random.randint(low=0, high=len(self.constellation), size=[L_send[k], 1])
                            send_block_k_list.append(self.constellation[s_index])
                            # pass

                        noise_t  += np.sqrt(sigma2 / 2) * np.random.randn(self.Nr, 1) + 1j * np.sqrt(sigma2 / 2) * np.random.randn(
                        self.Nr, 1)   # self.Nr,1
                    noise_list.append(noise_t)      # len_total 个 self.Nr,1
                    send_block_k = np.concatenate(send_block_k_list,axis=0)   #(L_send_all, 1)
                    send_block.append(send_block_k)   # len_total 个  L_send_all, k


                noise_all = np.concatenate(noise_list,axis=1) # Nr ,len_total
                send_all = np.concatenate(send_block,axis=1)  # Lsend_all, len_total

                # 过信道接收
                y =  H_all@send_all + noise_all
                sigma2_all = self.Nt*self.User / (np.power(10, self.SNR_dBs[s] / 10))
                x_hat,G = self.MMSE(y,H_all,sigma2_all)

                rou = np.zeros((H_all.shape[1],1))   # 先统计所有流的情况
                for i in range(H_all.shape[1]):
                    rou[i,0] = G[i,:]@H_all[:,i]
                    # rou[i,0] = 1

                # 测试用代码
                # mse = np.sum(np.square(np.abs(x_hat-send_all)))
                # send_origin = np.reshape(modulation[0],(block_num[0],L_send[0])).T
                # mse = np.sum(np.square(np.abs(send_all-send_origin)))



                #码块拆分+译码+统计误块率
                start=0
                for k in range(self.User):

                    x_hat_k = x_hat[start:start+L_send[k],:block_num[k]]
                    # x_hat_k = send_all[start:start+L_send[k],:block_num[k]]
                    # x_hat_k = np.reshape(modulation[k],(block_num[k],L_send[k])).T          #直接用原数据是可以的
                    rou_k = rou[start:start+L_send[k],0:1]
                    bit_llr = np.zeros((1,np.shape(trans_punc[k])[1]))
                    for i in range(block_num[k]):
                        llr_block = self.linear_soft_cacu(x_hat_k[:,i:i+1],sigma2,rou_k)
                        bit_llr[:,i*L_send[k]*self.order:(i+1)*L_send[k]*self.order] = llr_block

                    zero_llr = np.zeros((1, 2 * self.Z))
                    receive = np.concatenate((zero_llr, bit_llr), axis=1)
                    decode_bit = decode_algorithm_NMS(receive,15, self.code_params)
                    decode_bit = np.expand_dims(decode_bit,axis=0)
                    ber += np.mean(abs(decode_bit - trans_ldpc[k]))/self.User
                    if np.mean(abs(decode_bit - trans_ldpc[k]))!=0:
                        bler += 1/self.User

                    start += L_send[k]


                ber_c = ber / (count + 1)
                bler_c = bler/(count + 1)

                if count % 5 == 0:

                    print("已完成%d帧数据译码，码块误比特率为" % ((count+1)*4), ber_c)
                    print("已完成%d帧数据译码，码块误块率为" % ((count+1)*4), bler_c)

                count+=1
            bler_all.append(bler_c)


        return bler_all







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
        'User': 4,  # Number of Users
        'M':16,


        # 测试检测算法的信噪比，一般误符号率到1e-4就行
        'SNR_dB_min_test':6,  # Minimum SNR value in dB for simulation
        'SNR_dB_max_test':7,  # Maximum SNR value in dB for simulation
        'SNR_step_test': 0.25,
        'test_symbol': 1,

        #信道编译码新加字段
        'rate':1/2,
        'test_code':500,
        'load_dir':"Tanner_R12_K120_Z12_BG2.mat",
        'Z':12,
        'N':240,
        'K':120

    }
    Mu_pre = Mu_precode_channelcoding(params)


    # bler_all = Mu_pre.test_func()
    # print("Water_fill", bler_all)

    bler_all_MMSE = Mu_pre.test_func_normal()
    print("MMSE",bler_all_MMSE)


