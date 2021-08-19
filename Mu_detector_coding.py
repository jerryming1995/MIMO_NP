
import numpy as np
# from Detector import Detector
import scipy.io as sio
from utils import *
from Dataset_Mu import *
from Mu_detector_np import *

# from CommonCom import *
np.random.seed(667)

from utils_LDPC import *


class Mu_detector_coding(Mu_detector_np):

    def __init__(self,params):

        super(Mu_detector_coding,self).__init__(params)

        self.rate = params["rate"]

        self.SNR_dBs = np.arange(params['SNR_dB_min_test'], params['SNR_dB_max_test'], params['SNR_step_test'])


        self.test_code = params.get("test_code",20)

        self.load_dir = params.get("loar_dir","Tanner_R13_K120_Z12_BG2.mat")

        self.code_params = Parameter_Gen(self.load_dir)


        self.G = construct_generate_matrix(convert(self.code_params),self.code_params)

        self.Z = params.get("Z",12)

        self.L = params["L"]

        self.distribution = params["distribution"]

        self.bitmap= params["bitmap"]


        if len(self.distribution)!=self.Nt:
            raise(ValueError("分布函数设置错误，请校对！！"))



    def generate_set(self):
        if len(self.constellation)==4:  #QSPK
            zero_set=[[0,1],[0,2]]
            one_set = [[2,3],[3,1]]

        else:
            raise(ValueError("暂时不支持其他调制!"))

        return zero_set,one_set



    def  linear_soft_cacu(self,x_hat,sigma2):
        """

        :param x_hat:  维度(Nt,)
        :return:
        """
        pi = 3.1415926
        llr_bit = np.zeros((1,self.Nt*self.order))
        x_hat=np.expand_dims(x_hat,axis=1)  #(Nt ,1)
        # cons = np.expand_dims(self.constellation,axis=0)
        # distance = np.square(np.abs(x_hat - cons))[:,0]
        distance = np.square(np.abs(x_hat - self.constellation))[:, 0]
        # LLR = 1/(np.sqrt(2*pi*sigma*sigma))*np.exp(-distance/(2*sigma*sigma))  # 暂时先用简化算法提升速度
        zero_set,one_set = self.generate_set()
        for i in range(self.Nt):
            for l in range(self.order):
                one_dis = distance[i,one_set[l]]
                zero_dis = distance[i,zero_set[l]]
                llr= 1/(2*sigma2)*(np.min(one_dis)-np.min(zero_dis))
                llr_bit[:,i*self.order+l]=llr
        return llr_bit


    def set_split(self,x_hat):
        bit_len = self.order

        one_set = [[] for _ in range(self.Nt*self.order)]  # 不要用[[]] * 10有大问题！！

        zero_set = [[] for _ in range(self.Nt*self.order)] # 不要用[[]] * 10有大问题！！


        for l in range(self.L):
            temp = x_hat[l,:]   #(10,)
            for i in range(self.Nt):
                cons = temp[i]
                index = self.bitmap[cons]
                for j in range(bit_len):
                    if index[j] == 0:

                        zero_set[bit_len*i+j].append(temp[:])

                    elif index[j]==1:
                        one_set[bit_len*i+j].append(temp[:])

        return one_set,zero_set




    def candidate_soft_cacu(self,x_hat,sigma2,y,H):
        """

        :param x_hat:
        :param sigma2:
        :param y: (20,1)
        :return:
        """
        one_set, zero_set = self.set_split(x_hat)
        lam = 0.3
        bit_num = len(one_set)
        llr_bit = np.zeros((1, self.Nt * self.order))
        for i in range(bit_num):
            if one_set[i] !=[]:
                one_np = np.array(one_set[i])  #(?,10)
                distance = np.square(np.abs(y-H@one_np.T))  #(20,?)
                one_dis = np.min(np.sum(distance,axis=0))
            if zero_set[i] !=[]:
                zero_np = np.array(zero_set[i])  # (?,10)
                distance = np.square(np.abs(y - H @ zero_np.T))  # (20,?)
                zero_dis = np.min(np.sum(distance, axis=0))

            if zero_set[i] == []:
                llr =  -1 / (2 * sigma2) * one_dis * lam
            elif one_set[i] ==[]:
                llr = 1 / (2 * sigma2) * zero_dis * lam
            elif zero_set[i] != [] and one_set[i] != []:
                llr = 1 / (2 * sigma2) * (one_dis - zero_dis)
            llr_bit[:, i] = llr

        return llr_bit




    def LCSD(self,y,H):

        Nr,Nt = H.shape[0],H.shape[1]

        # Ns_distribution = [1] * (Nt - self.fc_layer) + [self.M] * (self.fc_layer)

        Ns_distribution = self.distribution

        Ns_distribution = Ns_distribution[::-1]
        #
        order_list, H_order = self.FC_Order(H, Ns_distribution)

        # 不排序结果 (复杂度低)
        # order_list = [i for i in range(self.Ns)]
        # H_order = H
        # Ns_distribution = [1] * 15 + [self.M] * 1
        # Ns_distribution = Ns_distribution[::-1]

        Q, R = np.linalg.qr(H_order)

        y_eff = np.matmul(np.conj(Q).T, y)

        path = []


        def FC_dfs(iter, temp):
            if iter == Nt:  # 达到最深深度
                path.append(temp[::-1])
                return

            infer = 0
            for i in range(iter):
                infer += R[Nt - 1 - iter, Nt - 1 - i] * temp[i]
            x = (y_eff[Nt - 1 - iter] - infer) / R[Nt - 1 - iter, Nt - 1 - iter]  # 算符号的估计

            Ns_i = Ns_distribution[iter]  # 本层的节点数
            # 当前版本先设计 只有  1, Ns分布的情况
            if Ns_i == 1:
                cons_dis = x - self.constellation
                distance = np.square(np.abs(cons_dis))
                k = np.argmin(distance)  # 硬判决只保留最小分支度量,直接硬判决符号
                FC_dfs(iter + 1, temp + [self.constellation[k]])
            elif Ns_i>1 and Ns_i<self.M:
                cons_dis = x - self.constellation
                distance = np.square(np.abs(cons_dis))
                for i in range(Ns_i):
                    k = np.argmin(distance)
                    FC_dfs(iter + 1, temp + [self.constellation[k]])
                    distance[k] = np.float("inf")
            else:
                for i in range(self.M):
                    FC_dfs(iter + 1, temp + [self.constellation[i]])  # 全搜索保留所有分支

        FC_dfs(0, [])

        res = np.zeros((self.L,Nt), dtype=np.complex)
        path_np = np.array(path)

        for i, item in enumerate(order_list):
            res[:,item] = path_np[:,i]


        # # 测试用代码
        # y_t = np.expand_dims(y,axis=1)
        #
        # distance = np.sum(np.square(np.abs(y_t - H@res.T)),axis=0)
        #
        # index = np.argmin(distance)
        #
        # self.x_hat = res[index:index+1,:]

        return res         # res : (L,Nt)





    def FC_Order(self, H, Ns_distribution):  # 都拆成分块矩阵去做
        """
        :param H:
        :return:list H_order
        """
        Nt = len(Ns_distribution)
        order_list = []
        find_list = [i for i in range(Nt)]
        H_prev = H
        for i in range(Nt):
            if not order_list:
                H_raw = H_prev

            HTH_inv = np.linalg.inv(np.matmul(np.conj(H_raw).T, H_raw))
            HT = np.conj(H_raw).T
            Hi_pinv = np.matmul(HTH_inv, HT)  # 伪逆 (Nt,Nt)

            if (Ns_distribution[i] < self.M):
                k = np.argmin(np.sum((Hi_pinv * np.conj(Hi_pinv)), axis=1))  # 按照(8)式  后检测的信号应当是噪声增益小的
            elif (Ns_distribution[i] == self.M):
                k = np.argmax(np.sum((Hi_pinv * np.conj(Hi_pinv)), axis=1))

            H_raw = np.delete(H_raw, k, axis=1)
            order_list.append(find_list[k])
            find_list.remove(find_list[k])

        order_list = order_list[::-1]
        H_new = H[:, order_list]

        return order_list, H_new







    def ZF(self, y, H):

        HTH_inv = np.linalg.inv(np.matmul(np.conj(H).T, H))
        HT = np.conj(H).T
        Hi_pinv = np.matmul(HTH_inv, HT)  # 伪逆 (Nt,Nt)
        res = Hi_pinv @ y
        res = np.expand_dims(np.array(res), axis=0)
        return res

    def MMSE(self, y, H, SNR):
        sigma2 = (self.Nt * self.User) / (np.power(10, SNR / 10) * self.Nr)
        HTH_inv = np.linalg.inv((np.matmul(np.conj(H).T, H) + sigma2 * np.eye(self.Nt * self.User)))
        Hty = np.conj(H).T @ y
        res = HTH_inv @ Hty
        res = np.expand_dims(np.array(res), axis=0)
        return res

    def test_func_with_code(self,algorithm):
        ber_all=[]
        for i in range(self.SNR_dBs.shape[0]):
            ber = 0.
            print('======================正在仿真SNR:%fdB================================' % (self.SNR_dBs[i]))
            for k in range(self.test_code):

                K = self.code_params["N"]-self.code_params["M"]

                send = np.random.randint(0, 2, (1, K))

                trans_ldpc = get_ldpc_code(send, self.G)

                # 打孔
                Z = self.Z
                trans_punc = trans_ldpc[:, 2 * Z:]

                #调制
                self.order = np.log2(self.M).astype(np.int)

                modu_len = int(np.shape(trans_punc)[1]/self.order)

                modu_index = np.zeros((1,modu_len),dtype=np.int)

                for m in range(modu_len):
                    for j in range(self.order):
                        modu_index[0,m]+= int(np.power(2,self.order-1-j)*trans_punc[0,self.order*m+j])

                modulation = self.constellation[modu_index]

                block_num = np.shape(modulation)[1] // self.Nt

                if np.shape(modulation)[1]%self.Nt!=0:
                    raise(ValueError("不能整除发送天线数,请假查"))


                receive_llr = np.zeros((1,len(trans_punc)))
                bit_llr = np.zeros((1,np.shape(trans_punc)[1]))

                # 过信道
                sigma2 = self.Nt / (np.power(10, self.SNR_dBs[i] / 10) * self.Nr)
                for t in range(block_num):
                    send_block = modulation[:,t*self.Nt:(t+1)*self.Nt]   # (1,Nt)
                    Hr = np.random.randn(self.Nr, self.Nt) * np.sqrt(0.5 / self.Nr)
                    Hi = np.random.randn(self.Nr, self.Nt) * np.sqrt(0.5 / self.Nr)
                    H = Hr + 1j * Hi


                    noise = np.sqrt(sigma2 / 2) * np.random.randn(self.Nr, 1) + 1j * np.sqrt(sigma2 / 2) * np.random.randn(
                        self.Nr, 1)
                    receive_block = H@send_block.T + noise   #(Nr,1)

                    #检测
                    if algorithm=="MMSE":
                        x_hat_block = self.MMSE(receive_block[:,0],H,self.SNR_dBs[i])  #(1,Nt)
                    elif algorithm == "ZF":
                        x_hat_block = self.ZF(receive_block[:,0],H)             #(1,Nt)
                    elif algorithm == "LCSD":
                        x_hat_block = self.LCSD(receive_block[:,0],H)           #(L,Nt)

                    else:
                        raise (ValueError("其他算法暂时不支持"))

                    #译码
                    if algorithm in ["MMSE","ZF"]: #
                        llr_block = self.linear_soft_cacu(x_hat_block.T,sigma2)
                        bit_llr[:,t*self.Nt*self.order:(t+1)*self.Nt*self.order] = llr_block

                    elif algorithm in ["LCSD"]:
                        # print("USE LCSD")
                        llr_block = self.candidate_soft_cacu(x_hat_block,sigma2,receive_block,H)
                        bit_llr[:, t * self.Nt * self.order:(t + 1) * self.Nt * self.order] = llr_block
                    elif algorithm =="SoftSIC":
                        pass

                    else:
                        raise (ValueError("无对应的似然比计算方法"))


                # 根据似然比译码
                zero_llr = np.zeros((1, 2 * Z))
                receive = np.concatenate((zero_llr, bit_llr), axis=1)
                decode_bit = decode_algorithm_NMS(receive, 20, self.code_params)

                # 计算误比特率
                ber += np.mean(abs(decode_bit - trans_ldpc))
                ber_c = ber / (k + 1)
                if k % 20 == 0:
                    print("已完成%d帧数据译码，码块误比特率为" % k, ber_c)
                # print("已完成%d帧数据译码，码块误比特率为" % k, ber)

            print("SNR:", self.SNR_dBs[i])
            print("ber:", ber / self.test_code)
            ber_all.append(ber / self.test_code)


        return ber_all






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
        'bitmap': {0.7071 + 0.7071j:[0,0],-0.7071 + 0.7071j:[0,1],0.7071 - 0.7071j:[1,0],-0.7071 - 0.7071j:[1,1]},   # 该映射也可以自动生成
        'Read_constealltaion': False,
        'Nt': 10,  # Number of transmit antennas
        'Nr': 20,  # Number of receive antennas
        'User': 1,  # Number of Users
        'batch_size': 1,
        'M':4,

        # 训练网络时的信噪比，一般选择误符号率在1e-2左右
        'SNR_dB_train': 20,
        # 网络的学习速率，手动调节
        'learning_rate': 0.001,  # Learning rate
        # 测试检测算法的信噪比，一般误符号率到1e-4就行
        'SNR_dB_min_test': -1.75,  # Minimum SNR value in dB for simulation
        'SNR_dB_max_test': 0,  # Maximum SNR value in dB for simulation
        'SNR_step_test': 0.25,
        # 测试检测算法的迭代次数，一般保证最低误符号率的错误符号在1e2以上
        'mode': 0,          # 部分算法可能有多种工作模式，默认使用0模式
        'num': 4,          # 分组大小
        'rho':0,           # 现在需要支持有相关性的矩阵
        'iteration': 5,    # 大迭代算法需要提供迭代次数信息
        'type':"related",
        'test_iterations': 2000,
        'p_mode': "ml",
        'i_mode': "ml",
        'fc_layer': 1,
        'L':  128,                #LCSD算法列表大小
        'distribution':[1]*5 +[2]*3+[4]*2, #节点分布
        #信道编译码新加字段
        'rate':1/3,
        'test_code':500,
        'load_dir':"Tanner_R13_K120_Z12_BG2.mat",
        'Z':12

    }
    det = Mu_detector_coding(params)

    data_node = genData_Mu(params)
    iterations = 500





    # ser_all = det.Test_func(iterations, "FC_SD")
    # print("FC_SD", ser_all)

    # ser_all = det.Test_func(iterations,"MMSE")
    ber_all = det.test_func_with_code("MMSE")
    print("MMSE",ber_all)

    # ber_all = det.test_func_with_code("ZF")
    # print("ZF", ber_all)


    # ser_all = det.test_func_with_code("LCSD")
    # print("LCSD", ser_all)

    # ser_all = det.Test_func(iterations, "LCSD")
    # print("LCSD", ser_all)