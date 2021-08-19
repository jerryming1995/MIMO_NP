import numpy as np
import scipy.io as scio
from Dataset import genData
import scipy.linalg as sclinalg
import h5py
import time
dt = time.localtime()
ft = '%Y%m%d%H%M%S'
nt = time.strftime(ft, dt)

class genData_Mu(genData):

    def __init__(self, params):
        super(genData_Mu, self).__init__(params)
        self.User = params['User']
        try:
            print("使用相关矩阵")
            self.rho = params['rho']  # 自相关系数
        except:
            print("使用iid矩阵")
            self.rho = 0
        self.Rt, self.Rr = self.Rcor()  # 计算发送和接收天线的自相关矩阵， 与rho有关
        self.Rt_sqrt = sclinalg.sqrtm(self.Rt)
        self.Rr_sqrt = sclinalg.sqrtm(self.Rr)

        if self.dataset_dir:
            # noinspection PyBroadException
            try:
                data_load = scio.loadmat(self.dataset_dir)
                self.H_load = data_load['H_save']
                print("load_success -v7 type, H.shape=", self.H_load.shape)
            except Exception:
                data_load = h5py.File(self.dataset_dir)
                # 读取matlab v7.3格式文件时数据类型会出错，TypeError: Cannot cast
                # array data from dtype([('real', '<f8'), ('imag', '<f8')]) to dtype('complex64') according to the rule 'unsafe'
                self.H_load = np.array(np.transpose(data_load['H_save'], [0, 2, 1, 3]), np.complex64)
                print("load_success -v7.3 type, H.shape=", self.H_load.shape)

    def Rcor(self): # 自相关矩阵,
        ranget = np.reshape(np.arange(1, self.Nt+1), [-1, 1])
        ranger = np.reshape(np.arange(1, self.Nr+1), [-1, 1])
        Rt = self.rho ** (np.abs(ranget - ranget.T))
        Rr = self.rho ** (np.abs(ranger - ranger.T))
        Rt_real = np.concatenate([np.concatenate([np.real(Rt), -1*np.imag(Rt)], axis=1), np.concatenate([np.imag(Rt), np.real(Rt)], axis=1)], axis=0)
        Rr_real = np.concatenate([np.concatenate([np.real(Rr), -1*np.imag(Rr)], axis=1), np.concatenate([np.imag(Rr), np.real(Rr)], axis=1)], axis=0)
        # return Rt_real, Rr_real
        # Rr = np.eye(self.Nr)
        # Rt = np.ones(shape=[self.Nt, self.Nt])*self.rho
        # for i in range(self.Nt):
        #     Rt[i, i] = 1
        # Rt_real = np.concatenate([np.concatenate([np.real(Rt), -1*np.imag(Rt)], axis=1), np.concatenate([np.imag(Rt), np.real(Rt)], axis=1)], axis=0)
        # Rr_real = np.concatenate([np.concatenate([np.real(Rr), -1*np.imag(Rr)], axis=1), np.concatenate([np.imag(Rr), np.real(Rr)], axis=1)], axis=0)
        return Rt_real, Rr_real


    def dataTest(self, number, snr):  # 更改为复数方法
        if self.dataset_dir:
            size = np.shape(self.H_load)    # Nt Nr number, ue
            if size[0] != self.Nt or size[1] != self.Nr or size[3] != self.User:
                raise(ValueError("H matrix does not match!!!!"))

            sigma2 = self.Nt / (np.power(10, snr / 10) * self.Nr)
            x_real = np.zeros((self.batch_size, 2 * self.Nt * self.User))
            H_real = np.zeros((self.batch_size, 2 * self.Nr, 2 * self.Nt * self.User))
            y_noise_real = np.zeros((self.batch_size, 2 * self.Nr))

            for i in range(self.User):
                index = np.random.randint(0, size[2], self.batch_size)
                H_sample = self.H_load[:, :, index, i]   # Nt,Nr,number
                H_sample = np.transpose(H_sample, (2, 1, 0)) * np.sqrt(self.Nt)  # number,Nr,Nt
                s = np.random.randint(low=0, high=np.shape(self.constellation)[0], size=[self.batch_size, self.Nt])
                x_complex = self.constellation[s]
                x_real_temp = np.concatenate([np.real(x_complex), np.imag(x_complex)], axis=1)
                x_real[:, i * 2 * self.Nt:(i + 1) * 2 * self.Nt] = x_real_temp
                Hr = np.real(H_sample)
                Hi = np.imag(H_sample)
                H_real_temp = np.concatenate([np.concatenate([Hr, -Hi], axis=2), np.concatenate([Hi, Hr], axis=2)], axis=1)
                H_real[:, :, i * 2 * self.Nt:(i + 1) * 2 * self.Nt] = H_real_temp
                noise = np.sqrt(sigma2 / 2) * np.random.randn(self.batch_size, 2 * self.Nr)
                y_noise_real += self.batch_matvec_mul(H_real_temp, x_real_temp)+noise
                # y_noise_real += self.batch_matvec_mul(H_real_temp, x_real_temp)  # 不加噪
            sigma2 = np.tile(np.expand_dims(np.expand_dims(sigma2, axis=0), axis=1), [self.batch_size, self.User])
        else:
            s = np.random.randint(low=0, high=np.shape(self.constellation)[0], size=[self.batch_size, self.Nt])  # [B, Nt], 发送多数据流
            x_complex = self.constellation[s]  # [B, Nt] 调制后信号
            x_real = np.concatenate([np.real(x_complex), np.imag(x_complex)], axis=1)  # 发送等效实数信号
            Hr = np.random.randn(self.batch_size, self.Nr, self.Nt) * np.sqrt(0.5 / self.Nr)  #
            Hi = np.random.randn(self.batch_size, self.Nr, self.Nt) * np.sqrt(0.5 / self.Nr)
            H_real = np.concatenate([np.concatenate([Hr, -Hi], axis=2), np.concatenate([Hi, Hr], axis=2)], axis=1)  # [B, 2*Nr, 2*Nt]等效实数矩阵
            H_real = self.Rr_sqrt@H_real@self.Rt_sqrt                                  #  相关信道矩阵
            y_real = self.batch_matvec_mul(H_real, x_real)         # [B, 2*Nr], 等效实数接收信号
            sigma2 = self.Nt / (np.power(10, snr / 10) * self.Nr)  # 噪声功率
            noise = np.sqrt(sigma2 / 2) * np.random.randn(self.batch_size, 2 * self.Nr)
            y_noise_real = y_real + noise
            # y_noise_real = y_real
            sigma2 = np.tile(np.expand_dims(np.expand_dims(sigma2, axis=0), axis=1), [self.batch_size, 1])
        return x_real, H_real, y_noise_real, sigma2

    def dataTrain(self, number, snr):
        if self.dataset_dir:
            size = np.shape(self.H_load)    # Nt Nr number, ue
            if size[0] != self.Nt or size[1] != self.Nr or size[3] != self.User:
                raise(ValueError("H matrix does not match!!!!"))
            sigma2 = self.Nt / (np.power(10, snr / 10) * self.Nr)
            x_real = np.zeros((self.batch_size, 2 * self.Nt * self.User))
            H_real = np.zeros((self.batch_size, 2 * self.Nr, 2 * self.Nt * self.User))
            y_noise_real = np.zeros((self.batch_size, 2 * self.Nr))
            for i in range(self.User):
                index = np.random.randint(0, size[2], self.batch_size)
                H_sample = self.H_load[:, :, index, i]   # Nt,Nr,number
                H_sample = np.transpose(H_sample, (2, 1, 0)) * np.sqrt(self.Nt)  # number,Nr,Nt
                s = np.random.randint(low=0, high=np.shape(self.constellation)[0], size=[self.batch_size, self.Nt])
                x_complex = self.constellation[s]
                x_real_temp = np.concatenate([np.real(x_complex), np.imag(x_complex)], axis=1)
                x_real[:, i * 2 * self.Nt:(i + 1) * 2 * self.Nt] = x_real_temp
                Hr = np.real(H_sample)
                Hi = np.imag(H_sample)
                H_real_temp = np.concatenate([np.concatenate([Hr, -Hi], axis=2), np.concatenate([Hi, Hr], axis=2)], axis=1)
                H_real[:, :, i * 2 * self.Nt:(i + 1) * 2 * self.Nt] = H_real_temp
                noise = np.sqrt(sigma2 / 2) * np.random.randn(self.batch_size, 2 * self.Nr)
                y_noise_real += self.batch_matvec_mul(H_real_temp, x_real_temp)+noise
                # y_noise_real += self.batch_matvec_mul(H_real_temp, x_real_temp)
            sigma2 = np.tile(np.expand_dims(np.expand_dims(sigma2, axis=0), axis=1), [self.batch_size, self.User])
        else:
            s = np.random.randint(low=0, high=np.shape(self.constellation)[0], size=[self.batch_size, self.Nt])
            x_complex = self.constellation[s]
            x_real = np.concatenate([np.real(x_complex), np.imag(x_complex)], axis=1)
            Hr = np.random.randn(self.batch_size, self.Nr, self.Nt) * np.sqrt(0.5 / self.Nr)
            Hi = np.random.randn(self.batch_size, self.Nr, self.Nt) * np.sqrt(0.5 / self.Nr)
            H_real = np.concatenate([np.concatenate([Hr, -Hi], axis=2), np.concatenate([Hi, Hr], axis=2)], axis=1)
            H_real = self.Rr_sqrt@H_real@self.Rt_sqrt
            y_real = self.batch_matvec_mul(H_real, x_real)
            sigma2 = self.Nt / (np.power(10, snr / 10) * self.Nr)
            noise = np.sqrt(sigma2 / 2) * np.random.randn(self.batch_size, 2 * self.Nr)
            y_noise_real = y_real + noise
            # y_noise_real = y_real  # 不加噪，测试代码用
            sigma2 = np.tile(np.expand_dims(np.expand_dims(sigma2, axis=0), axis=1), [self.batch_size, 1])  # shape=[B,1]
        return x_real, H_real, y_noise_real, sigma2

    def dataTrain_mu(self, number, snr):
        if self.dataset_dir:
            x_real_l, H_real_l, y_noise_real_l, sigma2_l = self.dataTrain(number, snr)

        else:
            x_real_l = np.zeros((self.batch_size, 2 * self.Nt * self.User))
            H_real_l = np.zeros((self.batch_size, 2 * self.Nr, 2 * self.Nt * self.User))
            y_noise_real_l = np.zeros((self.batch_size, 2 * self.Nr))
            sigma2_l = np.zeros((self.batch_size, 1 * self.User))
            for i in range(self.User):
                x_real, H_real, y_noise_real, sigma2 = self.dataTrain(number, snr)
                x_real_l[:, i * 2 * self.Nt:(i + 1) * 2 * self.Nt] = x_real
                H_real_l[:, :, i * 2 * self.Nt:(i + 1) * 2 * self.Nt] = H_real
                y_noise_real_l += y_noise_real
                sigma2_l[:, i:i + 1] = sigma2[:, :]
            sigma2_l = np.sum(sigma2_l,axis=1)
            sigma2_l = np.expand_dims(sigma2_l, axis=1)
            sigma2_l = np.tile(sigma2_l,[1,self.User])
        # 接收信号能量E{||Hx+n||_2^2}, 噪声能量E{||n||_2^2}， 信噪比评估SNR=E{||Hx||_2^2 / E{||n||_2^2}}, 发送信号能量
        # SNR = tr(H^HH)E{x^Hx} / NtNr\sigma^2
        # power_tx = np.mean(np.sum(np.square(x_real_l), axis=1))  # E{x^Hx}
        # H_power = np.mean(np.trace(np.matmul(np.transpose(H_real_l, [0, 2, 1]), H_real_l), axis1=1, axis2=2))  # E{tr(H^HH)}
        # power_rx = np.mean(np.sum(np.square(y_noise_real_l), axis=1))  # E{||Hx+n||_2^2}
        # sigma2 = sum([x * self.Nr for x in sigma2_l[0, :]])  # E{||n||_2^2}
        # snr_dB = 10 * np.log((power_rx - sigma2) / sigma2) / np.log(10)  # SNR=E{||Hx||_2^2 / E{||n||_2^2}}
        # print('发送信号能量{:.2f}, 信道能量{:.2f}，接收信号能量(含噪){:.2f}，噪声能量{:.2f}，信噪比{:.2f}dB'.format(
        #     power_tx, H_power, power_rx, sigma2, snr_dB))
        return x_real_l, H_real_l, y_noise_real_l, sigma2_l

    def dataTest_mu(self, number, snr):              # 暂时认为所有用户的信噪比一样
        if self.dataset_dir:
            x_real_l, H_real_l, y_noise_real_l, sigma2_l = self.dataTest(number,snr)

        else:
            x_real_l = np.zeros((self.batch_size, 2*self.Nt*self.User))
            H_real_l = np.zeros((self.batch_size, 2*self.Nr, 2*self.Nt*self.User))
            y_noise_real_l = np.zeros((self.batch_size, 2*self.Nr))
            sigma2_l = np.zeros((self.batch_size, 1*self.User))
            for i in range(self.User):
                x_real, H_real, y_noise_real, sigma2 = self.dataTest(number, snr)
                x_real_l[:, i*2*self.Nt:(i+1)*2*self.Nt] = x_real
                H_real_l[:, :, i*2*self.Nt:(i+1)*2*self.Nt] = H_real
                y_noise_real_l += y_noise_real
                sigma2_l[:, i:i+1] = sigma2[:, :]
            sigma2_l = np.sum(sigma2_l,axis=1)
            sigma2_l = np.expand_dims(sigma2_l,axis=1)
            sigma2_l = np.tile(sigma2_l, [1, self.User])
        # 接收信号能量E{||Hx+n||_2^2}, 噪声能量E{||n||_2^2}， 信噪比评估SNR=E{||Hx||_2^2 / E{||n||_2^2}}, 发送信号能量
        # SNR = tr(H^HH)E{x^Hx} / NtNr\sigma^2
        power_tx = np.mean(np.sum(np.square(x_real_l), axis=1))                                 # E{x^Hx}
        H_power = np.mean(np.trace(np.matmul(np.transpose(H_real_l, [0, 2, 1]), H_real_l), axis1=1, axis2=2))  # E{tr(H^HH)}
        power_rx = np.mean(np.sum(np.square(y_noise_real_l), axis=1))                           # E{||Hx+n||_2^2}
        # sigma2 = sum([x*self.Nr for x in sigma2_l[0, :]])                                       # E{||n||_2^2}
        snr_dB = 10*np.log((power_rx-sigma2) / sigma2) / np.log(10)                             # SNR=E{||Hx||_2^2 / E{||n||_2^2}}
        # print('发送信号能量{:.2f}, 信道能量{:.2f}，接收信号能量(含噪){:.2f}，噪声能量{:.2f}，信噪比{:.2f}dB'.format(
        #     power_tx, H_power, power_rx, sigma2, snr_dB))
        return x_real_l, H_real_l, y_noise_real_l, sigma2_l


if __name__ == "__main__":
    params = {
        'dataset_dir': None,  # 程序运行时生成数据集
        'mod_name': 'QAM_4',
        'constellation': np.array([0.7071+0.7071j, -0.7071+0.7071j,0.7071-0.7071j,-0.7071-0.7071j]),
        'Nt': 2,  # Number of transmit antennas
        'Nr': 32,  # Number of receive antennas
        'User': 8,  # Number of Users
        'batch_size': 5000,
        'rho': 0.9,
    }
    print("参数设置：", params)
    # 数据生成对象
    gen_data = genData_Mu(params)
    temp = gen_data.dataTest_mu(0,3)

    pass
    print("end")

