# import tensorflow as tf
import numpy as np
import scipy.linalg as sclinalg


def model_train(sess, nodes, data):
    feed_dict = {
        nodes['x']: data['x'],
        nodes['y']: data['y'],
        nodes['H']: data['H'],
        nodes['noise_sigma2']: data['noise_sigma2'],
    }
    sess.run(nodes['train'], feed_dict)

def model_loss(sess, nodes, data):
    feed_dict = {
        nodes['x']: data['x'],
        nodes['y']: data['y'],
        nodes['H']: data['H'],
        nodes['noise_sigma2']: data['noise_sigma2'],
    }
    ser, loss = sess.run([nodes['ser'], nodes['loss']], feed_dict)
    return ser, loss

def model_est(sess, nodes, data):
    feed_dict = {
        nodes['x']: data['x'],
        nodes['y']: data['y'],
        nodes['H']: data['H'],
        nodes['noise_sigma2']: data['noise_sigma2'],
    }
    xhat= sess.run(nodes['xhat'], feed_dict)
    return xhat

def model_eval(sess, params, detector, Data, iterations=2000):
    SNR_dBs = np.arange(params['SNR_dB_min_test'], params['SNR_dB_max_test'], params['SNR_step_test'])
    # print(SNR_dBs)
    ser_all = []
    for i in range(SNR_dBs.shape[0]):
        ser = 0.
        print('======================正在仿真SNR:%ddB================================' % (SNR_dBs[i]))
        for j in range(iterations):
            # 生成测试数据
            x_Feed, H_Feed, y_Feed, noise_sigma2_Feed = Data.dataTest(j, SNR_dBs[i])
            # print("SNR",SNR_dBs)
            # 信号检测
            feed_dict = {
                detector['x']: x_Feed,
                detector['y']: y_Feed,
                detector['H']: H_Feed,
                detector['noise_sigma2']: noise_sigma2_Feed,
                }
            ser += sess.run(detector['ser'], feed_dict) / iterations
        ser_all.append(ser)
    return ser_all

def model_debug(sess, params, detector, Data, iterations=2000):
    SNR_dBs = np.arange(params['SNR_dB_min_test'], params['SNR_dB_max_test'], params['SNR_step_test'])
    # print(SNR_dBs)
    ser_all = []
    xhat_r = []
    for i in range(SNR_dBs.shape[0]):
        ser = 0.
        print('======================正在仿真SNR:%ddB================================' % (SNR_dBs[i]))
        for j in range(iterations):
            # 生成测试数据
            x_Feed, H_Feed, y_Feed, noise_sigma2_Feed = Data.dataTest(j, SNR_dBs[i])
            # 信号检测
            feed_dict = {
                detector['x']: x_Feed,
                detector['y']: y_Feed,
                detector['H']: H_Feed,
                detector['noise_sigma2']: noise_sigma2_Feed,
                }
            ser += sess.run(detector['ser'], feed_dict) / iterations
            xhat = sess.run(detector['xhat'],feed_dict)
            xhat_r.append(xhat)
        ser_all.append(ser)
    return ser_all,xhat

def model_mu_eval(sess, params, detector, Data, iterations=2000):
    SNR_dBs = np.arange(params['SNR_dB_min_test'], params['SNR_dB_max_test'], params['SNR_step_test'])
    # print(SNR_dBs)
    ser_all = []
    for i in range(SNR_dBs.shape[0]):
        ser = 0.
        print('======================正在仿真SNR:%ddB================================' % (SNR_dBs[i]))
        for j in range(iterations):
            # 生成测试数据
            x_Feed, H_Feed, y_Feed, noise_sigma2_Feed = Data.dataTest_mu(j, SNR_dBs[i])
            # print(noise_sigma2_Feed)
            # print(H_Feed.shape)
            # 信号检测
            feed_dict = {
                detector['x']: x_Feed,
                detector['y']: y_Feed,
                detector['H']: H_Feed,
                detector['noise_sigma2']: noise_sigma2_Feed,
                }
            ser += sess.run(detector['ser'], feed_dict) / iterations
        # 测试用
        # xhat = sess.run(detector['xhat'],feed_dict)
        # print(xhat)
        # print(sess.run(detector['distance'],feed_dict))
        # print(x_Feed)
        # h_i = sess.run(detector['Hi'],feed_dict)
        # Rii = sess.run(detector['Rii'], feed_dict)
        # print(sess.run(detector['infer'],feed_dict))
        # print(sess.run(detector['Rii'], feed_dict))
        # print(H_Feed[0,:,4:8])
        # print(h_i[0,:,4:8])
        # print(sess.run(detector['Hi'],feed_dict))
        # print(sess.run(detector['eye'],feed_dict))
        # eye1 = eye[0]
        # print(eye[0,:,:])
        # print("hello")
        ser_all.append(ser)
    return ser_all

def model_mu_eval_np(params, detector_np, Data, iterations=2000):    # 如有特殊需求再补充完善 1024
    SNR_dBs = np.arange(params['SNR_dB_min_test'], params['SNR_dB_max_test'], params['SNR_step_test'])
    # print(SNR_dBs)
    ser_all = []
    for i in range(SNR_dBs.shape[0]):
        ser = 0.
        print('======================正在仿真SNR:%ddB================================' % (SNR_dBs[i]))
        for j in range(iterations):
            # 生成测试数据
            x_Feed, H_Feed, y_Feed, noise_sigma2_Feed = Data.dataTest_mu(j, SNR_dBs[i])
            # 信号检测
            ser += detector_np.MMSE(x_Feed, H_Feed, y_Feed, noise_sigma2_Feed)/iterations  #暂时只写一个MMSE的

        ser_all.append(ser)
    return ser_all


#####################################utlis for Mu_detector_np###################################
#########numpy 需要使用的一些通用调制解调函数######################
def accuracy(x,y,type="ser"):
    """
    :param x:
    :param y:
    :param type: 工作模式
    :return: 返回一个数据向量中的检测情况
    """
    # print(x)
    # print(y)

    if type=="ser":
        error = 1-np.mean(np.equal(x,y))
        # if(error!=0.0):
        #     print("x",x)
        #     print("y",y)
    elif type=="ber":
        pass
    else:
        raise(ValueError("not supported type in accuracy"))
    return error

def dataConver2complex(data,type):
    """

    :param data: 输入数据有类型
    :param type: 输入的数据类型
    :return:
    """
    pass

def Gen_data_mu(path,number,Nr,Nt,user,snr,constellation,type="iid",rho=0):
    """

    :param path: 如果是读取模式
    :param number:
    :param Nr:
    :param Nt:
    :param user:
    :param snr:
    :param constellation:
    :param type: iid:独立高斯过程; related:相关矩阵； read: 实时读取
    :param rho:  相关系数大小
    :return:
    """
    if type == "read": # read的逻辑不同如果是read读完可以直接return
        pass
        return
    Ns = Nt * user
    H_all = np.zeros((Nr,Ns),dtype=np.complex64)
    x_all = np.zeros((Ns,1),dtype=np.complex128)
    y_noise = np.zeros((Nr,1),dtype=np.complex64)
    for i in range(user):
        Hr = np.random.randn(Nr, Nt) * np.sqrt(0.5 / Nr)
        Hi = np.random.randn(Nr, Nt) * np.sqrt(0.5 / Nr)
        H = Hr + 1j * Hi
        s_index = np.random.randint(low=0, high=len(constellation), size=[Nt, 1])
        x = constellation[s_index]
        sigma2 = Nt / (np.power(10, snr / 10) * Nr)
        noise = np.sqrt(sigma2 / 2) * np.random.randn(Nr,1) + 1j * np.sqrt(sigma2 / 2) * np.random.randn(Nr,1)
        if type == "iid":
            y= H @ x
            y_noise += y+ noise
            H_all[:,i*Nt:(i+1)*Nt] = H
            x_all[i*Nt:(i+1)*Nt,0:1] = x
        elif type == "related":
            #使用相关性不是特别高的矩阵
            # print("simplified Rcor")
            Rr = np.eye(Nr)
            Rt = np.ones(shape=[Nt, Nt]) * rho
            for k in range(Nt):
                Rt[k, k] = 1

            #使用标准相关性矩阵
            # print("standard Rcor")
            # ranget = np.reshape(np.arange(1, Nt + 1), [-1, 1])
            # ranger = np.reshape(np.arange(1, Nr + 1), [-1, 1])
            # Rt = rho ** (np.abs(ranget - ranget.T))
            # Rr = rho ** (np.abs(ranger - ranger.T))

            Rt_sqrt = sclinalg.sqrtm(Rt)
            Rr_sqrt = sclinalg.sqrtm(Rr)
            H_cor = Rr_sqrt @ H @ Rt_sqrt  # 相关信道矩阵

            H = H_cor
            y = np.matmul(H_cor, x)
            y_noise += y+ noise
            H_all[:, i * Nt:(i + 1) * Nt] = H_cor
            x_all[i*Nt:(i + 1) * Nt,0:1] = x
        else:
            raise(ValueError("not supported type in Gen_Mat"))
    return x_all[:,0],y_noise[:,0],H_all

def Gen_data(path,Nr,Nt,snr,constellation,type="iid",rho=0):
    if type == "read":
        pass #待完成
        return
    Hr = np.random.randn(Nr, Nt) * np.sqrt(0.5 / Nr)
    Hi = np.random.randn(Nr, Nt) * np.sqrt(0.5 / Nr)
    H = Hr + 1j * Hi
    s_index = np.random.randint(low=0, high=len(constellation), size=[Nt, 1])
    x = constellation[s_index]
    # x = np.expand_dims(constellation,-1)
    sigma2 = Nt / (np.power(10, snr / 10) * Nr)
    noise = np.sqrt(sigma2 / 2) * np.random.randn(Nr, 1) + 1j * np.sqrt(sigma2 / 2) * np.random.randn(Nr, 1)
    if type == "iid":

        y = np.matmul(H, x)
        y_noise = y + noise

    elif type=="related":
        Rr = np.eye(Nr)
        Rt = np.ones(shape=[Nt, Nt]) * rho
        for i in range(Nt):
             Rt[i, i] = 1
        Rt_sqrt = sclinalg.sqrtm(Rt)
        Rr_sqrt = sclinalg.sqrtm(Rr)
        H_cor = Rr_sqrt @ H @ Rt_sqrt  # 相关信道矩阵
        H = H_cor
        y = np.matmul(H_cor, x)
        y_noise = y + noise
    else:
        raise(ValueError("not supported type in Gen_Mat"))
    return x[:, 0], y_noise[:, 0], H






def demodulate_np(x, constellation):
    """
    基于numpy的复数信号解调
    :param x:shape=[d1,d2, ...], dtype=np.complex64, 待解调信号
    :param constellation: shape=[d1,],
    """
    x_complex = np.reshape(x, [-1, 1])
    constellation_complex = np.reshape(constellation, [1, -1])
    d_square = np.square(np.abs(x_complex - constellation_complex))
    indices = np.argmin(d_square, axis=1)
    ans = constellation[indices]
    return ans


def x_real2complex_np(x: np, K: int, Nt: int):
    x_complex = x[:, :Nt] + 1j*x[:, Nt:2 * Nt]  # 第1个用户的复数信号
    for k in range(1, K):
        xk = x[:, (k * 2 * Nt):(k * 2 * Nt + Nt)] + 1j*x[:, (k * 2 * Nt + Nt):(k * 2 * Nt + 2 * Nt)]  # 第k个用户的复数信号
        x_complex = np.concatenate([x_complex, xk], axis=1)
    return x_complex

def batch_matvec_mul(A, b):
    """
    矩阵A与矩阵b相乘，其中A.shape=(batch_size, Nr, Nt)
    b.shape = (batch_size, Nt)
    输出矩阵C，C.shape = (batch_size, Nr)
    """
    C = np.matmul(A, np.expand_dims(b, axis=2))
    return np.squeeze(C, -1)





