import tensorflow as tf
import numpy  as  np
"""
一些会共用的操作

"""

def batch_matvec_mul(self, A, b, transpose_a=False):
    """
    矩阵A与矩阵b相乘，其中A.shape=(batch_size, Nr, Nt)
    b.shape = (batch_size, Nt)
    输出矩阵C，C.shape = (batch_size, Nr)
    """
    C = tf.matmul(A, tf.expand_dims(b, axis=2), transpose_a=transpose_a)
    return tf.squeeze(C, -1)

def demodulate(self, x):
    """
    信号解调(复数域解调）
    Input:
    y: 检测器检测后的信号 Tensor(shape=(batchsize, 2*Nt), dtype=float32)
    constellation: 星座点，即发送符号可能的离散值 Tensor(shape=(np.sqrt(调制阶数), ), dtype=float32)
    Output:
    indices: 解调后的基带数据信号，Tensor(shape=(batchsize, Nt)， dtype=tf.int32)
    """
    x_complex = tf.complex(x[:, 0: self.Nt], x[:, self.Nt: self.Nt * 2])
    x_complex = tf.reshape(x_complex, shape=[-1, 1])
    constellation = tf.reshape(self.constellation, [1, -1])
    constellation_complex = tf.reshape(tf.complex(constellation, 0.)
                                       - tf.complex(0., tf.transpose(constellation)), [1, -1])
    indices = tf.cast(tf.argmin(tf.abs(x_complex - constellation_complex), axis=1), tf.int32)
    indices = tf.reshape(indices, shape=tf.shape(x_complex))
    return indices

def accuracy(self, x, y):
    """
    Computes the fraction of elements for which x and y are equal
    """
    return tf.reduce_mean(tf.cast(tf.equal(x, y), tf.float32))




# ************************************** 基于numpy的常用方法 *******************************************
def accuracy_np(x,y,type="ser"):
    """
    :param x:
    :param y:
    :param type: 工作模式
    :return: 返回一个数据向量中的检测情况
    """
    if type=="ser":
        error = np.mean(np.equal(x,y))
    elif type=="ber":
        pass
    else:
        raise(ValueError("not supported type in accuracy"))
    return error

def Gen_data(path,Nr,Nt,snr,constellation,type="iid"):
    if type=="iid":
        Hr = np.random.randn(Nr, Nt) * np.sqrt(0.5 / Nr)

        Hi = np.random.randn(Nr, Nt) * np.sqrt(0.5 / Nr)

        H = Hr + 1j * Hi

        s_index = np.random.randint(low=0, high=len(constellation),size=[Nt,1])

        x =  constellation[s_index]

        sigma2 = Nt / (np.power(10, snr / 10) * Nr)

        y =  np.matmul(H,x)

        y_noise = y + np.sqrt(sigma2)* np.random.randn(Nr,1)

    elif type=="read":
        pass

    else:
        raise(ValueError("not supported type in Gen_Mat"))

    return x[:,0],y_noise[:,0],H






if __name__ == "__main__":
   pass


