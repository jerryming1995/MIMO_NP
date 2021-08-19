import numpy as np
import scipy.io as sio
import sys

def Parameter_Gen(load_dir):
    # try:
    data = sio.loadmat(load_dir)
    BG1 =  data['H']
    # except:
    #     print("无法读取数据")
    #     sys.exit()

    N = BG1[0][0]  # length of the code
    M = BG1[0][1]  # number of rows of original check matrix
    wc = BG1[1][0]  # maximum column weight
    wr = BG1[1][1]  # maximum row weight
    var_degree = np.array(BG1[2, 0:N])
    check_degree = np.array(BG1[3, 0:M])
    var_index = np.zeros((N, wc), dtype=int)
    for n in range(N):
        var_index[n, :] = BG1[4 + n, 0:wc]
    check_index = np.zeros((M, wr), dtype=int)
    for m in range(M):
        check_index[m, :] = BG1[N + 4 + m, 0:wr]

    params = {'N':N,
              'M':M,
              'wc':wc,
              'wr':wr,
              'var_degree':var_degree,
              'check_degree':check_degree,
              'var_index': var_index,
              'check_index':check_index,
              'H':BG1

    }

    return params


def check_ldpc_code(codeseq,params):                    # codeseq是一个N长度的LDPC码
    """"检验解码是否正确"""
    sum_constraint = 0

    check_degree = params['check_degree']
    check_index = params['check_index']
    M= params["M"]

    # 不满足监督关系式的个数
    for i in range(M):
        temp = 0
        for j in range(check_degree[i]):
            temp ^= codeseq[check_index[i, j] - 1]

        if temp != 0:
            sum_constraint += 1

    return sum_constraint


def construct_generate_matrix(H, params):
    """通过H求解生成矩阵G"""


    M = params["M"]
    N = params["N"]


    global exchange_col
    tempH = H  # 对tempH进行高斯消去，形成tempH = [P|I]
    col_record = list(range(N))  # 记录交换列的位置
    exchange_num = 0
    for row_cnt in range(M - 1, -1, -1):  # 从最后一行开始
        handling_col = N - (M - row_cnt)
        row_place = row_cnt
        for row_temp in range(row_cnt, -1, -1):
            if tempH[row_temp, handling_col] == 1:
                break
            else:
                row_place = row_place - 1

        # if this column has not bit 1
        if row_place == -1:
            exchange_col_findflag = 0
            for exchange_col in range(handling_col - 1, -1, -1):
                row_place = row_cnt
                for row_temp in range(row_cnt, -1, -1):
                    if tempH[row_temp, exchange_col] == 1:
                        exchange_col_findflag = 1
                        break
                    else:
                        row_place = row_place - 1
                if exchange_col_findflag == 1:
                    break

            if exchange_col_findflag == 1:
                exchange_num = exchange_num + 1

                temp = col_record[handling_col]
                col_record[handling_col] = col_record[exchange_col]
                col_record[exchange_col] = temp

                temp = tempH[:, handling_col]
                tempH[:, handling_col] = tempH[:, exchange_col]
                tempH[:, exchange_col] = temp
            else:
                print("error! K is not consistent with the definition! need examine the coding program!")

        if row_place != row_cnt:
            tempH[row_cnt, :] = np.bitwise_xor(tempH[row_cnt, :], tempH[row_place, :])

        for row_temp in range(M - 1, -1, -1):
            if tempH[row_temp, handling_col] == 1 and row_temp != row_cnt:
                tempH[row_temp, :] = np.bitwise_xor(tempH[row_temp, :], tempH[row_cnt, :])

    K = N - M  # 信息比特长度
    P = tempH[0:M, 0:K]  # 得到一个M * K阶矩阵
    Q = P.T  # Q是一个K * M阶矩阵
    G = np.zeros((K, N), dtype=int)  # 生成G是一个K * N阶矩阵
    G[0:K, 0:K] = np.eye(K, dtype=int)
    G[:, K:] = Q
    return G


def get_ldpc_code(baseband_bit, generated_matrix):
    # generated_matrix = construct_generate_matrix()
    output_bit = np.mod(np.dot(baseband_bit, generated_matrix), 2)
    output_bit = np.array(output_bit, dtype=int)

    # test_flag = check_ldpc_code(output_bit)            # 用来测试生成的LDPC码
    # if test_flag == 1:                                 # 需要注意的是一次只能测一个码，不能一次测多个码
    #     print("error!")

    return output_bit





def decode_algorithm_NMS(bitsoft,max_iteration,params):               # bitsoft是软信息
    """MinSum算法"""



    wr =params['wr']
    M = params['M']
    N = params['N']
    check_degree = params['check_degree']
    check_index = params['check_index']

    bitsoft = np.squeeze(bitsoft,axis=0)


    alpha = 0.7  # NMS算法的系数
    outseq = np.zeros(N, dtype=int)


    # hard decision
    for i in range(N):
        if bitsoft[i] > 0:
            outseq[i] = 0
        else:
            outseq[i] = 1
    if check_ldpc_code(outseq,params) == 0:
        return outseq

    # NMS Algorithm
    p = np.array(bitsoft, dtype=float)          # 初始化最大似然后验概率
    r = np.zeros((M, wr), dtype=float)          # 初始化校验节点传向变量节点的消息
    for k in range(max_iteration):
        p1 = p
        p = np.zeros(N, dtype=float)
        for i in range(M):
            sign = 1
            pos = -1
            min1 = 1e10       # 最小值
            min2 = 1e10       # 次最小值
            sgn_value = []
            for j in range(check_degree[i]):
                tempd = p1[check_index[i, j] - 1] - r[i, j]
                if tempd < 0:
                    sgn_value.append(-1)
                    sign = 0 - sign
                    tempd = 0 - tempd
                else:
                    sgn_value.append(1)
                if tempd < min1:
                    min2 = min1
                    min1 = tempd
                    pos = j
                else:
                    if tempd < min2:
                        min2 = tempd

            for j in range(check_degree[i]):
                if j == pos:
                    r[i, j] = min2 * alpha
                else:
                    r[i, j] = min1 * alpha
                r[i, j] = sign * sgn_value[j] * r[i, j]
                p[check_index[i, j] - 1] += r[i, j]

        # hard decision
        for i in range(N):
            p[i] = p[i] + bitsoft[i]
            if p[i] > 0:
                outseq[i] = 0
            else:
                outseq[i] = 1


        if check_ldpc_code(outseq,params) == 0:
            break

    return outseq

def convert(params):
    N = params["N"]
    M = params["M"]
    check_degree = params["check_degree"]
    check_index = params["check_index"]
    H_t = params["H"]

    H = np.zeros((M, N), dtype=int)  # 校验矩阵H

    for m in range(M):
        for n in range(check_degree[m]):
            H[m, check_index[m, n] - 1] = 1

    return H








if __name__ == '__main__':


    load_dir = "Tanner_R13_K120_Z12_BG2.mat"
    params = Parameter_Gen(load_dir)
    params["Z"] =12  #暂时手动设置

    H_1 = convert(params)
    G = construct_generate_matrix(H_1, params)


    K= params["N"] - params["M"]
    print(K)
    seed =7
    np.random.seed(seed)

    test_code = 500
    res = []

    points = 5
    snr_start = -4
    interval = 0.5

    for point in range(points):
        snr = snr_start+interval*(point)
        print(snr)
        sigma = np.sqrt(1.0 / 2 * 10 ** (-snr / 10))
        ber = 0
        for k in range(test_code):

            send = np.random.randint(0,2,(1,K))



            trans_ldpc = get_ldpc_code(send, G)


            #打孔
            Z= params["Z"]
            trans_punc = trans_ldpc[:,2*Z:]
            #BPSK调制
            BPSK = 1 - 2*trans_punc
            BPSK = np.array(BPSK,dtype =np.float32)
            #信道
            BPSK = BPSK+sigma * np.random.randn(1,BPSK.shape[1])

            #MIMO-OFDM过程demo中不提供

            # 补零
            zero_llr= np.zeros((1,2*Z))
            receive = np.concatenate((zero_llr,BPSK),axis=1)





            #解调
            #取得LLR
            #译码：
            decode_bit = decode_algorithm_NMS(receive,20,params)
            ber += np.mean(abs(decode_bit-trans_ldpc))
            ber_c = ber/(k+1)
            if k%20==0:
                print("已完成%d帧数据译码，码块误比特率为"%k,ber_c)
                print("")
        print("SNR:",snr)
        print("ber:",ber/test_code)
        res.append(ber/test_code)

    print("误比特率",res)













