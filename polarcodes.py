import numpy as np
import sys
sys.path.append('construction_method')

import function
import decoder
import CRC

def polarcodes(n,rate,snr,channel_con,decode_method,decode_para,information_pos,frozen_bit,crc_n):
    N = 2 ** n
    information_num = int(N * rate)
    # 信息-information_bit生成
    information_bit = np.random.randint(0, 2, size=(1, information_num))

    if crc_n == 0:
        pass
    else:
        informationbit=information_bit.copy()
        informationbit.resize(information_num,)
        information_bit_list=list(informationbit)
        crc_info=CRC.CRC(information_bit_list,crc_n)
        crc_information_bit=crc_info.code
        crc_information_bit=np.array(crc_information_bit)
        crc_information_bit.resize(1,information_num+crc_n)

    # 编码前序列-u生成
    u = np.ones((1, N)) * frozen_bit
    j = 0
    #print(u.size)
    #print(information_bit.size)
    if crc_n == 0:
        for i in information_pos:
            u[0][i] = information_bit[0][j]
            j += 1
    else:
        for i in information_pos:
            u[0][i] = crc_information_bit[0][j]
            j += 1

    #print(information_pos)
    # 生成矩阵-G生成
    G = function.generate_matrix(n)

    # 编码比特-x生成
    x = u*G
    x = np.array(x % 2)

    # 经过信道生成y
    y = function.channel(x, channel_con, snr, rate)

    # y进入译码器生成u_d
    u_d = decoder.decoder(y, decode_method, decode_para, information_pos, frozen_bit, channel_con, snr, rate, crc_n)
    #print(u_d)
    # 计算错误数
    information_pos=information_pos[0:information_num]
    information_bit_d = u_d[information_pos]
    error_num = int(np.sum((information_bit_d + information_bit) % 2))
    if error_num != 0:
        decode_fail=1
    else:
        decode_fail=0
    r_value=np.array([error_num,decode_fail])


    return r_value


