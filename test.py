import numpy as np
import time

import polarcodes
import construction
import plotbler
import snr_generate


# 初始值预设
run_num=1000

n = 8
K=2**(n-1)
SNR_para = [2.5,0.5,2.5]                                   # snr在bsc为转移概率，在bec为删除概率，都为float64，在awgn为Eb/N0信噪比(dB)，为列表,[start,step,end]
crc_n=0                                          #有0,4,8,12,16,32，0表示无crc，其他的多项式在CRC里面改

frozen_bit = 0
construction_method = 'ga'                  # 有zw，ga，pw, hpw
channel_con = 'awgn'                        # 有awgn，bsc和bec
decode_method = 'sc'                        # 有sc，scl, scf, fsc 和 bp            注意bp用的是带扩展因子s的，具体去function.f_hf_SMS看
para1= 4
para2= 'hf'
#                                 para1                                                               para2
# SC                              / 忽略                                                              / 忽略
# BP              最大迭代次数，int，一般为      30-70                       提前终止方法，str，有               max_iter, g_matrix, crc_es
# SCL                   列表数量, int, 华为推荐   8                              路径度量计算方法，str, 一般有       hf， exact
# SCF                  最大翻转次数，int, 推荐   16                            翻转方法，目前只翻转1阶,str，只有     llr_based
# FSC                           / 忽略                                     节点识别，有regular_node, type_node, 目前只有regular_node
#初始值预设完毕

decode_para=[para1,para2]
N=2**n
rate=K/N
SNR = snr_generate.SNR_array(SNR_para)
len_snr=SNR.size
SNR_list=list(SNR)
BER_list=[]
BLER_list=[]
time_start=time.time()

for i in range(len_snr):
    SNRi=SNR[i]
    run = 1
    error = np.array([0, 0])
    information_pos = construction.construction(N,SNRi,channel_con,construction_method,K+crc_n,rate)

    while run <= run_num:
        #print('The run_num is : ' + str(run), end='\r')
        value=polarcodes.polarcodes(n, rate, SNRi, channel_con, decode_method, decode_para, information_pos, frozen_bit,crc_n)
        error += value
        run += 1

    ber_bler=np.array([error[0] / (N * rate * run_num), error[1] / run_num])
    print('When SNR= ',SNRi,'dB, ')
    print('the BER is : ',ber_bler[0],', and the BLER is : ',ber_bler[1])
    BER_list.append(ber_bler[0])
    BLER_list.append(ber_bler[1])

time_end=time.time()
time_cost=int(time_end-time_start)
print('Time cost : ',time_cost,' s')

#绘图
plotbler.plotfig(SNR_list,BER_list,BLER_list)