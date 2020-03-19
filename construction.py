import sys
sys.path.append('construction_method')

import ZW
import GA
import PW


def construction(N,snr,channel_con,construction_method,K,rate):
    if construction_method == 'zw':
        if channel_con == 'awgn':
            print('When the channel is AWGN, do not use the ZW construction method !')
            quit()
        else:
            information_pos = ZW.zw(N, snr, K, channel_con)
    elif construction_method == 'ga':
        if channel_con == 'bec' or channel_con == 'bsc':
            print('When the channel is BEC or BSC, do not use the GA construction method !')
            quit()
        else:
            information_pos = GA.ga(N, snr, K, rate)
    elif construction_method == 'pw':
        information_pos = PW.pw(N, K)
    elif construction_method == 'hpw':
        information_pos = PW.hpw(N, K)
    else:
        print('This is not a construction method or s have not added it to the construction_method !')
        quit()
    # print(information_pos)
    return information_pos