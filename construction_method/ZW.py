import numpy as np

def zw(N,snr,information_num,channel_con):
    n=np.log2(N)
    if channel_con == 'bec':
        zw=snr
    else:
        zw=2*np.sqrt(snr*(1-snr))
    zwi=[0 for i in range(N)]
    zwi[0]=zw
    m=1
    while m <= N/2:
        for k in range(1,m+1,1):
            z_temp=zwi[k-1]
            zwi[k-1]=2*z_temp-z_temp**2
            zwi[k+m-1]=z_temp**2
        m=2*m
    zwi_array=np.array(zwi)
    zwi_sorted=np.argsort(zwi_array)
    zwi_sort=list(zwi_sorted)
    information_pos=zwi_sort[0:information_num]
    information_pos=sorted(information_pos)

    return information_pos