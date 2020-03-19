import numpy as np

def pw(N,information_num):
    n=int(np.log2(N))
    beta=2**0.25
    channels=np.zeros(N)
    for i in range(N):
        bin_seq=np.binary_repr(i,n)
        sum=0
        for j in range(n):
            if bin_seq[j] == '1':
                sum+= beta**(n-j-1)
        channels[i]=sum

    pw_array = np.array(channels)
    pw_sorted = np.argsort(pw_array)
    pw_sort = list(pw_sorted)
    information_pos = pw_sort[-information_num:]
    information_pos = sorted(information_pos)
    return information_pos

def hpw(N,information_num):
    n = int(np.log2(N))
    beta = 2 ** 0.25
    channels = np.zeros(N)
    for i in range(N):
        bin_seq = np.binary_repr(i, n)
        sum = 0
        for j in range(n):
            if bin_seq[j] == '1':
                sum += beta ** (n - j - 1)+ 0.25 * (beta**(0.25*(n-j-1)))
        channels[i] = sum

    hpw_array = np.array(channels)
    hpw_sorted = np.argsort(hpw_array)
    hpw_sort = list(hpw_sorted)
    information_pos = hpw_sort[-information_num:]
    information_pos = sorted(information_pos)
    return information_pos