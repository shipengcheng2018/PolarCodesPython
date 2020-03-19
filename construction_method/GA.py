import numpy as np

import GA_function

def ga(N,snr,information_num,rate):
    n=np.log2(N)
    sigma2=(1/(2*rate))*(10**(-snr/10))
    llr=2/sigma2

    llri = [0 for i in range(N)]
    llrcopy = llri.copy()
    llri[0] = llr
    m = 1
    while m <= N / 2:
        for k in range(m):
            llr_temp = llri[k]
            llrcopy[k*2] = GA_function.phi_inverse(1 - (1 - GA_function.phi(llr_temp))**2)
            llrcopy[2*k + 1] = llr_temp * 2
        llri = llrcopy.copy()
        #print(llri)
        m = 2 * m


    llri_array = np.array(llri)
    llri_sorted = np.argsort(llri_array)
    llri_sort = list(llri_sorted)
    information_pos = llri_sort[-information_num:]
    information_pos = sorted(information_pos)

    return information_pos