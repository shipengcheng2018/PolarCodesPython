import numpy as np

def SNR_array(SNR):
    if len(SNR) == 3:
        value=np.linspace(SNR[0],SNR[2],int((SNR[2]-SNR[0])/SNR[1]+1))
        value1=[round(value[i],2) for i in range(value.size)]
        value1=np.array(value1)
    elif len(SNR) == 2:
        value = np.linspace(SNR[0],SNR[1],int((SNR[1]-SNR[0])/0.1+1))
        value1=[round(value[i],2) for i in range(value.size)]
        value1 = np.array(value1)
    elif len(SNR) == 1:
        value=np.array([SNR[0]])
        value1=[round(value[i],2) for i in range(value.size)]
        value1 = np.array(value1)
    else:
        print('Input the right SNR value')
        quit()
    return value1