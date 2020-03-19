import matplotlib.pyplot as plt

def plotfig(SNR_list,BER_list,BLER_list):
    plt.semilogy(SNR_list,BER_list, linewidth=1,label='BER', linestyle='-',c='b')
    plt.semilogy(SNR_list,BLER_list, linewidth=1,label='BLER', linestyle='--',c='b')
    plt.legend()
    plt.grid()
    plt.show()
    plt.title('BER', fontsize=15)
    plt.xlabel('SNR(dB)', fontsize=14)
    plt.ylabel('BER/BLER', fontsize=14)
    plt.tick_params(axis='x', labelsize=10)
    plt.tick_params(axis='y', labelsize=10, width=2)
    # x = range(0, 10, 1)
    # y = range(0, 12, 1)
    # plt.xticks(x)
    # plt.yticks(y)