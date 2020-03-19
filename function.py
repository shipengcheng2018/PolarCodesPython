import numpy as np

def generate_matrix(n):
    F=np.matrix([[1,0],[1,1]])
    n_i=1
    G=F
    while n_i < n:
        G=np.kron(G,F)
        n_i+=1
    return G

def channel(x,channel_con,snr,rate):
    if channel_con == 'bsc':
        y_1=np.random.choice([0, 1],size=x.size,p=[1-snr,snr])
        y=(y_1+x)%2
    elif channel_con == 'bec':
        y_1=np.random.choice([-2, 0], size=x.size,p=[snr,1-snr])
        y_2=y_1+x
        y_2[y_2 == -2] = -1
        y=y_2.copy()
    elif channel_con == 'awgn':
        sigma=(1/np.sqrt(2*rate))*(10**(-snr/20))
        #print(sigma)
        #print(x)
        #print(1-2*x)
        y=(1-2*x)+np.random.randn(x.size)*sigma
        #在这里用bpsk将0映射为1,1映射为-1
    else:
        print('This is not a channel name or s have not added it to the function: channel!')
        quit()
    return y

def all_num(x):
    length=x.size
    flag=1
    for i in range(length):
        if np.isnan(x[i]) == True:
            flag = 0
            break
    return flag

def leftdown(position):
    p0=position[0]+1
    p1=position[1]
    p2=position[2]
    p3=position[3]
    p=[p0,p1,p2,p3]
    return p

def rightdown(position):
    p0=position[0]+1
    p1=position[1]+2**(position[2]-1-position[0])
    p2=position[2]
    p3=position[3]
    p=[p0,p1,p2,p3]
    return p

def up(position):
    p0=position[0]-1
    p1_t=np.floor(position[1]/(2**(position[2]-position[0]+1)))*(2**(position[2]-position[0]+1))
    p1=int(p1_t)
    p2=position[2]
    p3=position[3]
    p=[p0,p1,p2,p3]
    return p

def get_up_bit(left_bit,right_bit):
    length=left_bit.size
    #print(left_bit)
    #print(right_bit)
    temp=np.array([(left_bit+right_bit)%2,right_bit])
    temp.resize((1,2*length))
    return temp

def get_right_bit(right_llr,information_pos,frozen_bit,right_bit_pos):
    if right_bit_pos in information_pos:
        if right_llr > 0:
            temp=0
        else:
            temp=1
    else:
        temp=frozen_bit
    return temp

def get_right_bit_flip1(right_llr,information_pos,frozen_bit,right_bit_pos,flip_pos):
    if right_bit_pos in information_pos:
        if right_bit_pos == flip_pos:
            if right_llr > 0:
                temp=1
            else:
                temp=0
        else:
            if right_llr > 0:
                temp = 0
            else:
                temp = 1
    else:
        temp=frozen_bit
    return temp

def get_right_llr(left_bit,up_llr):
    length=int(left_bit.size)
    temp=np.array([g(up_llr[i],up_llr[i+length],left_bit[i]) for i in range(length)])
    return temp

def get_left_bit(left_llr,information_pos,frozen_bit,left_bit_pos):
    if left_bit_pos in information_pos:
        #print(left_llr)
        if left_llr >= 0:
            temp = 0
        else:
            temp = 1
    else:
        temp = frozen_bit
    return temp

def get_left_bit_flip1(left_llr,information_pos,frozen_bit,left_bit_pos,flip_pos):
    if left_bit_pos in information_pos:
        if left_bit_pos == flip_pos:
            if left_llr >= 0:
                temp = 1
            else:
                temp = 0
        else:
            if left_llr >= 0:
                temp = 0
            else:
                temp = 1
    else:
        temp = frozen_bit
    return temp

def get_left_llr(up_llr):
    length=int(up_llr.size/2)
    temp=np.array([f_hf(up_llr[i],up_llr[i+length]) for i in range(length)])
    return temp

def f(L1,L2):
    temp=np.log((1+np.exp(L1+L2))/(np.exp(L1)+np.exp(L2)))
    return temp

def f_hf(L1,L2):
    #硬件友好型Hard-Friendly的
    #print(L1)
    s1=np.sign(L1)
    s2=np.sign(L2)
    if s1 == 0:
        s1=1
    if s2 == 0:
        s2=1
    temp=s1*s2*np.min([np.abs(L1),np.abs(L2)])
    return temp

def f_hf_SMS(L1,L2):
    # 硬件友好型Hard-Friendly的，有一个扩展因子s
    s=0.9375
    s1 = np.sign(L1)
    s2 = np.sign(L2)
    if s1 == 0:
        s1 = 1
    if s2 == 0:
        s2 = 1
    temp = s*s1 * s2 * np.min([np.abs(L1), np.abs(L2)])
    return temp

def g(L1,L2,U1):
    temp=(1-2*U1)*L1+L2
    return temp

def element_update_left(left,right):
    value=np.zeros(2)
    value[0]=f_hf_SMS(right[1]+left[1],left[0])
    value[1]=f_hf_SMS(left[0],right[0])+left[1]
    return value

def element_update_right(left,right):
    value=np.zeros(2)
    value[0]=f_hf_SMS(right[1]+left[1],right[0])
    value[1]=f_hf_SMS(left[0],right[0])+right[1]
    return value


def bp_update_left(left_array,right_array,left_array_n):
    N=left_array.size
    interval=2**(left_array_n-1)
    num=int(N/(interval*2))
    value=np.zeros(N)
    for i in range(num):
        for j in range(interval):
            left_ele=np.zeros(2)
            right_ele=np.zeros(2)
            left_ele[0]=left_array[2*i*interval+j]
            left_ele[1]=left_array[2*i*interval+j+interval]
            right_ele[0] = right_array[2 * i * interval + j]
            right_ele[1] = right_array[2 * i * interval + j + interval]
            get_value=element_update_left(left_ele,right_ele)
            value[2*i*interval+j]=get_value[0]
            value[2*i*interval+j+interval]=get_value[1]
    return value

def bp_update_right(left_array,right_array,left_array_n):
    N=left_array.size
    interval=2**(left_array_n-1)
    num=int(N/(interval*2))
    value=np.zeros(N)
    for i in range(num):
        for j in range(interval):
            left_ele=np.zeros(2)
            right_ele=np.zeros(2)
            left_ele[0]=left_array[2*i*interval+j]
            left_ele[1]=left_array[2*i*interval+j+interval]
            right_ele[0] = right_array[2 * i * interval + j]
            right_ele[1] = right_array[2 * i * interval + j + interval]
            get_value=element_update_right(left_ele,right_ele)
            value[2*i*interval+j]=get_value[0]
            value[2*i*interval+j+interval]=get_value[1]
            #print('\n',get_value)

    return value

def get_up_loc(bit_matrix):
    N=int(bit_matrix[0].size)
    n=int(np.log2(N))
    detect_array=bit_matrix[n]
    for i in range(N):
        if detect_array[i] == 1 or detect_array[i] == 0:
            pass
        else:
            detect = i-1
            break
    if detect%2 == 0:
        loc_row = n - 1
        loc_col=detect
    else:
        loc_row = n - 1
        loc_col = detect-1
    if detect == -1:
        loc_row=0
        loc_col=0
    #print(loc_row)
    #print(loc_col)
    return [loc_row,loc_col]

def get_pm_update(llr_array,bit_array,pm_method):
    l=llr_array.size
    pm=0
    if pm_method == 'exact':
        for i in range(l):
            pm=pm+np.log(1+np.exp(-1*(1-2*bit_array[i])*llr_array[i]))
    elif pm_method == 'hf':
        for i in range(l):
            if np.sign(llr_array[i]) != np.sign(1-2*bit_array[i]):
                pm += np.abs(llr_array[i])
    else:
        print('Notice: This is not a PM compute method or s have not added it to the function.get_pm_update !')
    return pm

def node_identify(N,information_pos,node_method):
    n = int(np.log2(N))
    init=['nan']
    node_list = [init]
    for i in range(n):
        node_list.append(node_list[-1]*2)
    node_down=[1 if i in information_pos else 0 for i in range(N)]
    node_down2=node_list[-2]
    for i in range(int(N/2)):
        if node_down[i*2] == 0 and node_down[i*2+1] == 0:
            node_down2[i]= 'rate0'
        elif node_down[i*2] == 0 and node_down[i*2+1] == 1:
            node_down2[i]= 'rep'
        elif node_down[i*2] == 1 and node_down[i*2+1] == 1:
            node_down2[i]= 'rate1'
        else:
            pass
    node_list[-1]=node_down
    node_list[-2] = node_down2

    if node_method == 'regular_node':
        depth=list(range(n-1))
        depth.reverse()
        for i in depth:
            for j in range(2**i):
                if node_list[i+1][j*2] == 'rate0' and node_list[i+1][j*2+1] == 'rate0':
                    node_list[i][j] = 'rate0'
                elif node_list[i+1][j*2] == 'rate0' and node_list[i+1][j*2+1] == 'rep':
                    node_list[i][j] = 'rep'
                elif node_list[i + 1][j * 2] == 'rate1' and node_list[i + 1][j * 2 + 1] == 'rate1':
                    node_list[i][j] = 'rate1'
                elif node_list[i+1][j*2] == 'spc' and node_list[i+1][j*2+1] == 'rate1':
                    node_list[i][j] = 'spc'
                elif node_list[i+1][j*2] == 'rep' and node_list[i+1][j*2+1] == 'rate1':
                    if i == depth[0]:
                        node_list[i][j] = 'spc'
                else:
                    pass
    else:
        print('s have not added the Type_1-5 nodes in function.node_process')
        quit()

    return node_list

def node_process(up_llr,node_id):
    N = up_llr.size
    #print(N)
    n = int(np.log2(N))
    if node_id == 'rate0':
        bit_depth_i=np.zeros(N)
        bit_depth_n=bit_depth_i
    elif node_id == 'rep':
        llr_sum = np.sum(up_llr)
        if llr_sum >= 0:
            temp = 0
            bit_depth_i = np.zeros(N)
        else:
            temp = 1
            bit_depth_i = np.ones(N)
        bit_depth_n = np.zeros(N)
        bit_depth_n[-1] = temp
    elif node_id == 'spc':
        bit_depth_i = np.zeros(N)
        for i in range(N):
            if up_llr[i] >= 0:
                bit_depth_i[i] = 0
            else:
                bit_depth_i[i] = 1
            #print(bit_depth_i)
            #print(i)
        if np.sum(bit_depth_i)%2 != 0:
            flip_index=np.argmin(np.abs(up_llr))
            bit_depth_i[flip_index] = (bit_depth_i[flip_index]+1)%2
        bit_depth_n = bit_depth_i*generate_matrix(n) %2
        bit_depth_n = np.array(bit_depth_n)
    elif node_id == 'rate1':
        bit_depth_i = np.zeros(N)
        for i in range(N):
            if up_llr[i] >= 0:
                bit_depth_i[i] = 0
            else:
                bit_depth_i[i] = 1
        bit_depth_n = bit_depth_i * generate_matrix(n) % 2
        bit_depth_n = np.array(bit_depth_n)

    return [bit_depth_i,bit_depth_n]


