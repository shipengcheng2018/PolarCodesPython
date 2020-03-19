import numpy as np

import function
import CRC

def decoder(y,decode_method,decode_para,information_pos,frozen_bit,channel_con,snr,rate,crc_n):
    N = y.size
    n = np.log2(N)
    #G = function.generate_matrix(n)
    if channel_con == 'awgn':
        sigma=(1/np.sqrt(2*rate))*(10**(-snr/20))
        y_llr = (2 * y) / (sigma**2)
        #print(y_llr)
    elif channel_con == 'bec':
        y_1=y.copy()
        y_1[y_1 == 0] = np.infty
        y_1[y_1 == 1] = -np.infty
        y_1[y_1 == -1] = 0
        y_llr=y_1.copy()
        #print(y_llr)
    elif channel_con == 'bsc':
        a=np.log((1-snr)/snr)
        b=-a
        y_1=y.copy()
        y_1[y_1 == 0] = a
        y_1[y_1 == 1] = b
        y_llr=y_1.copy()
    else:
        print('This is not a channel name or s have not added it to the function: channel!')
        quit()
    #print(y_llr)
    #至此求得解码前需要的llr值，下面开始译码
    if decode_method == 'sc':
        if crc_n != 0:
            print('Notice: SC decoder can not use CRC !')
        u_d=sc_decoder(y_llr,information_pos,frozen_bit)
        u_d=u_d[0]
    elif decode_method == 'scl':
        if crc_n == 0:
            print('Notice: There is no CRC to check and select the right path !')
        if type(decode_para[0]) != int or decode_para[0] <= 0:
            print('Please input the right L(e.g. the number of list, para1) ')
            quit()
        u_d = scl_decoder(y_llr, information_pos, frozen_bit,decode_para,crc_n)
    elif decode_method == 'bp':
        if crc_n != 0 and decode_para[1] != 'crc_es':
            print('Notice: This BP early stopping method can not use CRC !')
        elif crc_n == 0 and decode_para[1] == 'crc_es':
            print('There is no CRC to early stop !')
            quit()
        u_d=bp_decoder(y_llr,information_pos,frozen_bit,decode_para,crc_n)
    elif decode_method == 'scf':
        if crc_n == 0:
            print('There is no CRC, so SCF decoder can not run !')
            quit()
        else:
            u_d = scf_decoder(y_llr, information_pos, frozen_bit,decode_para,crc_n)
    elif decode_method == 'fsc':
        if crc_n != 0:
            print('Notice: FSC decoder can not use CRC !')
        if decode_para[1] != 'regular_node' and decode_para[1] != 'type_node':
            print('The node process method is not right, revise the para2 !')
            quit()
        u_d=fsc_decoder(y_llr,information_pos,frozen_bit,decode_para)
    else:
        print('s have not added the decode_method in the function of decoder.decoder !')
        quit()
    return u_d

def sc_decoder(y_llr,information_pos,frozen_bit):
    N = y_llr.size
    #print(N)
    n = int(np.log2(N))
    llr_matrix=np.ones((n+1,N))
    llr_matrix[llr_matrix == 1] = float('nan')
    bit_matrix=llr_matrix.copy()
    llr_matrix[0]=y_llr
    position=[0,0,n,N]
    while function.all_num(bit_matrix[n]) == 0:
        up_llr=llr_matrix[position[0]][position[1]:position[1]+2**(position[2]-position[0])]
        up_bit=bit_matrix[position[0]][position[1]:position[1]+2**(position[2]-position[0])]
        left_llr=llr_matrix[position[0]+1][position[1]:position[1]+2**(position[2]-position[0]-1)]
        left_bit=bit_matrix[position[0]+1][position[1]:position[1]+2**(position[2]-position[0]-1)]
        right_llr=llr_matrix[position[0]+1][position[1]+2**(position[2]-position[0]-1):position[1]+2**(position[2]-position[0])]
        right_bit=bit_matrix[position[0]+1][position[1]+2**(position[2]-position[0]-1):position[1]+2**(position[2]-position[0])]
        #print(left_llr)

        if function.all_num(up_bit) == 1:
            position=function.up(position)
        else:
            if function.all_num(right_bit) == 1:
                up_bit=function.get_up_bit(left_bit,right_bit)
                bit_matrix[position[0]][position[1]:position[1] + 2 ** (position[2] - position[0])]=up_bit.copy()
            else:
                if function.all_num(right_llr) == 1:
                    if position[0] == position[2]-1:
                        right_bit_pos=position[1]+1
                        right_bit=function.get_right_bit(right_llr,information_pos,frozen_bit,right_bit_pos)
                        bit_matrix[position[0]+1][position[1]+2**(position[2]-position[0]-1):position[1]+2**(position[2]-position[0])]=right_bit
                    else:
                        position=function.rightdown(position)
                else:
                    if function.all_num(left_bit) == 1:
                        right_llr=function.get_right_llr(left_bit,up_llr)
                        llr_matrix[position[0]+1][position[1]+2**(position[2]-position[0]-1):position[1]+2**(position[2]-position[0])]=right_llr
                    else:
                        if function.all_num(left_llr) == 0:
                            left_llr = function.get_left_llr(up_llr)
                            llr_matrix[position[0] + 1][position[1]:position[1] + 2 ** (position[2] - position[0] - 1)] = left_llr
                        else:
                            if position[0] == position[2]-1:
                                left_bit_pos=position[1]
                                left_bit=function.get_left_bit(left_llr,information_pos,frozen_bit,left_bit_pos)
                                bit_matrix[position[0]+1][position[1]:position[1]+2**(position[2]-position[0]-1)]=left_bit
                            else:
                                position = function.leftdown(position)

    u_d=[bit_matrix[n],llr_matrix[n]]
    return u_d

def scl_decoder(y_llr, information_pos, frozen_bit, decode_para, crc_n):
    N = y_llr.size
    n = int(np.log2(N))
    llr_matrix = np.ones((n + 1, N))
    llr_matrix[llr_matrix == 1] = float('nan')
    bit_matrix = llr_matrix.copy()
    llr_matrix[0] = y_llr
    list_max= decode_para[0]
    pm_method=decode_para[1]
    split_pos=information_pos                                               #采用传统的方法，在所有信息位分裂，未裁剪和减复杂度
    llr_list = [llr_matrix]
    bit_list = [bit_matrix]
    #print(llr_list)
    #print(bit_list)
    pm_list = [0]
    split_loc=0
    split_len=len(split_pos)
    l_now=1

    while split_len-1 >= split_loc:
        for i in range(l_now):
            llr_matrix_temp = llr_list[i]
            bit_matrix_temp = bit_list[i]
            pm_temp = pm_list[i]

            matrix_temp = sc_stepping_decoder(llr_matrix_temp, bit_matrix_temp, information_pos, frozen_bit,
                                              split_pos[split_loc])

            llr_list[i] = matrix_temp[0]
            bit_list[i] = matrix_temp[1]
            right_pm_update = function.get_pm_update(
                matrix_temp[0][n][split_pos[split_loc - 1] + 1:split_pos[split_loc] + 1],
                matrix_temp[1][n][split_pos[split_loc - 1] + 1:split_pos[split_loc] + 1], pm_method)
            pm_list[i] = pm_temp + right_pm_update
            llr_list.append(matrix_temp[0].copy())
            bit_matrix_wrong = matrix_temp[1].copy()
            bit_matrix_wrong[n][split_pos[split_loc]] = 1 - bit_matrix_wrong[n][split_pos[split_loc]]
            bit_list.append(bit_matrix_wrong.copy())
            wrong_pm_update = function.get_pm_update(
                matrix_temp[0][n][split_pos[split_loc - 1] + 1:split_pos[split_loc] + 1],
                bit_matrix_wrong[n][split_pos[split_loc - 1] + 1:split_pos[split_loc] + 1], pm_method)
            pm_list.append(pm_temp + wrong_pm_update)

        if l_now > list_max/2:
            pm_list_arg=np.argsort(pm_list)
            del_list_arg=pm_list_arg[list_max:]
            #删减多余的列表分支
            pm_list = [pm_list[i] for i in range(len(pm_list)) if i not in del_list_arg]
            llr_list = [llr_list[i] for i in range(len(llr_list)) if i not in del_list_arg]
            bit_list = [bit_list[i] for i in range(len(bit_list)) if i not in del_list_arg]
        l_now = len(pm_list)
        split_loc += 1

    if split_pos[-1] != N-1:
        for i in range(l_now):
            llr_matrix_temp = llr_list[i]
            bit_matrix_temp = bit_list[i]
            pm_temp = pm_list[i]
            matrix_temp = sc_stepping_decoder(llr_matrix_temp, bit_matrix_temp, information_pos, frozen_bit,N-1)
            llr_list[i] = matrix_temp[0]
            bit_list[i] = matrix_temp[1]
            right_pm_update = function.get_pm_update(
                matrix_temp[0][n][split_pos[split_loc - 1]+1:N],
                matrix_temp[1][n][split_pos[split_loc - 1]+1:N], pm_method)
            pm_list[i] = pm_temp + right_pm_update

    #接下来提取list中的译码值经过crc校验,采取的是排序pm，从最小的开始经过CRC，第一个通过CRC的即作为译码值输出
    pm_argsort=np.argsort(pm_list)
    for i in pm_argsort:
        u_d_temp=bit_list[i][n]
        u_d_temp = np.array([0 if u_d_temp[i] == 0 else 1 for i in range(u_d_temp.size)])
        u_d_temp_info = u_d_temp[information_pos]
        u_d_temp_info = list(u_d_temp_info)
        crc_c = CRC.CRC(u_d_temp_info, crc_n)
        flag = crc_c.detection()
        if flag == 1:
            u_d=u_d_temp
            break
        elif flag == 0 and i == pm_argsort[-1]:
            u_d_temp=bit_list[pm_argsort[0]][n]
            u_d_temp = np.array([0 if u_d_temp[i] == 0 else 1 for i in range(u_d_temp.size)])
            u_d=u_d_temp

    return u_d

def sc_stepping_decoder(llr_matrix,bit_matrix,information_pos,frozen_bit,split_pos):           #运行到判决split_pos的位置，注意是split_pos判决完成后返回的
    N = int(bit_matrix[0].size)
    n = int(np.log2(N))
    loc=function.get_up_loc(bit_matrix)
    position = [loc[0], loc[1], n, N]

    while bit_matrix[n][split_pos] != 0 and bit_matrix[n][split_pos] != 1:
        up_llr=llr_matrix[position[0]][position[1]:position[1]+2**(position[2]-position[0])]
        up_bit=bit_matrix[position[0]][position[1]:position[1]+2**(position[2]-position[0])]
        left_llr=llr_matrix[position[0]+1][position[1]:position[1]+2**(position[2]-position[0]-1)]
        left_bit=bit_matrix[position[0]+1][position[1]:position[1]+2**(position[2]-position[0]-1)]
        right_llr=llr_matrix[position[0]+1][position[1]+2**(position[2]-position[0]-1):position[1]+2**(position[2]-position[0])]
        right_bit=bit_matrix[position[0]+1][position[1]+2**(position[2]-position[0]-1):position[1]+2**(position[2]-position[0])]

        if function.all_num(up_bit) == 1:
            position = function.up(position)
        else:
            if function.all_num(right_bit) == 1:
                up_bit = function.get_up_bit(left_bit, right_bit)
                bit_matrix[position[0]][position[1]:position[1] + 2 ** (position[2] - position[0])] = up_bit.copy()
            else:
                if function.all_num(right_llr) == 1:
                    if position[0] == position[2] - 1:                  #右底
                        right_bit_pos = position[1] + 1
                        right_bit = function.get_right_bit(right_llr, information_pos, frozen_bit, right_bit_pos)
                        bit_matrix[position[0] + 1][position[1] + 2 ** (position[2] - position[0] - 1):position[1] + 2 ** (
                                    position[2] - position[0])] = right_bit
                    else:
                        position = function.rightdown(position)
                else:
                    if function.all_num(left_bit) == 1:
                        right_llr = function.get_right_llr(left_bit, up_llr)
                        llr_matrix[position[0] + 1][position[1] + 2 ** (position[2] - position[0] - 1):position[1] + 2 ** (
                                    position[2] - position[0])] = right_llr
                    else:
                        if function.all_num(left_llr) == 0:
                            left_llr = function.get_left_llr(up_llr)
                            llr_matrix[position[0] + 1][
                            position[1]:position[1] + 2 ** (position[2] - position[0] - 1)] = left_llr
                        else:
                            if position[0] == position[2] - 1:         #左底
                                left_bit_pos = position[1]
                                left_bit = function.get_left_bit(left_llr, information_pos, frozen_bit, left_bit_pos)
                                bit_matrix[position[0] + 1][
                                position[1]:position[1] + 2 ** (position[2] - position[0] - 1)] = left_bit
                            else:
                                position = function.leftdown(position)
    return [llr_matrix,bit_matrix]

def fsc_decoder(y_llr,information_pos,frozen_bit,decode_para):
    N = y_llr.size
    n = int(np.log2(N))
    node_list = function.node_identify(N,information_pos,decode_para[1])

    llr_matrix = np.ones((n + 1, N))
    llr_matrix[llr_matrix == 1] = float('nan')
    bit_matrix = llr_matrix.copy()
    llr_matrix[0] = y_llr
    position = [0, 0, n, N]
    while function.all_num(bit_matrix[n]) == 0:
        node_col = int(np.floor(position[1]/(2**(n-position[0]))))
        if node_list[position[0]][node_col] == 'nan':
            up_llr = llr_matrix[position[0]][position[1]:position[1] + 2 ** (position[2] - position[0])]
            up_bit = bit_matrix[position[0]][position[1]:position[1] + 2 ** (position[2] - position[0])]
            left_llr = llr_matrix[position[0] + 1][position[1]:position[1] + 2 ** (position[2] - position[0] - 1)]
            left_bit = bit_matrix[position[0] + 1][position[1]:position[1] + 2 ** (position[2] - position[0] - 1)]
            right_llr = llr_matrix[position[0] + 1][
                        position[1] + 2 ** (position[2] - position[0] - 1):position[1] + 2 ** (position[2] - position[0])]
            right_bit = bit_matrix[position[0] + 1][
                        position[1] + 2 ** (position[2] - position[0] - 1):position[1] + 2 ** (position[2] - position[0])]

            if function.all_num(up_bit) == 1:
                position = function.up(position)
            else:
                if function.all_num(right_bit) == 1:
                    up_bit = function.get_up_bit(left_bit, right_bit)
                    bit_matrix[position[0]][position[1]:position[1] + 2 ** (position[2] - position[0])] = up_bit.copy()
                else:
                    if function.all_num(right_llr) == 1:
                        if position[0] == position[2] - 1:
                            right_bit_pos = position[1] + 1
                            right_bit = function.get_right_bit(right_llr, information_pos, frozen_bit, right_bit_pos)
                            bit_matrix[position[0] + 1][
                            position[1] + 2 ** (position[2] - position[0] - 1):position[1] + 2 ** (
                                        position[2] - position[0])] = right_bit
                        else:
                            position = function.rightdown(position)
                    else:
                        if function.all_num(left_bit) == 1:
                            right_llr = function.get_right_llr(left_bit, up_llr)
                            llr_matrix[position[0] + 1][
                            position[1] + 2 ** (position[2] - position[0] - 1):position[1] + 2 ** (
                                        position[2] - position[0])] = right_llr
                        else:
                            if function.all_num(left_llr) == 0:
                                left_llr = function.get_left_llr(up_llr)
                                llr_matrix[position[0] + 1][
                                position[1]:position[1] + 2 ** (position[2] - position[0] - 1)] = left_llr
                            else:
                                if position[0] == position[2] - 1:
                                    left_bit_pos = position[1]
                                    left_bit = function.get_left_bit(left_llr, information_pos, frozen_bit, left_bit_pos)
                                    bit_matrix[position[0] + 1][
                                    position[1]:position[1] + 2 ** (position[2] - position[0] - 1)] = left_bit
                                else:
                                    position = function.leftdown(position)

        else:
            up_llr = llr_matrix[position[0]][position[1]:position[1] + 2 ** (position[2] - position[0])]
            node_result = function.node_process(up_llr,node_list[position[0]][node_col])
            bit_matrix[position[0]][position[1]:position[1] + 2 ** (position[2] - position[0])] = node_result[0]
            bit_matrix[n][position[1]:position[1] + 2 ** (position[2] - position[0])] = node_result[1]
            position = function.up(position)
    u_d = bit_matrix[n]
    return u_d






def sc_flip1_decoder(y_llr,information_pos,frozen_bit,flip_pos):
    N = y_llr.size
    #print(N)
    n = int(np.log2(N))
    llr_matrix=np.ones((n+1,N))
    llr_matrix[llr_matrix == 1] = float('nan')
    bit_matrix=llr_matrix.copy()
    llr_matrix[0]=y_llr
    position=[0,0,n,N]
    while function.all_num(bit_matrix[n]) == 0:
        up_llr=llr_matrix[position[0]][position[1]:position[1]+2**(position[2]-position[0])]
        up_bit=bit_matrix[position[0]][position[1]:position[1]+2**(position[2]-position[0])]
        left_llr=llr_matrix[position[0]+1][position[1]:position[1]+2**(position[2]-position[0]-1)]
        left_bit=bit_matrix[position[0]+1][position[1]:position[1]+2**(position[2]-position[0]-1)]
        right_llr=llr_matrix[position[0]+1][position[1]+2**(position[2]-position[0]-1):position[1]+2**(position[2]-position[0])]
        right_bit=bit_matrix[position[0]+1][position[1]+2**(position[2]-position[0]-1):position[1]+2**(position[2]-position[0])]

        if function.all_num(up_bit) == 1:
            position=function.up(position)
        else:
            if function.all_num(right_bit) == 1:
                up_bit=function.get_up_bit(left_bit,right_bit)
                bit_matrix[position[0]][position[1]:position[1] + 2 ** (position[2] - position[0])]=up_bit.copy()
            else:
                if function.all_num(right_llr) == 1:
                    if position[0] == position[2]-1:
                        right_bit_pos=position[1]+1
                        right_bit=function.get_right_bit_flip1(right_llr,information_pos,frozen_bit,right_bit_pos,flip_pos)
                        bit_matrix[position[0]+1][position[1]+2**(position[2]-position[0]-1):position[1]+2**(position[2]-position[0])]=right_bit
                    else:
                        position=function.rightdown(position)
                else:
                    if function.all_num(left_bit) == 1:
                        right_llr=function.get_right_llr(left_bit,up_llr)
                        llr_matrix[position[0]+1][position[1]+2**(position[2]-position[0]-1):position[1]+2**(position[2]-position[0])]=right_llr
                    else:
                        if function.all_num(left_llr) == 0:
                            left_llr = function.get_left_llr(up_llr)
                            llr_matrix[position[0] + 1][position[1]:position[1] + 2 ** (position[2] - position[0] - 1)] = left_llr
                        else:
                            if position[0] == position[2]-1:
                                left_bit_pos=position[1]
                                left_bit=function.get_left_bit_flip1(left_llr,information_pos,frozen_bit,left_bit_pos,flip_pos)
                                bit_matrix[position[0]+1][position[1]:position[1]+2**(position[2]-position[0]-1)]=left_bit
                            else:
                                position = function.leftdown(position)

    u_d=[bit_matrix[n],llr_matrix[n]]
    return u_d

def scf_decoder(y_llr, information_pos, frozen_bit,decode_para,crc_n):
    # 第一次用SC译码的过程
    N = y_llr.size
    n = int(np.log2(N))
    u_d_1_list = sc_decoder(y_llr,information_pos,frozen_bit)
    u_d_1_llr=u_d_1_list[1].copy()
    u_d_1=u_d_1_list[0].copy()
    u_d_1 = np.array([0 if u_d_1[i] == 0 else 1 for i in range(u_d_1.size)])
    u_d_1_info=u_d_1[information_pos]
    u_d_1_llr_info=u_d_1_llr[information_pos]
    #print(u_d_1_llr_info)
    u_d_1_info=list(u_d_1_info)
    #print(u_d_1)
    crc_c = CRC.CRC(u_d_1_info, crc_n)
    flag = crc_c.detection()
    #print(flag)
    flip_info_pos1 = list(np.argsort(np.abs(u_d_1_llr_info)))
    flip_info_pos = flip_info_pos1[0:decode_para[0]]
    information_pos_array=np.array(information_pos)
    flip_pos=information_pos_array[flip_info_pos]
    #print(flip_pos)
    if flag == 0:
        T=1
        while T <= decode_para[0]:
            #print(T)
            u_d_2_list=sc_flip1_decoder(y_llr,information_pos,frozen_bit,flip_pos[T-1])
            u_d_2_llr = u_d_2_list[1].copy()
            u_d_2 = u_d_2_list[0].copy()
            u_d_2 = np.array([0 if u_d_2[i] == 0 else 1 for i in range(u_d_2.size)])
            u_d_2_info=u_d_2[information_pos]
            u_d_2_info=list(u_d_2_info)
            crc_c = CRC.CRC(u_d_2_info, crc_n)
            flag1 = crc_c.detection()
            if flag1 == 1:
                u_d = np.array(u_d_2)
                #print('The scf decoder flip: ',T,' times and it works.')
                break
            else:
                T += 1
                u_d = np.array(u_d_1)
    else:
        #print('no flip and success')
        u_d=np.array(u_d_1)
    return u_d

def bp_decoder(y_llr,information_pos,frozen_bit,decode_para,crc_n):
    N = y_llr.size
    n = int(np.log2(N))

    if decode_para[1] == 'max_iter':

        bp_max_iter=int(decode_para[0])
        #初始化左传播和右传播矩阵
        left_matrix=np.zeros([N,n+1])
        right_matrix=np.zeros([N,n+1])
        left_matrix[:,n]=y_llr
        temp_value=(1-2*frozen_bit)*np.infty
        temp=[temp_value if i not in information_pos else 0 for i in range(N)]
        right_matrix[:,0]=temp

        for iter in range(bp_max_iter):
            for i in range(n):
                left_matrix[:,n-i-1]=function.bp_update_left(left_matrix[:,n-i],right_matrix[:,n-i-1],n-i)

            for i in range(n):
                right_matrix[:,i+1]=function.bp_update_right(left_matrix[:,i+1],right_matrix[:,i],i+1)


        u_d_llr=left_matrix[:,0]+right_matrix[:,0]
        u_d=[0 if u_d_llr[i] >= 0 else 1 for i in range(N)]
        u_d=np.array(u_d)


    elif decode_para[1] == 'g_matrix':

        # 初始化左传播和右传播矩阵
        left_matrix = np.zeros([N, n + 1])
        right_matrix = np.zeros([N, n + 1])
        left_matrix[:, n] = y_llr
        temp_value = (1 - 2 * frozen_bit) * np.infty
        temp = [temp_value if i not in information_pos else 0 for i in range(N)]
        right_matrix[:, 0] = temp
        flag=0
        iter_num=1

        while flag == 0 and iter_num <= decode_para[0]:
            for i in range(n):
                left_matrix[:, n - i - 1] = function.bp_update_left(left_matrix[:, n - i], right_matrix[:, n - i - 1],n - i)

            for i in range(n):
                right_matrix[:, i + 1] = function.bp_update_right(left_matrix[:, i + 1], right_matrix[:, i], i + 1)

            u_d_llr = left_matrix[:, 0] + right_matrix[:, 0]
            u_d = [0 if u_d_llr[i] >= 0 else 1 for i in range(N)]
            x_d_llr = left_matrix[:, n] + right_matrix[:, n]
            x_d = [0 if x_d_llr[i] >= 0 else 1 for i in range(N)]
            G = function.generate_matrix(n)
            x_g = u_d * G
            x_g = np.array(x_g % 2)
            #print(x_g)
            cor = [0 if x_g[0][i]==x_d[i] else 1 for i in range(N)]
            if sum(cor) == 0:
                flag = 1
            else:
                flag = 0
            iter_num += 1
        u_d = np.array(u_d)


    elif decode_para[1] == 'crc_es':

        # 初始化左传播和右传播矩阵
        left_matrix = np.zeros([N, n + 1])
        right_matrix = np.zeros([N, n + 1])
        left_matrix[:, n] = y_llr
        temp_value = (1 - 2 * frozen_bit) * np.infty
        temp = [temp_value if i not in information_pos else 0 for i in range(N)]
        right_matrix[:, 0] = temp
        flag = 0
        iter_num = 1

        while flag == 0 and iter_num <= decode_para[0]:
            for i in range(n):
                left_matrix[:, n - i - 1] = function.bp_update_left(left_matrix[:, n - i], right_matrix[:, n - i - 1],n - i)

            for i in range(n):
                right_matrix[:, i + 1] = function.bp_update_right(left_matrix[:, i + 1], right_matrix[:, i], i + 1)

            u_d_llr = left_matrix[:, 0] + right_matrix[:, 0]
            u_d = np.array([0 if u_d_llr[i] >= 0 else 1 for i in range(N)])
            u_d_info = u_d[information_pos]
            u_d_info = list(u_d_info)
            crc_c=CRC.CRC(u_d_info,crc_n)
            flag=crc_c.detection()
            # if flag == 1:
            #     print('It work!',iter_num)

            iter_num += 1
        u_d = np.array(u_d)
    else:
        print('This is not a early stopping method of BP decoder or s have not added it to the endecoder.bp_decoder !')
        quit()
    return u_d













