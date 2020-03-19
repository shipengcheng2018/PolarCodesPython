def writetxt(N,K,SNR_para,crc_n,decode_method,BER_list,BLER_list,time_cost,construction_method,decode_para):
    if crc_n == 0:
        txt_line1 = 'Polar('+ str(N) + ','+ str(K)+','+ construction_method.upper()+',' + decode_method.upper() +'), '+ 'decode_para='+'['+str(decode_para[0])+','+str(decode_para[1])+']. '+'SNR= '+ str(
            SNR_para[0]) + ' : ' + str(SNR_para[1]) + ' : ' + str(SNR_para[2])+ '. No CRC.' + ' Time Cost: '+str(time_cost)+'s'+'\n'
    else:
        txt_line1 = 'Polar('+ str(N) + ','+ str(K)+','+ construction_method.upper()+',' + decode_method.upper() +'), '+ 'decode_para='+'['+str(decode_para[0])+','+str(decode_para[1])+']. '+'SNR= '+ str(
            SNR_para[0]) + ' : ' + str(SNR_para[1]) + ' : ' + str(SNR_para[2])+ '. CRC-' + str(crc_n) +'. Time Cost: '+str(time_cost)+'s'+ '\n'
    text = 'BER: ' + str(BER_list) + '\n' + 'BLER: ' + str(BLER_list) + '\n'

    with open('ber_bler.txt','a') as txtfile:
        txtfile.write(txt_line1)
        txtfile.write(text)