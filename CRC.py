class CRC:
    """循环冗余检验
    parameters
    ----------
    info : list
        需要被编码的信息
    crc_n : int, default: 32
        生成多项式的阶数
    p : list
        生成多项式
    q : list
        crc后得到的商
    check_code : list
        crc后得到的余数，即计算得到的校验码
    code : list
        最终的编码
    """

    def __init__(self, info, crc_n):
        self.info = info

        # 初始化生成多项式p
        if crc_n == 8:
            loc = [8, 2, 1, 0]
        elif crc_n ==12:
            loc = [12, 11, 3, 1, 0]
        elif crc_n == 16:
            loc = [16, 15, 2, 0]
        elif crc_n == 32:
            loc = [32, 26, 23, 22, 16, 12, 11, 10, 8, 7, 5, 2, 1, 0]
        elif crc_n == 4:
            loc = [4, 1, 0]
        else:
            print('crc_n have not been added in CRC !')
            quit()
        p = [0 for i in range(crc_n + 1)]
        for i in loc:
            p[i] = 1
        p=p[::-1]

        info = self.info.copy()
        times = len(info)

        # 左移补零
        for i in range(crc_n):
            info.append(0)
        # 除
        q = []
        for i in range(times):
            if info[i] == 1:
                q.append(1)
                for j in range(crc_n+1):
                    info[j + i] = info[j + i] ^ p[j]
            else:
                q.append(0)

        # 余数
        check_code = info[-crc_n::]

        # 生成编码
        code = self.info.copy()
        for i in check_code:
            code.append(i)

        self.crc_n = crc_n
        self.p = p
        self.q = q
        self.check_code = check_code
        self.code = code

    def detection(self):
        codes=self.info.copy()
        crc_n=self.crc_n
        info_len=len(codes)-crc_n
        info=codes[0:info_len]
        codes_1=CRC(info,crc_n)
        recodes=codes_1.code
        compare=[codes[i]^recodes[i] for i in range(len(codes))]
        cc=sum(compare)
        if cc == 0:
            value = 1
        else:
            value = 0
        return value
