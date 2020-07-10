# 修改pc_width的话要同时修改address_calculation部分
pc_width = 32
block_length_width = 4
header_width = 2

format_1_header = 1
format_2_header = 2
format_3_header = 3
format_4_header = 4

shift_iset = 4
width_iset = 2

set_size = 2
way_size = 2
Nset = 1 << set_size
Nway = 1 << way_size

mask = 0x0000007f
is_branch = 0x00000063
is_jal = 0x0000006f
is_jalr = 0x00000067

class LBP():
    '''
        no struct in python
        so use class
    '''
    def __init__(self):
        self.iset = [0 for i in range(Nway * Nset)]
        self.iway = [0 for i in range(Nway * Nset)]
lbp = LBP()

class BDC():
    def __init__(self):
        self.state = [[0 for i in range(Nway)] for i in range(Nset)]
        self.block_address = [['00000000000000000000000000000000' for i in range(Nway)] for i in range(Nset)]
        self.block_length = [['0000' for i in range(Nway)] for i in range(Nset)]

    # def BDC_print(self):
    #     print(self.state)
    #     print(self.block_address)
    #     print(self.block_length)
bdc = BDC()

class Text():
    '''
        the main class of the procession
    '''
    def __init__(self, filename1, filename3):
        '''

        :param filename1: the name of file1
        :param filename3: the name of file3
        '''
        self.filename1 = filename1
        self.filename3 = filename3
        self.address_dict = {}
        # 考虑到需要用到行号遍历，这里加入一个行号索引的（adderss, instruction）元组
        self.line_dict = {}
        self.lines = []
        self.column = 0
        self.type = 0
        self.historybuffer = 0
        self.result = ''
        self.current_address = '00000000000000000000000000000000'
        self.not_end = True
        self.max_num = 0


    def file2dict(self):
        '''
        the filter of file1;
        transform it into a address_index_dict: {address, (line, instruction)}
        and a line_index_dict: {line, address}
        :return: none
        '''
        line_index = 0
        file = open(self.filename1, 'r')
        while True:
            text_line = file.readline()
            # text_line = str(file.readline())
            # print(text_line)
            if text_line:
                list_line = text_line.split()
                if len(list_line) >= 2 and list_line[0][0].isdigit():
                    if list_line[1][0].isalnum():
                        line_index += 1
                        new_line = list_line[0] + list_line[1]
                        new_line = new_line.replace(":", "  ")
                        # print(new_line)
                        list_line[0] = list_line[0].replace(":", "")
                        self.address_dict[list_line[0]] = (line_index, list_line[1])
                        self.line_dict[line_index] = (list_line[0], list_line[1])
                        # with open('result.txt', 'a') as result_line:
                        #     result_line.write(new_line)
                        #     result_line.write('\n')
            else:
                break
        file.close()

    def file2lines(self):
        '''
        convert the file3 into a single line of 1-bit-data
        store into the lines
        :return:none
        '''
        file = open(self.filename3, 'r')
        lines = str(file.readlines())
        for i in range(0, len(lines)):
            if lines[i].isdigit():
                self.lines.append(lines[i])
                self.max_num += 1

    def inc_byte(self):
        '''
        increase the index by one bit
        :return: none
        '''
        self.column += 1
        if self.column >= self.max_num:
            self.not_end = False

    def get_byte(self, lenth):
        '''
        read lenth-bit elements
        :param lenth:
        :return:lenth-bit data
        '''
        self.result = self.lines[self.column]
        self.inc_byte()
        for i in range(1, lenth):
            self.result += self.lines[self.column]
            self.inc_byte()
        return self.result

    def type_judgment(self):
        '''
        read header_width-bits data and give the type of this instruction in file3
        :return: none
        '''
        typenum = self.get_byte(header_width)
        if typenum == '00':
            self.type = format_1_header
        elif typenum == '01':
            self.type = format_2_header
        elif typenum == '10':
            self.type = format_3_header
        elif typenum == '11':
            self.type = format_4_header

    def replay(self, block_address, block_length):
        '''
        find the line with block_address and write it and the next block_length-1 lines into file4
        :param block_address:
        :param block_length:
        :return:
        '''
        line = self.address_dict[block_address][0]
        for i in range(0, block_length):
            with open('file4.txt', 'a') as result_line:
                result_line.write(self.line_dict[line + i][0])
                result_line.write('\n')
            self.current_address = self.line_dict[line + i][0]

    def update_lbp(self, iset, iway):
        if self.type == format_2_header:
            lbp.iset[self.historybuffer] = iset
            lbp.iway[self.historybuffer] = iway
        self.historybuffer = (iset << way_size) | iway

    def update_bdc(self, iset, block_address, block_length):
        for i in range(Nway):
            if bdc.state[iset][i] == 0:
                bdc.block_address[iset][i] = block_address
                bdc.block_length[iset][i] = block_length
                bdc.state[iset][i] = 1
                if sum(bdc.state[iset]) == Nway:
                    for j in range(Nway):
                        bdc.state[iset][j] = 0
                break

    def get_iset(self, block_address, block_length):
        '''
        just use this function in format_header_3 and _4
        :param block_address:
        :param block_length:
        :return:
        '''
        iset = 0
        for i in range(0, width_iset):
            iset = iset * 2 + int(block_address[pc_width - shift_iset - i - 1] == block_length[block_length_width - shift_iset - i - 1])
        return iset

    def address_calculation(self):
        '''

        :return:
        '''
        instruction_num = int(self.address_dict[self.current_address][1], 16)
        temp = instruction_num & mask
        instruction = bin(instruction_num).replace('0b', '').zfill(pc_width)
        if(temp == is_branch):
            offset = instruction[0] + instruction[24] + instruction[1:7] + instruction[20:24]
            if instruction[0] == '1':
                offset = '1' * 19 + offset
            block_address = int(self.current_address, 16) + int(offset, 2) * 2
        elif(temp == is_jal):
            offset = instruction[0] + instruction[12:20] + instruction[11] + instruction[1:11]
            block_address = int(self.current_address, 16) + int(offset, 2) * 2
        elif(temp == is_jalr):
            offset = instruction[0:12]
            block_address = int(self.current_address, 16) + int(offset, 2)
        block_address = bin(block_address).replace('0b', '').zfill(pc_width)
        if len(block_address) > pc_width:
            length_temp = len(block_address)
            block_address = block_address[length_temp - pc_width : length_temp]
        return block_address

    def format_header_1(self):
        iset = lbp.iset[self.historybuffer]
        iway = lbp.iway[self.historybuffer]
        self.update_lbp(iset, iway)
        block_address = bdc.block_address[iset][iway]
        block_address = hex(int(block_address, 2)).replace('0x', '').zfill(8)
        block_length = bdc.block_length[iset][iway]
        block_length = int(block_length, 2)
        self.replay(block_address, block_length)

    def format_header_2(self):
        iset = self.get_byte(set_size)
        iset = int(iset, 2)
        iway = self.get_byte(way_size)
        iway = int(iway, 2)
        self.update_lbp(iset, iway)
        block_address = bdc.block_address[iset][iway]
        block_address = hex(int(block_address, 2)).replace('0x', '').zfill(8)
        block_length = bdc.block_length[iset][iway]
        block_length = int(block_length, 2)
        self.replay(block_address, block_length)

    def format_header_3(self):
        block_address = self.address_calculation()
        block_length = self.get_byte(4)
        iset = self.get_iset(block_address, block_length)
        self.update_bdc(iset, block_address, block_length)
        block_address = hex(int(block_address, 2)).replace('0x', '').zfill(8)
        block_length = int(block_length, 2)
        self.replay(block_address, block_length)

    def format_header_4(self):
        block_address = self.get_byte(32)
        block_length = self.get_byte(4)
        self.get_iset(block_address, block_length)
        iset = self.get_iset(block_address, block_length)
        self.update_bdc(iset, block_address, block_length)
        block_address = hex(int(block_address, 2)).replace('0x', '').zfill(8)
        block_length = int(block_length, 2)
        self.replay(block_address, block_length)

if __name__ == "__main__":
    # 每次运行程序自动清空file4
    with open("file4.txt", "r+") as file:
        file.truncate(0)
    text = Text('file1.dump', 'file3_compress.txt')
    text.file2dict()
    text.file2lines()
    while text.not_end:
        text.type_judgment()
        if text.type == 1:
            text.format_header_1()
        elif text.type == 2:
            text.format_header_2()
        elif text.type == 3:
            text.format_header_3()
        elif text.type == 4:
            text.format_header_4()