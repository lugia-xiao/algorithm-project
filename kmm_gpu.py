import numpy as np
import valid
import time
import pandas as pd
from numba import cuda
import numba as nb
import os
import math
import numpy as np


def read_fa(file):
    name = []
    seq_all = []
    file = open(file, "r")
    seqi = ""
    for line in file:
        if line.startswith('>'):
            if len(seqi) >= 1:
                seq_all.append(seqi)
                seqi = ""
            if line[1:-1].find("|") > 0:
                name.append(line[1:-1].split("|")[4].split(" ")[1])
            else:
                name.append(line[1:-1])
            continue
        elif line.startswith('A') or line.startswith('T') or line.startswith('C') or line.startswith('G'):
            seqi += line[:-1]
            continue
    seq_all.append(seqi)  # 补上最后一条序列
    file.close()
    return name, seq_all


def get_score0(seq, trans, k):  # for only one k, 后面同理
    score = 0
    dic = {'A': 0, 'T': 1, 'C': 2, 'G': 3}
    for i in range(k, len(seq) - 1):
        tmp = seq[i - k:i + 1]
        flag = True
        for j in range(k + 1):
            if tmp[j] != "A" and tmp[j] != "T" and tmp[j] != "C" and tmp[j] != "G":  # 排除不确定碱基
                flag = False
                break
        if flag == False:
            continue
        index = 0
        for j in range(k):
            index += (4 ** (k - j - 1)) * dic[tmp[j]]
        if trans[index][dic[tmp[k]]] != 0:
            score -= math.log(trans[index][dic[tmp[k]]])
        else:
            return 1000000000000
    return score


def get_score(seq, trans_all, k_all):
    result = []
    t = 0
    for k in k_all:
        trans = trans_all[t]
        score = 0
        dic = {'A': 0, 'T': 1, 'C': 2, 'G': 3}
        for i in range(k, len(seq) - 1):
            tmp = seq[i - k:i + 1]
            flag = True
            for j in range(k + 1):
                if tmp[j] != "A" and tmp[j] != "T" and tmp[j] != "C" and tmp[j] != "G":  # 排除不确定碱基
                    flag = False
                    break
            if flag == False:
                continue
            index = 0
            for j in range(k):
                index += (4 ** (k - j - 1)) * dic[tmp[j]]
            if trans[index][dic[tmp[k]]] != 0:
                score -= math.log(trans[index][dic[tmp[k]]])
            else:
                score = 1000000000000
                break
        result.append(score)
        t += 1
    result = np.sum(np.array(result))
    return result


def assign0(seq, genome, k):
    result = ""
    mini = get_score0(seq, genome[0][1], k)
    for i in range(len(genome)):
        dis1 = get_score0(seq, genome[i][1], k)
        dis2 = get_score0(seq[::-1], genome[i][1], k)
        dis = min(dis1, dis2)
        if dis <= mini:
            mini = dis
            result = genome[i][0]
    return result


def assign0_one_way(seq, genome, k):
    result = ""
    mini = get_score0(seq, genome[0][1], k)
    for i in range(len(genome)):
        dis = get_score0(seq, genome[i][1], k)
        if dis <= mini:
            mini = dis
            result = genome[i][0]
    return result


def assign(seq, genome, k):
    result = ""
    mini = get_score(seq, genome[0][1], k)
    for i in range(len(genome)):
        dis = get_score(seq, genome[i][1], k)
        if dis <= mini:
            mini = dis
            result = genome[i][0]
    return result


def tocsv(data, header, file_path):
    dataframe = pd.DataFrame(columns=header, data=data)
    dataframe.to_csv(file_path, index=False, encoding='utf-8')
    return dataframe


def valid(df, x, speciex):
    speciey = df.iloc[x, 1]
    if speciey == speciex:
        return 1
    else:
        return 0


def get_trans(seq):
    tran = []
    for i in range(len(seq)):
        if (seq[i] == 'A'):
            tran.append(0)
        if (seq[i] == 'T'):
            tran.append(1)
        if (seq[i] == 'C'):
            tran.append(2)
        if (seq[i] == 'G'):
            tran.append(3)
    return tran


@cuda.jit
def get_trans0(seq, k, trans):
    i = cuda.grid(1)
    gridStride = cuda.gridDim.x * cuda.blockDim.x
    N = len(seq)
    # for i in range(k, len(seq) - 1):
    for m in range(i, N - k - 1, gridStride):
        tmp = seq[m:m + k + 1]
        index = 0
        for j in range(k):
            index += (4 ** (k - j - 1)) * tmp[j]
        trans[m] = int((4 ** k) * tmp[k] + index)

    # trans = trans / np.sum(trans, axis=1, keepdims=True)


def get_trans1(trans):
    trans1 = np.zeros((4 ** k, 4)).astype(int)
    n = len(trans)
    x = 0
    y = 0
    for i in range(n):
        y = int(trans[i] // (4 ** k))
        x = int(trans[i] - y * (4 ** k))
        trans1[x][y] = trans1[x][y] + 1
    return (trans1 / np.sum(trans1, axis=1, keepdims=True))


if __name__ == '__main__':
    os.chdir('C:/Users/zhoulonghao/Desktop/大三下/生物信息算法/algorithm-project-main/algorithm-project-main')  # 转移地址到文件所在位置
    # kmm-single-one-way
    all = []
    for k in range(3, 10):
        time_start = time.process_time()
        genome = []
        for file in os.listdir("./genomes"):
            print("k=", k, "||method=kmm-single-one-way", "||reading", file)
            seq = get_trans(read_fa("./genomes/" + file)[1][0])  # 转化字符型seq为int型序列便于GPU处理
            seq_1 = cuda.to_device(seq)  # 存放seq到GPU device上加快读取速度
            trans = cuda.device_array(len(seq_1) - k)
            get_trans0[2048, 512](seq_1, k, trans)  # 采用2048*512的线程模式
            cuda.synchronize
            genomei = get_trans1(trans.copy_to_host())
            genome.append([read_fa("./genomes/" + file)[0][0], genomei])
        name, seq_all = read_fa("test.fa")
        df = pd.read_csv("valid.csv")
        # calculate accuracy
        real = 0
        for i in range(len(name)):
            speciei = assign0_one_way(seq_all[i], genome, k)
            real += valid(df, i, speciei)
        print("k=", k, "||method=kmm", "||accuracy=", real / len(name) * 100, "%")

        # time
        time_end = time.process_time()  # 记录结束时间
        time_sum = time_end - time_start  # 计算的时间差为程序的执行时间，单位为秒/s

        all.append([k, real / len(name), time_sum])
        tocsv(all, ["k", "accuracy", "GPU_time"], "accuracy-kmm-one_way_gpu.csv")

# kmm-single-two-way

"""if __name__ == '__main__':
    os.chdir('C:/Users/zhoulonghao/Desktop/大三下/生物信息算法/algorithm-project-main/algorithm-project-main')
    # kmm-single-one-way
    all = []
    for k in range(3, 10):
        time_start = time.process_time() 
        genome = []
        for file in os.listdir("./genomes"):
             print("k=", k, "||method=kmm-single-two-way", "||reading", file)
             seq=get_trans(read_fa("./genomes/" + file)[1][0])
             seq_1=cuda.to_device(seq)
             trans = cuda.device_array(len(seq_1)-k)
             get_trans0[2048,512](seq_1,k,trans)
             cuda.synchronize
             genomei = get_trans1(trans.copy_to_host())
             genome.append([read_fa("./genomes/" + file)[0][0], genomei])
        name, seq_all = read_fa("test.fa")
        df = pd.read_csv("valid.csv")
         # calculate accuracy
        real = 0
        for i in range(len(name)):
            speciei = assign0(seq_all[i], genome, k)
            real += valid(df, i, speciei)
        print("k=", k, "||method=kmm", "||accuracy=", real / len(name) * 100, "%")

        # time
        time_end = time.process_time()  # 记录结束时间
        time_sum = time_end - time_start  # 计算的时间差为程序的执行时间，单位为秒/s

        all.append([k, real / len(name), time_sum])
        tocsv(all, ["k", "accuracy", "GPU_time"], "accuracy-kmm-two_way_gpu.csv")"""

## 计算打分时的并行计算函数，实验时作用不明显


"""
@cuda.jit
def get_score0(seq,trans,k,score):# for only one k, 后面同理
    i=cuda.grid(1)
    gridStride = cuda.gridDim.x * cuda.blockDim.x
    N=len(seq)
    for m in range(i,N-k,gridStride):
        tmp = seq[m:m+k+1]
        index = 0
        for j in range(k):
            index += (4 ** (k - j - 1)) * tmp[j]
        if trans[index][tmp[k]]!=0:
            score[m]=-math.log(trans[index][tmp[k]])
        else :
            score[m]=1000000



def assign0(seq,genome,k):
    result=""
    seq=get_trans(seq)
    mini =100000000
    for i in range(len(genome)):
        dis1=cuda.device_array(len(seq)-k)
        dis2=cuda.device_array(len(seq)-k)
        seq_1=cuda.to_device(seq)
        seq_2=cuda.to_device(seq[::-1])
        get_score0[1024,512](seq_1,genome[i][1],k.dis1) #用GPU计算
        get_score0[1024,512](seq_2,genome[i][1],k,dis2) #用GPU计算
        cuda.synchronize()
        dis1=np.sum(dis1.copy_to_host())
        dis2=np.sum(dis2.copy_to_host())
        dis = min(dis1, dis2)
        if dis <=mini:
            mini = dis
            result = genome[i][0]
    return result

def assign0_one_way(seq,genome,k):
    result=""
    seq=get_trans(seq)
    mini = 100000000
    for i in range(len(genome)):
        dis=cuda.device_array(len(seq)-k)
        seq_1=cuda.to_device(seq)
        get_score0[1024,512](seq_1,genome[i][1],k,dis)
        cuda.synchronize()
        dis=np.sum(dis.copy_to_host())
        print(dis)
        if dis <=mini:
            mini = dis
            result = genome[i][0]
    return result

def assign(seq,genome,k):
    result=""
    mini = get_score(seq,genome[0][1],k)
    for i in range(len(genome)):
        dis = get_score(seq,genome[i][1],k)
        if dis <=mini:
            mini = dis
            result = genome[i][0]
    return result

    

