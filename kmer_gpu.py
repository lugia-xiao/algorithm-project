import os
import numpy as np
import pandas as pd
import time
from numba import cuda
import numba as nb

def get_trans(seq):
    tran=[]
    for i in range(len(seq)):
        if(seq[i]=='A'):
            tran.append(0)
        if (seq[i]=='T'):
            tran.append(1)
        if (seq[i]=='C'):
            tran.append(2)
        if (seq[i]=='G'):
            tran.append(3)
    return tran

@cuda.jit
def get_kmer(seq,k,kmer):
    i=cuda.grid(1)
    gridStride = cuda.gridDim.x * cuda.blockDim.x
    N=len(seq)
    for m in range(i,N-k+1,gridStride):
        tmp=seq[m:m+k]
        index=0
        for j in range(k):
            index+=(4**(k-j-1))*tmp[j]
        kmer[m]+=index

def get_trans1(trans):
    trans1=np.zeros((4**k))
    n=len(trans)
    x=0
    y=0
    for i in range(n):
        trans1[int(trans[i])]=trans1[int(trans[i])]+1
    return (trans1/np.sum(trans1)).tolist()
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


def cos_distance(kmer1, kmer2):
    vec1 = np.array(kmer1)
    vec2 = np.array(kmer2)
    cos = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    return cos


def cos_assign(seq, genome, k):  # 注意：短序列未知方向
    kmeri1 = get_kmer(seq, k)
    kmeri2 = get_kmer(seq[::-1], k)
    result = ""
    maxi = 0
    for i in range(len(genome)):
        dis1 = cos_distance(kmeri1, genome[i][1])
        dis2 = cos_distance(kmeri2, genome[i][1])
        dis = max(dis1, dis2)
        if dis > maxi:
            maxi = dis
            result = genome[i][0]
    return result


def pearson_distance(kmer1, kmer2):
    vec1 = np.array(kmer1)
    vec2 = np.array(kmer2)
    X = np.vstack([vec1, vec2])
    d2 = np.corrcoef(X)[0][1]
    return d2


def pearson_assign(seq, genome, k):  # 注意：短序列未知方向
    kmeri1 = get_kmer(seq, k)
    kmeri2 = get_kmer(seq[::-1], k)
    result = ""
    maxi = 0
    for i in range(len(genome)):
        dis1 = pearson_distance(kmeri1, genome[i][1])
        dis2 = pearson_distance(kmeri2, genome[i][1])
        dis = max(dis1, dis2)
        if dis > maxi:
            maxi = dis
            result = genome[i][0]
    return result


def L1_distance(kmer1, kmer2):
    vec1 = np.array(kmer1)
    vec2 = np.array(kmer2)
    dist1 = np.sum(abs(vec1 - vec2))
    return dist1


def L1_assign(seq, genome, k):  # 注意：短序列未知方向
    seq = get_trans(seq)
    print(seq1)
    seq_l1 = cuda.to_device(seq)  # 存放seq到GPU device上加快读取速度
    k_mer1 = cuda.device_array(len(seq_l1) - k)
    get_kmer[2048, 512](seq_1, k, k_mer1)  # 采用2048*512的线程模式
    cuda.synchronize()
    print(k_mer1.copy_to_host())
    kmeri1 = get_trans1(k_mer1.copy_to_host())

    seq_2 = cuda.to_device(seq[::-1])  # 存放seq到GPU device上加快读取速度
    k_mer2 = cuda.device_array(len(seq_2) - k)
    get_kmer[2048, 512](seq_2, k, k_mer2)  # 采用2048*512的线程模式
    cuda.synchronize()
    kmeri2 = get_trans1(k_mer2.copy_to_host())
    result = ""
    mini = L1_distance(kmeri1, genome[0][1])
    for i in range(len(genome)):
        dis1 = L1_distance(kmeri1, genome[i][1])
        dis2 = L1_distance(kmeri2, genome[i][1])
        dis = min(dis1, dis2)
        if dis <= mini:
            mini = dis
            result = genome[i][0]
    return result


def L2_distance(kmer1, kmer2):
    vec1 = np.array(kmer1)
    vec2 = np.array(kmer2)
    dist2 = np.sqrt(np.sum((vec1 - vec2) ** 2))
    return dist2


def L2_assign(seq, genome, k):  # 注意：短序列未知方向
    seq_1 = cuda.to_device(seq)  # 存放seq到GPU device上加快读取速度
    k_mer1 = cuda.device_array(len(seq_1) - k)
    get_kmer[128, 512](seq_1, k, k_mer1)  # 采用2048*512的线程模式
    cuda.synchronize
    kmeri1 = get_trans1(k_mer1.copy_to_host())

    seq_2 = cuda.to_device(seq[::-1])  # 存放seq到GPU device上加快读取速度
    k_mer2 = cuda.device_array(len(seq_2) - k)
    get_kmer[128, 512](seq_2, k, k_mer2)  # 采用2048*512的线程模式
    cuda.synchronize
    kmeri2 = get_trans1(k_mer2.copy_to_host())
    result = ""
    mini = L2_distance(kmeri1, genome[0][1])
    for i in range(len(genome)):
        dis1 = L2_distance(kmeri1, genome[i][1])
        dis2 = L2_distance(kmeri2, genome[i][1])
        dis = min(dis1, dis2)
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


if __name__ == '__main__':
    os.chdir('C:/Users/zhoulonghao/Desktop/大三下/生物信息算法/algorithm-project-main/algorithm-project-main')
    all = []
    for k in range(3,10):
        # 只记录Gpu时间
        time_start = time.process_time()  # 记录开始时间
        genome = []
        for file in os.listdir("./genomes"):
            print("k=", k,"||method=L1", "||reading", file)
            seq=get_trans(read_fa("./genomes/" + file)[1][0]) #转化字符型seq为int型序列便于GPU处理
            seq_1=cuda.to_device(seq)           #存放seq到GPU device上加快读取速度
            k_mer= cuda.device_array(len(seq_1)-k)
            get_kmer[2048,512](seq_1,k,k_mer) #采用2048*512的线程模式
            cuda.synchronize()
            genomei = get_trans1(k_mer.copy_to_host())
            genomei = get_trans1(k_mer.copy_to_host())
            genome.append([read_fa("./genomes/" + file)[0][0], genomei])
        name, seq_all = read_fa("test.fa")
        df = pd.read_csv("valid.csv")

        # calculate accuracy
        real = 0
        for i in range(len(name)):
            speciei = L1_assign(seq_all[i], genome, k)
            real += valid(df, i, speciei)
        print("k=", k, "||method=L1", "||accuracy=", real / len(name) * 100, "%")

        # time
        time_end = time.process_time()  # 记录结束时间
        time_sum = time_end - time_start  # 计算的时间差为程序的执行时间，单位为秒/s

        all.append([k, real / len(name), time_sum])
    tocsv(all, ["k", "accuracy", "GPU_time"], "accuracy-L1_gpu.csv")

    # L2
    all = []
    for k in range(3,10):
        # 只记录cpu时间
        time_start = time.process_time()  # 记录开始时间
        genome = []
        for file in os.listdir("./genomes"):
            print("k=", k, "||method=L2","||reading", file)
            seq=get_trans(read_fa("./genomes/" + file)[1][0]) #转化字符型seq为int型序列便于GPU处理
            seq_1=cuda.to_device(seq)           #存放seq到GPU device上加快读取速度
            k_mer= cuda.device_array(len(seq_1)-k)
            get_kmer[2048,512](seq_1,k,k_mer) #采用2048*512的线程模式
            cuda.synchronize
            genomei = get_trans1(k_mer.copy_to_host())
            genome.append([read_fa("./genomes/" + file)[0][0], genomei])
        name, seq_all = read_fa("test.fa")
        df = pd.read_csv("valid.csv")

        # calculate accuracy
        real = 0
        for i in range(len(name)):
            speciei = L2_assign(seq_all[i], genome)
            real += valid(df, i, speciei)
        print("k=", k, "||method=L2", "||accuracy=", real / len(name) * 100, "%")

        # time
        time_end = time.process_time()  # 记录结束时间
        time_sum = time_end - time_start  # 计算的时间差为程序的执行时间，单位为秒/s

        all.append([k, real / len(name), time_sum])
    tocsv(all, ["k", "accuracy", "GPU_time"], "accuracy-L2_gpu.csv")