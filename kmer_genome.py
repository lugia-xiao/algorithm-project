import os
import numpy as np
import pandas as pd
from valid import valid
import time
from main import *

def get_kmer(seq,k):
    kmer=np.zeros(4**k)
    dic={'A':0,'T':1,'C':2,'G':3}
    m=0
    for i in range(0, len(seq) - k + 1):
        tmp=seq[i:i+k]
        flag=True
        for j in range(k):
            if tmp[j]!="A" and tmp[j]!="T" and tmp[j]!="C" and tmp[j]!="G":#排除不确定碱基
                flag=False
                m+=1
                break
        if flag==False:
            continue
        index=0
        for j in range(k):
            index+=(4**(k-j-1))*dic[seq[i+j]]
        kmer[index] += 1
    kmer=kmer/(len(seq)-k+1-m)
    return kmer.tolist()

def read_fa(file):
    name=[]
    seq_all=[]
    file = open(file,"r")
    seqi = ""
    for line in file:
        if line.startswith('>'):
            if len(seqi)>=1:
                seq_all.append(seqi)
                seqi = ""
            if line[1:-1].find("|")>0:
                name.append(line[1:-1].split("|")[4].split(" ")[1])
            else:
                name.append(line[1:-1])
            continue
        elif line.startswith('A') or line.startswith('T') or line.startswith('C') or line.startswith('G'):
            seqi+=line[:-1]
            continue
    seq_all.append(seqi)#补上最后一条序列
    file.close()
    return name,seq_all

def cos_distance(kmer1,kmer2):
    vec1=np.array(kmer1)
    vec2 = np.array(kmer2)
    cos = np.dot(vec1,vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    return cos
def cos_assign(seq,genome,k):#注意：短序列未知方向
    kmeri1=get_kmer(seq,k)
    kmeri2 = get_kmer(seq[::-1], k)
    result=""
    maxi=0
    for i in range(len(genome)):
        dis1=cos_distance(kmeri1,genome[i][1])
        dis2=cos_distance(kmeri2,genome[i][1])
        dis=max(dis1,dis2)
        if dis>maxi:
            maxi=dis
            result=genome[i][0]
    return result

def pearson_distance(kmer1,kmer2):
    vec1=np.array(kmer1)
    vec2 = np.array(kmer2)
    X = np.vstack([vec1, vec2])
    d2 = np.corrcoef(X)[0][1]
    return d2
def pearson_assign(seq,genome,k):#注意：短序列未知方向
    kmeri1=get_kmer(seq,k)
    kmeri2 = get_kmer(seq[::-1], k)
    result=""
    maxi=0
    for i in range(len(genome)):
        dis1=pearson_distance(kmeri1,genome[i][1])
        dis2=pearson_distance(kmeri2,genome[i][1])
        dis=max(dis1,dis2)
        if dis>maxi:
            maxi=dis
            result=genome[i][0]
    return result

def L1_distance(kmer1,kmer2):
    vec1=np.array(kmer1)
    vec2 = np.array(kmer2)
    dist1 = np.sum(abs(vec1-vec2))
    return dist1
def L1_assign(seq,genome,k):#注意：短序列未知方向
    kmeri1=get_kmer(seq,k)
    kmeri2 = get_kmer(seq[::-1], k)
    result=""
    mini=L1_distance(kmeri1,genome[0][1])
    for i in range(len(genome)):
        dis1=L1_distance(kmeri1,genome[i][1])
        dis2=L1_distance(kmeri2,genome[i][1])
        dis=min(dis1,dis2)
        if dis<=mini:
            mini=dis
            result=genome[i][0]
    return result

def L2_distance(kmer1,kmer2):
    vec1=np.array(kmer1)
    vec2 = np.array(kmer2)
    dist2 = np.sqrt(np.sum((vec1 - vec2)**2))
    return dist2
def L2_assign(seq,genome,k):#注意：短序列未知方向
    kmeri1=get_kmer(seq,k)
    kmeri2 = get_kmer(seq[::-1], k)
    result=""
    mini=L2_distance(kmeri1,genome[0][1])
    for i in range(len(genome)):
        dis1=L2_distance(kmeri1,genome[i][1])
        dis2=L2_distance(kmeri2,genome[i][1])
        dis=min(dis1,dis2)
        if dis<=mini:
            mini=dis
            result=genome[i][0]
    return result

def tocsv(data,header,file_path):
    dataframe=pd.DataFrame(columns=header,data=data)
    dataframe.to_csv(file_path, index=False, encoding='utf-8')
    return dataframe

if __name__ == '__main__':
    '''# cos
    all=[]
    for k in range(3,10):
        #只记录cpu时间
        time_start = time.process_time()  # 记录开始时间
        genome = []
        for file in os.listdir("./genomes"):
            print("k=",k,"||method=cos","||reading", file)
            genomei = get_kmer(read_fa("./genomes/" + file)[1][0], k)
            genome.append([read_fa("./genomes/" + file)[0][0], genomei])
        name, seq_all = read_fa("test.fa")
        df = pd.read_csv("valid.csv")

        # calculate accuracy
        real = 0
        for i in range(len(name)):
            speciei = cos_assign(seq_all[i], genome, k)
            real += valid(df, i, speciei)
        print("k=", k, "||method=cos","||accuracy=", real / len(name) * 100, "%")

        # time
        time_end = time.process_time() # 记录结束时间
        time_sum = time_end - time_start  # 计算的时间差为程序的执行时间，单位为秒/s

        all.append([k,real / len(name),time_sum])
    tocsv(all,["k","accuracy","CPU_time"],"accuracy-cos.csv")

    #pearson correlation
    all = []
    for k in range(3,10):
        # 只记录cpu时间
        time_start = time.process_time()  # 记录开始时间
        genome = []
        for file in os.listdir("./genomes"):
            print("k=", k, "||method=pearson","||reading", file)
            genomei = get_kmer(read_fa("./genomes/" + file)[1][0], k)
            genome.append([read_fa("./genomes/" + file)[0][0], genomei])
        name, seq_all = read_fa("test.fa")
        df = pd.read_csv("valid.csv")

        # calculate accuracy
        real = 0
        for i in range(len(name)):
            speciei = pearson_assign(seq_all[i], genome, k)
            real += valid(df, i, speciei)
        print("k=", k, "||method=pearson", "||accuracy=", real / len(name) * 100, "%")

        # time
        time_end = time.process_time()  # 记录结束时间
        time_sum = time_end - time_start  # 计算的时间差为程序的执行时间，单位为秒/s

        all.append([k, real / len(name), time_sum])
    tocsv(all, ["k", "accuracy", "CPU_time"], "accuracy-pearson.csv")'''

    # L1
    all = []
    for k in range(3,10):
        # 只记录cpu时间
        time_start = time.process_time()  # 记录开始时间
        genome = []
        for file in os.listdir("./genomes"):
            print("k=", k,"||method=L1", "||reading", file)
            genomei = get_kmer(read_fa("./genomes/" + file)[1][0], k)
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
    tocsv(all, ["k", "accuracy", "CPU_time"], "accuracy-L1.csv")

    # L2
    all = []
    for k in range(3,10):
        # 只记录cpu时间
        time_start = time.process_time()  # 记录开始时间
        genome = []
        for file in os.listdir("./genomes"):
            print("k=", k, "||method=L2","||reading", file)
            genomei = get_kmer(read_fa("./genomes/" + file)[1][0], k)
            genome.append([read_fa("./genomes/" + file)[0][0], genomei])
        name, seq_all = read_fa("test.fa")
        df = pd.read_csv("valid.csv")

        # calculate accuracy
        real = 0
        for i in range(len(name)):
            speciei = L2_assign(seq_all[i], genome, k)
            real += valid(df, i, speciei)
        print("k=", k, "||method=L2", "||accuracy=", real / len(name) * 100, "%")

        # time
        time_end = time.process_time()  # 记录结束时间
        time_sum = time_end - time_start  # 计算的时间差为程序的执行时间，单位为秒/s

        all.append([k, real / len(name), time_sum])
    tocsv(all, ["k", "accuracy", "CPU_time"], "accuracy-L2.csv")
