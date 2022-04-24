import numpy as np
from valid import valid
import time
import pandas as pd
import os
import math
import numpy as np

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

def get_trans0(seq,k):
    trans = np.zeros(shape=(4 ** k, 4))
    dic = {'A': 0, 'T': 1, 'C': 2, 'G': 3}
    for i in range(k, len(seq) - 1):
        tmp = seq[i - k:i + 1]
        flag = True
        for j in range(k+1):
            if tmp[j] != "A" and tmp[j] != "T" and tmp[j] != "C" and tmp[j] != "G":  # 排除不确定碱基
                flag = False
                break
        if flag == False:
            continue
        index = 0
        for j in range(k):
            index += (4 ** (k - j - 1)) * dic[tmp[j]]
        trans[index][dic[tmp[k]]] += 1
    trans = trans / np.sum(trans, axis=1, keepdims=True)
    if np.any(np.sum(trans, axis=1, keepdims=True)==0):
        print(seq)
    return trans

def get_trans(seq,k_all):# for more than one k
    trans_all=[]
    for k in k_all:
        trans = np.zeros(shape=(4 ** k, 4))
        dic = {'A': 0, 'T': 1, 'C': 2, 'G': 3}
        for i in range(k, len(seq) - 1):
            tmp = seq[i - k:i + 1]
            flag = True
            for j in range(k+1):
                if tmp[j] != "A" and tmp[j] != "T" and tmp[j] != "C" and tmp[j] != "G":  # 排除不确定碱基
                    flag = False
                    break
            if flag == False:
                continue
            index = 0
            for j in range(k):
                index += (4 ** (k - j - 1)) * dic[tmp[j]]
            trans[index][dic[tmp[k]]] += 1
        trans = trans / np.sum(trans, axis=1, keepdims=True)
        trans_all.append(trans)
    return trans_all

def get_score0(seq,trans,k):# for only one k, 后面同理
    score=0
    dic = {'A': 0, 'T': 1, 'C': 2, 'G': 3}
    for i in range(k, len(seq) - 1):
        tmp = seq[i - k:i + 1]
        flag = True
        for j in range(k+1):
            if tmp[j] != "A" and tmp[j] != "T" and tmp[j] != "C" and tmp[j] != "G":  # 排除不确定碱基
                flag = False
                break
        if flag == False:
            continue
        index = 0
        for j in range(k):
            index += (4 ** (k - j - 1)) * dic[tmp[j]]
        if trans[index][dic[tmp[k]]]!=0:
            score-=math.log(trans[index][dic[tmp[k]]])
        else:
            return 1000000000000
    return score

def get_score(seq,trans_all,k_all):
    result=[]
    t=0
    for k in k_all:
        trans=trans_all[t]
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
                score=1000000000000
                break
        result.append(score)
        t+=1
    result=np.sum(np.array(result))
    return result

def assign0(seq,genome,k):
    result=""
    mini = get_score0(seq,genome[0][1],k)
    for i in range(len(genome)):
        dis1 = get_score0(seq,genome[i][1],k)
        dis2 = get_score0(seq[::-1],genome[i][1],k)
        dis = min(dis1, dis2)
        if dis <=mini:
            mini = dis
            result = genome[i][0]
    return result

def assign0_one_way(seq,genome,k):
    result=""
    mini = get_score0(seq,genome[0][1],k)
    for i in range(len(genome)):
        dis = get_score0(seq,genome[i][1],k)
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

def get_score_softmax(seq,trans_all,k_all):
    result=[]
    t=0
    for k in k_all:
        trans=trans_all[t]
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
                score=1000000000000
                break
        result.append(score)
        t+=1
    return result

def softmax(a):
    a=np.array(a)
    c = np.max(a)
    exp_a = np.exp(a - c)  # 防止溢出的对策
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    return y


def assign_softmax(seq,genome,k):
    df=[]
    result=""
    for i in range(len(genome)):
        dis1 = [genome[i][0]]+get_score_softmax(seq,genome[i][1],k)
        df.append(dis1)
    df=pd.DataFrame(data=df)
    for i in range(1,len(k)):
        df.iloc[:,i]=softmax(df.iloc[:,i])
    mini=np.sum(np.array(df.iloc[0,1:]))
    for i in range(len(df)):
        tmp=np.sum(np.array(df.iloc[i,1:]))
        if tmp<=mini:
            mini=tmp
            result=df.iloc[i,0]
    return result

def tocsv(data,header,file_path):
    dataframe=pd.DataFrame(columns=header,data=data)
    dataframe.to_csv(file_path, index=False, encoding='utf-8')
    return dataframe

if __name__ == '__main__':
    '''
    # kmm-single-one-way
    all = []
    for k in range(3, 10):
        # 只记录cpu时间
        time_start = time.process_time()  # 记录开始时间
        genome = []
        for file in os.listdir("./genomes"):
            print("k=", k, "||method=kmm-single-one-way", "||reading", file)
            genomei = get_trans0(read_fa("./genomes/" + file)[1][0], k)
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
        tocsv(all, ["k", "accuracy", "CPU_time"], "accuracy-kmm-one_way.csv")
    '''

    # kmm-single-two-way
    all = []
    for k in range(3, 10):
        # 只记录cpu时间
        time_start = time.process_time()  # 记录开始时间
        genome = []
        for file in os.listdir("./genomes"):
            print("k=", k, "||method=kmm-single-two-way", "||reading", file)
            genomei = get_trans0(read_fa("./genomes/" + file)[1][0], k)
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
        tocsv(all, ["k", "accuracy", "CPU_time"], "accuracy-kmm-two_way.csv")

    '''
    # kmm-combination-plus
    all = []
    k_all=[[5,7],[3,5],[4,8],[8,9],[3,9],[5,9],[7,9],[3,6,9],[5,7,9]]
    for k in k_all:
        # 只记录cpu时间
        time_start = time.process_time()  # 记录开始时间
        genome = []
        for file in os.listdir("./genomes"):
            print("k=", k, "||method=kmm-combine-plus", "||reading", file)
            genomei = get_trans(read_fa("./genomes/" + file)[1][0], k)
            genome.append([read_fa("./genomes/" + file)[0][0], genomei])
        name, seq_all = read_fa("test.fa")
        df = pd.read_csv("valid.csv")

        # calculate accuracy
        real = 0
        for i in range(len(name)):
            speciei = assign(seq_all[i], genome, k)
            real += valid(df, i, speciei)
        print("k=", k, "||method=kmm", "||accuracy=", real / len(name) * 100, "%")

        # time
        time_end = time.process_time()  # 记录结束时间
        time_sum = time_end - time_start  # 计算的时间差为程序的执行时间，单位为秒/s
        all.append([k,real / len(name),time_sum])
        tocsv(all,["k","accuracy","CPU_time"],"accuracy-kmm-many_plus.csv")
    '''
    '''
    # kmm-combination-plus
    all = []
    k_all = [[5, 7], [3, 5], [4, 8], [8, 9], [3, 9], [5, 9], [7, 9], [3, 6, 9], [5, 7, 9]]
    for k in k_all:
        # 只记录cpu时间
        time_start = time.process_time()  # 记录开始时间
        genome = []
        for file in os.listdir("./genomes"):
            print("k=", k, "||method=kmm-softmax", "||reading", file)
            genomei = get_trans(read_fa("./genomes/" + file)[1][0], k)
            genome.append([read_fa("./genomes/" + file)[0][0], genomei])
        name, seq_all = read_fa("test.fa")
        df = pd.read_csv("valid.csv")

        # calculate accuracy
        real = 0
        for i in range(len(name)):
            speciei = assign_softmax(seq_all[i], genome, k)
            real += valid(df, i, speciei)
        print("k=", k, "||method=kmm", "||accuracy=", real / len(name) * 100, "%")

        # time
        time_end = time.process_time()  # 记录结束时间
        time_sum = time_end - time_start  # 计算的时间差为程序的执行时间，单位为秒/s
        all.append([k, real / len(name), time_sum])
        tocsv(all, ["k", "accuracy", "CPU_time"], "accuracy-kmm-softmax.csv")
    '''

