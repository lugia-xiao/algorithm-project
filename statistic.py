import os
from kmm import *

if __name__ == '__main__':
    k=9
    genome = []
    for file in os.listdir("./genomes"):
        print("k=", k, "||method=kmm", "||reading", file)
        genomei = get_trans0(read_fa("./genomes/" + file)[1][0], k)
        genome.append([read_fa("./genomes/" + file)[0][0], genomei])
    name, seq_all = read_fa("test.fa")

    # statistic
    dic={}
    group=np.zeros(len(os.listdir("./genomes")))
    for file in os.listdir("./genomes"):
        dic[read_fa("./genomes/" + file)[0][0]]=0
    print(dic)
    for i in range(len(name)):
        speciei = assign0_one_way(seq_all[i], genome, k)
        dic[speciei]+=1
    print(dic)
    all=[[k,v] for k,v in dic.items()]
    tocsv(all,["specie","number"],"assigned-test.csv")