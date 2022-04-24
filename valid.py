import pandas as pd
import os
import numpy as np
#from kmm import * # used for calculate reads.fa

def valid(df,x,speciex):
    speciey = df.iloc[x, 1]
    if speciey == speciex:
      return 1
    else:
        return 0

if __name__ == '__main__':
    # get real assignment
    f=open("seq_id.map","r")
    name=[]
    specie=[]
    for line in f:
        name.append(line.split("\t")[0])
        specie.append(line.split("\t")[1].split(" ")[0])
    df=pd.DataFrame({"name":name,"specie":specie})
    df.set_index("name")
    df.to_csv("valid.csv",index=False)
    '''# used for calculate reads.fa
    # statistic
    dic = {}
    group = np.zeros(len(os.listdir("./genomes")))
    for file in os.listdir("./genomes"):
        dic[read_fa("./genomes/" + file)[0][0]] = 0
    print(dic)
    for i in range(len(df)):
        dic[df.iloc[i,1]] += 1
    all = [[k, v] for k, v in dic.items()]
    tocsv(all, ["specie", "number"], "assigned-map.csv")
    '''




