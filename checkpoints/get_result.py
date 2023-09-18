import pandas as pd 

fi= pd.read_csv("test_results.csv")
import sys

label_txt=open("./src/based-models/scripts/icd_coding/labels.txt","r")
labels=[]
for i in label_txt.readlines():
    labels.append(i.strip())


k=int(sys.argv[1])
import heapq
s=0
t=0
dic={}
dic2={}
for row in fi.iterrows():
    probs_list=eval(row[1]["probas"])
    real_value=int(row[1]["labels"]) 
    if real_value not in dic:
        dic[real_value]=0
        dic2[real_value]=0
    top_k=heapq.nlargest(k,range(len(probs_list)),probs_list.__getitem__)
    if real_value in top_k:
        s+=1
        dic2[real_value]+=1
    t+=1
    dic[real_value]+=1

print(float(s)/t)	   


for num,i in enumerate(dic):
    print("{}: {},totalcount:{}".format(labels[num],float(dic2[i])/dic[i],dic[i]))
