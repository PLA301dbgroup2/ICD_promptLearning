import pandas as pd
disease_list=[]
f=open("label.txt","r")
for line in f.readlines():
    disease_list.append(line.strip())
print(disease_list)

disease_list.remove("其他心血管疾病")
data=pd.read_csv("out_formatted.csv",sep="\t",error_bad_lines=False)
data=data.sample(frac=1.0)
#data["text"]=data["text"].str.split("初步诊断").str[0]


for dis in disease_list:
    data.loc[data['label'].str.contains(dis,na=False),'label']=dis

data.loc[data['label'].str.contains("Ⅲ度房室传导阻滞",na=False),'label']="3度房室传导阻滞"
data.loc[data['label'].str.contains("恶性高血压",na=False),'label']="高血压急症"

data.loc[data['label'].str.contains("爆发型心肌炎",na=False),'label']="暴发性心肌炎"
data.loc[data['label'].str.contains("暴发型心肌炎",na=False),'label']="暴发性心肌炎"

data.loc[~data['label'].isin(disease_list),"label"]="其他心血管疾病"

data=data[data.label!="其他心血管疾病"]


data.to_csv("final.csv",sep=",",index=False)




