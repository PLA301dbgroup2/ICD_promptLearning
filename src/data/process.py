from keybert import KeyBERT
import pandas as pd
from flair.embeddings import TransformerDocumentEmbeddings

def get_pure(l):
    res=set()
    for ele in l:
        single_grams=ele[0].split(" ")
        for gram in single_grams:
            res.add(gram)
    return res
        
bert = TransformerDocumentEmbeddings('/home/sr/pretrain/bert/')
kw_model = KeyBERT(model=bert)

f=open("else.csv","w")
f.write("text,label\n")
cou=0
data=pd.read_excel("./tt.xlsx")
for row in data.iterrows():
    cou+=1
    if cou%100==0:
        print(cou)  
    row=row[1]
    docs=row["text"]
   
    n=int(len(docs)/20)
    keywords = kw_model.extract_keywords(docs, keyphrase_ngram_range=(1, 3), use_mmr=True, diversity=1, top_n=n)
    keywords=[i[0] for i in keywords]
    f.write("\""+",".join(keywords)+"\"")
    code=row["label"]
    f.write(","+code+"\n")
f.close()

text=pd.read_csv("else.csv")

text["text"]=text["text"].str[0:512]

text.to_csv(name+".csv",index=False)                                              
