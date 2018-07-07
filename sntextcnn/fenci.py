import jieba
if  __name__ == '__main__':
    with open('/erp/CLOUD_DISK/notebook/Me/taxCode.txt','r') as f:
        lines_list=[]
        #devide lines into items
        for line in f.readlines():
            line_list=line.split("\t")[0].split(" ")
            label=line.split("\t")[-1]
            tmp=""
            for item in line_list:
                tmp+=" ".join(jieba.cut(item))
            tmp=tmp+" "+label
            lines_list.append(tmp)
    #feed into word2vec model
    #model=Word2Vec(lines_list,size=300,window=5,min_count=1,workers=2)
    with open('/taxCodeChange.txt','w') as f:
        for line in lines_list:
            f.write(line+"\n")