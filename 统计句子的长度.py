import matplotlib.pyplot as plt
nums={}
with open("./data/train.txt",mode="r",encoding="utf-8") as fr:
    for line in fr:
        line=line.strip()
        if line!="":
            for word in line.split("--xhm--")[0].split():
                k=len(word)


                if k in nums:
                    nums[k]+=1
                else:
                    nums[k]=1

# -*- coding: utf-8 -*-
x=[]
y=[]
for k,v in nums.items():
    x.append(k)
    y.append(v)

plt.bar(x, y)
plt.show()

