
from tokenization import FullTokenizer


'''

{"postag": [{"word": "查尔斯", "pos": "nr"}, {"word": "·", "pos": "w"}, {"word": "阿兰基斯", "pos": "nr"}, {"word": "（", "pos": "w"}, {"word": "Charles Aránguiz", "pos": "nz"}, {"word": "）", "pos": "w"}, {"word": "，", "pos": "w"}, {"word": "1989年4月17日", "pos": "t"}, {"word": "出生", "pos": "v"}, {"word": "于", "pos": "p"}, {"word": "智利圣地亚哥", "pos": "ns"}, {"word": "，", "pos": "w"}, {"word": "智利", "pos": "ns"}, {"word": "职业", "pos": "n"}, {"word": "足球", "pos": "n"}, {"word": "运动员", "pos": "n"}, {"word": "，", "pos": "w"}, {"word": "司职", "pos": "v"}, {"word": "中场", "pos": "n"}, {"word": "，", "pos": "w"}, {"word": "效力", "pos": "v"}, {"word": "于", "pos": "p"}, {"word": "德国", "pos": "ns"}, {"word": "足球", "pos": "n"}, {"word": "甲级", "pos": "a"}, {"word": "联赛", "pos": "n"}, {"word": "勒沃库森足球俱乐部", "pos": "nt"}], 
 "text": "查尔斯·阿兰基斯（Charles Aránguiz），1989年4月17日出生于智利圣地亚哥，智利职业足球运动员，司职中场，效力于德国足球甲级联赛勒沃库森足球俱乐部", 
 "spo_list": [{"predicate": "出生地", "object_type": "地点", "subject_type": "人物", "object": "圣地亚哥", "subject": "查尔斯·阿兰基斯"}, {"predicate": "出生日期", "object_type": "Date", "subject_type": "人物", "object": "1989年4月17日", "subject": "查尔斯·阿兰基斯"}]}

'''

tokenizer=FullTokenizer(vocab_file="./bert_model/chinese_L-12_H-768_A-12/vocab.txt",
                        pos_vocab_file="./data/postag.txt",
                        label_vocab_file="./data/labels.txt")


class DataProcess:

    @staticmethod
    def process_from_json_2_test(filepath_r,filepath_w):
        fw=open(filepath_w,mode="w",encoding="utf-8")
        with open(filepath_r,mode="r",encoding="utf-8") as fr:
            c=0
            for line in fr:
                line = line.strip()
                if line == "":
                    continue
                text_dict = eval(line)

                # 提取出这句话
                text_list=[]
                for word_pos in text_dict["postag"]:
                    text_list.append([word_pos["word"],word_pos["pos"]])

                wordlist=[]
                pos_list=[]

                for token in text_list:
                    word = "".join(tokenizer.tokenize(token[0])).replace("##","")
                    if word=="":
                        continue
                    wordlist.append(word)
                    pos_list.append(token[1])


                if len(wordlist)==0:
                    print("出现词为0的情况")
                    print(line)
                    continue
                if len(pos_list)==0:
                    print("出现词性为0的情况")
                    print(line)
                    continue

                assert len(wordlist)==len(pos_list)
                new_line_w=[" ".join(wordlist)," ".join(pos_list)]
                # print("--xhm--".join(new_line_w)+"\n")
                fw.write("--xhm--".join(new_line_w)+"\n")

                if c%1000==0:
                    print("语料处理中，，，，",c)
                c+=1




    @staticmethod
    def process_from_json_2_train(filepath_r,filepath_w):
        fw=open(filepath_w,mode="w",encoding="utf-8")
        with open(filepath_r,mode="r",encoding="utf-8") as fr:
            c=0
            for line in fr:
                line = line.strip()
                if line == "":
                    continue
                text_dict = eval(line)
                #提取出 里面的关系词
                predicates = []
                for spo in text_dict["spo_list"]:
                    if spo["predicate"] not in predicates:
                        predicates.append(spo["predicate"])
                # 提取出这句话
                text_list=[]
                for word_pos in text_dict["postag"]:
                    text_list.append([word_pos["word"],word_pos["pos"]])

                wordlist=[]
                pos_list=[]

                for token in text_list:
                    word = "".join(tokenizer.tokenize(token[0])).replace("##","")
                    if word=="":
                        continue
                    wordlist.append(word)
                    pos_list.append(token[1])

                if len(predicates)==0:
                    print("出现关系词为0的情况")
                    print(line)
                    continue
                if len(wordlist)==0:
                    print("出现词为0的情况")
                    print(line)
                    continue
                if len(pos_list)==0:
                    print("出现词性为0的情况")
                    print(line)
                    continue

                assert len(wordlist)==len(pos_list)
                new_line_w=[" ".join(wordlist)," ".join(pos_list)," ".join(predicates)]
                # print("--xhm--".join(new_line_w)+"\n")
                fw.write("--xhm--".join(new_line_w)+"\n")

                if c%1000==0:
                    print("语料处理中，，，，",c)
                c+=1



if __name__=="__main__":
    # DataProcess.process_from_json_2_train("./data/dev_data.json","./data/dev.txt")
    # print("dev,数据处理完毕")
    # DataProcess.process_from_json_2_train("./data/train_data.json","./data/train.txt")
    # print("train","数据处理完毕")
    DataProcess.process_from_json_2_test("./data/test1_data_postag.json","./data/test.txt")
    print("train","数据处理完毕")