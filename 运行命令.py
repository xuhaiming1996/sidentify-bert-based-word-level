python run_sid_xhm.py \
        --task_name=SID \
        --do_train=True \
        --do_eval=true   \
        --max_word_length=6 \
        --max_sen_length=75 \

        --train_file=./data/ \
        --vocab_file=./bert_model/chinese_L-12_H-768_A-12/vocab.txt  \
        --bert_config_file=./bert_model/chinese_L-12_H-768_A-12/bert_config.json  \
        --init_checkpoint=./bert_model/chinese_L-12_H-768_A-12/bert_model.ckpt \
        --output_dir=./resluts/ \
        --pos_vocab_file=./data/postag.txt   \
        --label_vocab_file=./data/labels.txt \
        --data_dir=./data/

 python run_ner_xhm.py         --task_name=ner          --do_eval=true
      --max_word_length=6         --max_sen_length=75         --train_file=./data/         --vocab_file=./bert_model/chinese_L-12_H-768_A-12/vocab
.txt          --bert_config_file=./bert_model/chinese_L-12_H-768_A-12/bert_config.json          --init_checkpoint=./bert_model/chinese_L-12_H-768_
A-12/bert_model.ckpt         --output_dir=./resluts/         --pos_vocab_file=./data/postag.txt           --ps_vocab_file=./data/ps.txt          -
-label_vocab_file=./data/tags.txt         --data_dir=./data/



