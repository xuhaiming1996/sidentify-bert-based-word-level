# coding=utf-8
"""BERT finetuning runner."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import numpy
import os
import modeling
import optimization
import tokenization

import attention
import tensorflow as tf

import copy
flags = tf.flags

FLAGS = flags.FLAGS

## Required parameters
flags.DEFINE_string(
    "data_dir", None,
    " 训练苏需要的所有数据 "
    "for the task.")


flags.DEFINE_integer(
    "num_tpu_cores", 8,
    "Only used if `use_tpu` is True. Total number of TPU cores to use.")


flags.DEFINE_string(
    "bert_config_file", None,
    "The config json file corresponding to the pre-trained BERT model. "
    "This specifies the model architecture.")

flags.DEFINE_string("task_name", None, "The name of the task to train.")

flags.DEFINE_string("vocab_file", None,  "bert本身自带的单词")
flags.DEFINE_string("pos_vocab_file", None,"词性单词")
flags.DEFINE_string("ps_vocab_file", None,  "关系词文件")
flags.DEFINE_string("label_vocab_file", None,  "标签单词")





flags.DEFINE_string(
    "output_dir", None,
    "The output directory where the model checkpoints will be written.")

## Other parameters

flags.DEFINE_string("init_checkpoint", None,
    "Initial checkpoint (usually from a pre-trained BERT model).")

flags.DEFINE_bool(
    "do_lower_case", True,
    "Whether to lower case the input text. Should be True for uncased "
    "models and False for cased models.")
flags.DEFINE_integer( "max_word_length",6,    "The maximum total input sequence length after WordPiece tokenization. "
)

flags.DEFINE_integer("max_sen_length",75,"一个句子最大的长度 ")

flags.DEFINE_bool("do_train", False, "Whether to run training.")

flags.DEFINE_bool("do_eval", False, "Whether to run eval on the dev set.")

flags.DEFINE_bool(
    "do_predict", False,
    "Whether to run the model in inference mode on the test set.")

flags.DEFINE_integer("train_batch_size", 8, "Total batch size for training.")

flags.DEFINE_integer("eval_batch_size", 8, "Total batch size for eval.")

flags.DEFINE_integer("predict_batch_size", 8, "Total batch size for predict.")

flags.DEFINE_float("learning_rate", 4e-5, "The initial learning rate for Adam.")

flags.DEFINE_float("num_train_epochs", 6.0,
                   "Total number of training epochs to perform.")

flags.DEFINE_float(
    "warmup_proportion", 0.1,
    "Proportion of training to perform linear learning rate warmup for. "
    "E.g., 0.1 = 10% of training.")

flags.DEFINE_integer("save_checkpoints_steps", 500,
                     "How often to save the model checkpoint.")

flags.DEFINE_integer("iterations_per_loop", 1000,
                     "How many steps to make in each estimator call.")


tf.flags.DEFINE_string(
    "tpu_name", None,
    "The Cloud TPU to use for training. This should be either the name "
    "used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 "
    "url.")

tf.flags.DEFINE_string(
    "tpu_zone", None,
    "[Optional] GCE zone where the Cloud TPU is located in. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

tf.flags.DEFINE_string(
    "gcp_project", None,
    "[Optional] Project name for the Cloud TPU-enabled project. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

tf.flags.DEFINE_string("master", None, "[Optional] TensorFlow master URL.")



class InputExample(object):
  def __init__(self, guid,text_sen,text_pos,text_label):
    self.guid = guid
    self.text_sen = text_sen
    self.text_pos = text_pos
    self.text_label = text_label


class InputFeatures(object):
  """A single set of features of data."""

  def __init__(self,
               input_words_ids,
               input_pos_ids,
               input_label_ids,
               input_sen_mask,
               input_words_mask,
               input_pos_mask,
               is_real_example=True):
    self.input_words_ids=input_words_ids
    self.input_pos_ids=input_pos_ids
    self.input_label_ids=input_label_ids
    self.input_sen_mask=input_sen_mask
    self.input_words_mask=input_words_mask
    self.input_pos_mask=input_pos_mask
    self.is_real_example = is_real_example


class DataProcessor(object):
  """Base class for data converters for sequence classification data sets."""

  def get_train_examples(self, data_dir):
    """Gets a collection of `InputExample`s for the train set."""
    raise NotImplementedError()

  def get_dev_examples(self, data_dir):
    """Gets a collection of `InputExample`s for the dev set."""
    raise NotImplementedError()

  def get_test_examples(self, data_dir):
    """Gets a collection of `InputExample`s for prediction."""
    raise NotImplementedError()

  def get_labels(self):
    """Gets the list of labels for this data set."""
    raise NotImplementedError()




  @classmethod
  def _read_txt(cls, input_file):
    """读取数据集  """
    with open(input_file,mode="r",encoding="utf-8") as fr:
        lines=[]
        for line in fr:
            # 我这里要求训练集的每一行的格式为 单词--xhm--词性--xhm--labels(也就理解为标签)
            # 我这里要求测试集的每一行的格式为 单词--xhm--词性
            line=line.strip()
            if line=="":
                continue
            tmp=line.strip().split("--xhm--")
            if len(tmp)==2:
                tmp.append(None)
            lines.append(tmp)
        return lines



class SIDCorpusProcessor(DataProcessor):
  def get_train_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_txt(os.path.join(data_dir, "train.txt")), "train")

  def get_dev_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_txt(os.path.join(data_dir, "dev.txt")), "dev")

  def get_test_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_txt(os.path.join(data_dir, "test.txt")), "test")


  def _create_examples(self, lines, set_type):
    """Creates examples for the training and dev sets."""
    examples = []
    for (i, line) in enumerate(lines):
      guid = "%s-%s" % (set_type, i)
      if set_type == "test":
        text_sen = tokenization.convert_to_unicode(line[0])
        text_pos=tokenization.convert_to_unicode(line[1])
        text_label = None
      else:
          text_sen = tokenization.convert_to_unicode(line[0])
          text_pos = tokenization.convert_to_unicode(line[1])
          text_label = tokenization.convert_to_unicode(line[2])
      examples.append(
          InputExample(guid=guid,text_sen=text_sen,text_pos=text_pos,text_label=text_label))
    return examples

# 许海明
def convert_single_example(ex_index, example, max_word_length,max_sen_length,
                           tokenizer,num_labels):
  """Converts a single `InputExample` into a single `InputFeatures`."""
  #为了可以输入，所以要提前构造好labels

  text_sen = example.text_sen.strip().split()
  text_pos=example.text_pos.strip().split()
  if example.text_label is None:
      text_label=["号"]*num_labels
  else:
      text_label = example.text_label.strip().split()

  assert len(text_sen)==len(text_pos)

  text_word=[]
  for word in text_sen:
      text_word.append(tokenizer.tokenize(word))
      #这里是二位列表
      # [
      #   [许，海，明]，
      #   [喜 ，欢] ，
      #   [玩]
      # ]


  text_sen=copy.deepcopy(text_word)

  # Account for  [SEP] with "- 1" #注意这里是句子的长度  原来的
  if len(text_sen) > max_sen_length - 1:
      text_sen = text_sen[0:(max_sen_length - 1)]
      text_pos = text_pos[0:(max_sen_length - 1)]
      text_label=text_label[0:(max_sen_length - 1)]
  text_sen.append(["[SEP]"])
  text_pos.append("[SEP]")


  len_sen=len(text_sen)
  len_pos=len(text_pos)


  while len(text_sen) < max_sen_length:
      text_sen.append(["[PAD]"])
      text_pos.append("[PAD]")


  '''
  处理单词级别
  '''
  #处理每个单词
  # Account for [CLS] ,[SEP] with "- 2" #注意这里是每个单词的长度


  for i,wordlist in enumerate(text_sen):
      if len(wordlist) > max_word_length - 2:
          text_sen[i]=wordlist[0:(max_word_length - 2)]
  # 为每一个单词添加 [CLS] [SEP]

  len_words=[]
  for i,wordlist in enumerate(text_sen):
      wordlist.insert(0,"[CLS]")
      wordlist.append("[SEP]")
      len_words.append(len(wordlist))
      while len(wordlist) < max_word_length:
          wordlist.append("[PAD]")
      text_sen[i]=wordlist

  input_word_ids =[]
  for tokens in text_sen:
      input_word_ids.append(tokenizer.convert_tokens_to_ids(tokens)) #这是一个二维


  input_pos_ids = tokenizer.convert_pos_to_ids(text_pos)    #这是一个list
  #处理label
  label_ids= tokenizer.convert_label_to_ids(text_label)
  input_label_ids=[0]*num_labels
  for id in label_ids:
      input_label_ids[id]=1


  # 制作一个input_sen_mask  这是句子级别的
  input_sen_mask   = [1] * len_sen
  input_pos_mask   = [1] * len_pos

  # Zero-pad up to the sequence length.
  while len(input_sen_mask) < max_sen_length:
      input_sen_mask.append(0)
      input_pos_mask.append(0)


  #为每一个单词制作一个mask
  input_words_mask=[]
  for word_len in len_words:
      word_mask = [1] * word_len
      while len(word_mask) < max_word_length:
          word_mask.append(0)
      input_words_mask.append(word_mask)



  assert len(input_word_ids) == max_sen_length  #句子长度
  assert len(input_pos_ids) == max_sen_length  #句子长度
  assert len(input_label_ids)==num_labels
  assert len(input_word_ids[0])==max_word_length
  assert len(input_pos_mask) == max_sen_length
  assert len(input_words_mask) == max_sen_length



  if ex_index < 5:
    tf.logging.info("*** Example ***")
    tf.logging.info("guid: %s" % (example.guid))
    tf.logging.info("句子单词: %s" % " ".join(
        ["["+" ".join(x)+"]" for x in text_word]))
    tf.logging.info("句子的mask: %s" % " ".join([str(x) for x in input_sen_mask]))



  feature = InputFeatures(
      input_words_ids=input_word_ids,
      input_pos_ids=input_pos_ids,
      input_label_ids=input_label_ids,
      input_sen_mask=input_sen_mask,
      input_words_mask=input_words_mask,
      input_pos_mask=input_pos_mask,
      is_real_example=True)
  return feature


def file_based_convert_examples_to_features(examples, max_word_length,max_sen_length,tokenizer, output_file,num_labels):
  """Convert a set of `InputExample`s to a TFRecord file."""

  writer = tf.python_io.TFRecordWriter(output_file)

  for (ex_index, example) in enumerate(examples):
    if ex_index % 10000 == 0:
      tf.logging.info("Writing example %d of %d" % (ex_index, len(examples)))


    feature = convert_single_example(ex_index, example,max_word_length,max_sen_length,tokenizer,num_labels)

    def create_int_feature(values):
      f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
      return f



    features = collections.OrderedDict()
    features["input_pos_ids"] = create_int_feature(feature.input_pos_ids)
    features["input_label_ids"] = create_int_feature(feature.input_label_ids)
    features["input_sen_mask"] = create_int_feature(feature.input_sen_mask)
    features["input_pos_mask"] = create_int_feature(feature.input_pos_mask)
    features["is_real_example"] = create_int_feature(
        [int(feature.is_real_example)])

    features["input_words_ids"] =     tf.train.Feature(
        bytes_list=tf.train.BytesList(value=[numpy.array(feature.input_words_ids).astype(numpy.int64).tostring()])
    )
    features["input_words_mask"] =  tf.train.Feature(
        bytes_list=tf.train.BytesList(value=[numpy.array(feature.input_words_mask).astype(numpy.int64).tostring()])
    )


    tf_example = tf.train.Example(features=tf.train.Features(feature=features))
    writer.write(tf_example.SerializeToString())
  writer.close()


def file_based_input_fn_builder(input_file, max_word_length,max_sen_length,num_labels, is_training,
                                drop_remainder):
  """Creates an `input_fn` closure to be passed to TPUEstimator."""

  name_to_features = {
      "input_words_ids": tf.FixedLenFeature([], tf.string),
      "input_pos_ids": tf.FixedLenFeature([max_sen_length], tf.int64),
      "input_label_ids": tf.FixedLenFeature([num_labels], tf.int64),
      "input_sen_mask": tf.FixedLenFeature([max_sen_length], tf.int64),
      "input_words_mask": tf.FixedLenFeature([], tf.string),
      "input_pos_mask": tf.FixedLenFeature([max_sen_length], tf.int64),
      "is_real_example": tf.FixedLenFeature([], tf.int64),
  }

  def _decode_record(record, name_to_features):
    """Decodes a record to a TensorFlow example."""
    example = tf.parse_single_example(record, name_to_features)

    # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
    # So cast all int64 to int32.
    for name in list(example.keys()):
      t = example[name]
      if t.dtype == tf.int64:
        t = tf.to_int32(t)
      elif t.dtype == tf.string:
          c_raw = example[name]
          t = tf.decode_raw(c_raw, tf.int64)
          t = tf.reshape(t, [max_sen_length, max_word_length])
          t = tf.to_int32(t)
      example[name] = t

    return example

  def input_fn(params):
    """The actual input function."""
    batch_size = params["batch_size"]

    # For training, we want a lot of parallel reading and shuffling.
    # For eval, we want no shuffling and parallel reading doesn't matter.
    d = tf.data.TFRecordDataset(input_file)
    if is_training:
      d = d.repeat()
      d = d.shuffle(buffer_size=100)

    d = d.apply(
        tf.contrib.data.map_and_batch(
            lambda record: _decode_record(record, name_to_features),
            batch_size=batch_size,
            drop_remainder=drop_remainder))

    return d

  return input_fn





# 这仅仅是创建模型
def create_model(bert_config,
                 is_training,
                 input_words_ids,
                 input_pos_ids,
                 input_label_ids,
                 input_sen_mask,
                 input_words_mask,
                 use_one_hot_embeddings):
  '''
    :param bert_config:
    :param is_training:
    :param input_words_ids: [bacth_size,max_sen_len,max_word_len]
    :param input_pos_ids: [bacth_size,max_sen_len]
    :param input_label_ids: [bacth_size,max_sen_len]
    :param input_sen_mask: [bacth_size,max_sen_len]
    :param input_words_mask: [bacth_size,max_sen_len,max_word_len]
    :param use_one_hot_embeddings:
    :return:
  '''

  #  在这里我要一次性走完所有单词

  input_shape=modeling.get_shape_list(input_words_ids,expected_rank=3)
  bacth_size, max_sen_len, max_word_len=input_shape[0],input_shape[1],input_shape[2]
  input_ids=tf.reshape(input_words_ids,[-1,max_word_len])
  input_mask=tf.reshape(input_words_mask,[-1,max_word_len])

  model = modeling.BertModel(
      config=bert_config,
      is_training=is_training,
      input_ids=input_ids,
      input_mask=input_mask,
      token_type_ids=None,
      use_one_hot_embeddings=use_one_hot_embeddings)


  words_tensor = model.get_pooled_output()


  word_embed=tf.reshape(words_tensor,[bacth_size,max_sen_len,modeling.get_shape_list(words_tensor)[-1]])
  '''
  下面构造
  '''

  # nermodel=nermodeling.NERModel(word_embdeddings,copy.deepcopy(bert_config))

  with tf.variable_scope("poses"):
      # 声明一个词性embedding
      _poses_embeddings = tf.get_variable(
          name="_poses_embeddings",
          dtype=tf.float32,
          shape=[bert_config.nposes, bert_config.dim_pos])
      poses_embeddings = tf.nn.embedding_lookup(_poses_embeddings,
                                                 input_pos_ids, name="poses_embeddings")
      word_embedding = tf.concat([word_embed, poses_embeddings], axis=-1)




  word_embeddings = word_embedding

  sequence_lengths=tf.reduce_sum(input_sen_mask,axis=1)
  with tf.variable_scope("bi-lstm"):
      cell_fw = tf.contrib.rnn.LSTMCell(bert_config.hidden_size_lstm)
      cell_bw = tf.contrib.rnn.LSTMCell(bert_config.hidden_size_lstm)
      (output_fw, output_bw), _ = tf.nn.bidirectional_dynamic_rnn(
          cell_fw, cell_bw, word_embeddings,
          sequence_length=sequence_lengths, dtype=tf.float32)

      output = tf.concat([output_fw, output_bw], axis=-1)
      if is_training:
             output = tf.nn.dropout(output, bert_config.hidden_dropout_prob)

  #再次应用attention

  output=attention.attention(inputs=output, attention_size=bert_config.attention_size_output)


  with tf.variable_scope("proj"):
      logits = tf.layers.dense(inputs=output,
                                units=bert_config.ntags,
                                activation=tf.nn.tanh,
                                use_bias=True)
  loss=tf.nn.sigmoid_cross_entropy_with_logits( labels=tf.to_float(input_label_ids),logits=logits)
  loss=tf.reduce_mean(loss)
  probabilities=tf.nn.sigmoid(logits)
  one = tf.ones_like(probabilities)
  zero = tf.zeros_like(probabilities)
  predict_label = tf.where(probabilities < 0.5, x=zero, y=one)

  return (loss, logits,probabilities,predict_label)


def model_fn_builder(bert_config, num_labels, init_checkpoint, learning_rate,
                     num_train_steps, num_warmup_steps, use_tpu, use_one_hot_embeddings,tokenizer):
  def model_fn(features, labels, mode, params):
    tf.logging.info("*** Features ***")
    for name in sorted(features.keys()):
      tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))
    input_words_ids = features["input_words_ids"]
    input_pos_ids = features["input_pos_ids"]
    input_label_ids = features["input_label_ids"]
    input_sen_mask = features["input_sen_mask"]
    input_words_mask = features["input_words_mask"]
    input_pos_mask = features["input_pos_mask"]
    segment_ids =None
    is_real_example = None
    if "is_real_example" in features:
      is_real_example = tf.cast(features["is_real_example"], dtype=tf.float32)
    else:
      is_real_example = tf.ones(tf.shape(input_label_ids), dtype=tf.float32)

    is_training = (mode == tf.estimator.ModeKeys.TRAIN)

    (loss, logits,probabilities,predict_label) = create_model(
        bert_config,
        is_training,
        input_words_ids,
        input_pos_ids,
        input_label_ids,
        input_sen_mask,
        input_words_mask,
        use_one_hot_embeddings)

    tvars = tf.trainable_variables()
    initialized_variable_names = {}
    scaffold_fn = None
    if init_checkpoint:
      (assignment_map, initialized_variable_names
      ) = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)

      tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

    tf.logging.info("**** Trainable Variables ****")
    for var in tvars:
      init_string = ""
      if var.name in initialized_variable_names:
        init_string = ", *INIT_FROM_CKPT*"
      tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                      init_string)

    output_spec = None
    if mode == tf.estimator.ModeKeys.TRAIN:
      train_op = optimization.create_optimizer(
          loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu)
      output_spec = tf.contrib.tpu.TPUEstimatorSpec(
          mode=mode,
          loss=loss,
          train_op=train_op,
          scaffold_fn=scaffold_fn)
    elif mode == tf.estimator.ModeKeys.EVAL:
      def metric_fn(input_label_ids,predict_label):
          accuracy = tf.metrics.accuracy(labels=input_label_ids, predictions=predict_label)
          f1=tf.contrib.metrics.f1_score(labels=input_label_ids,predictions=predict_label)
          return {
                "eval_accuracy": accuracy,
                "f1": f1,
            }

      eval_metrics = (metric_fn,[input_label_ids,predict_label])
      output_spec = tf.contrib.tpu.TPUEstimatorSpec(
          mode=mode,
          loss=loss,
          eval_metrics=eval_metrics,
          scaffold_fn=scaffold_fn)
    else:
      output_spec = tf.contrib.tpu.TPUEstimatorSpec(
          mode=mode,
          predictions={"predictions": predict_label,"probabilities":probabilities},
          scaffold_fn=scaffold_fn)
    return output_spec

  return model_fn




def main(_):
  tf.logging.set_verbosity(tf.logging.INFO)
  processors = {
      "sid": SIDCorpusProcessor
  }

  tokenization.validate_case_matches_checkpoint(FLAGS.do_lower_case,
                                                FLAGS.init_checkpoint)

  if not FLAGS.do_train and not FLAGS.do_eval and not FLAGS.do_predict:
    raise ValueError(
        "At least one of `do_train`, `do_eval` or `do_predict' must be True.")

  bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)

  if FLAGS.max_word_length > bert_config.max_position_embeddings:
    raise ValueError(
        "Cannot use sequence length %d because the BERT model "
        "was only trained up to sequence length %d" %
        (FLAGS.max_word_length, bert_config.max_position_embeddings))

  tf.gfile.MakeDirs(FLAGS.output_dir)

  task_name = FLAGS.task_name.lower()

  if task_name not in processors:
    raise ValueError("Task not found: %s" % (task_name))

  processor = processors[task_name]()


  print(FLAGS.vocab_file)
  print(FLAGS.pos_vocab_file)
  print(FLAGS.label_vocab_file)


  tokenizer = tokenization.FullTokenizer(
      vocab_file=FLAGS.vocab_file,
      pos_vocab_file= FLAGS.pos_vocab_file,
      label_vocab_file=FLAGS.label_vocab_file,
      do_lower_case=FLAGS.do_lower_case)

  tpu_cluster_resolver = None
  is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
  run_config = tf.contrib.tpu.RunConfig(
      cluster=tpu_cluster_resolver,
      master=FLAGS.master,
      model_dir=FLAGS.output_dir,
      save_checkpoints_steps=FLAGS.save_checkpoints_steps,
      tpu_config=tf.contrib.tpu.TPUConfig(
          iterations_per_loop=FLAGS.iterations_per_loop,
          num_shards=FLAGS.num_tpu_cores,
          per_host_input_for_training=is_per_host))

  train_examples = None
  num_train_steps = None
  num_warmup_steps = None
  if FLAGS.do_train:
    train_examples = processor.get_train_examples(FLAGS.data_dir)
    num_train_steps = int(
        len(train_examples) / FLAGS.train_batch_size * FLAGS.num_train_epochs)
    num_warmup_steps = int(num_train_steps * FLAGS.warmup_proportion)

  model_fn = model_fn_builder(
      bert_config=bert_config,
      num_labels=bert_config.ntags,
      init_checkpoint=FLAGS.init_checkpoint,
      learning_rate=FLAGS.learning_rate,
      num_train_steps=num_train_steps,
      num_warmup_steps=num_warmup_steps,
      use_tpu=False,
      use_one_hot_embeddings=False,
      tokenizer=tokenizer)

  # If TPU is not available, this will fall back to normal Estimator on CPU
  # or GPU.
  estimator = tf.contrib.tpu.TPUEstimator(
      use_tpu=False,
      model_fn=model_fn,
      config=run_config,
      train_batch_size=FLAGS.train_batch_size,
      eval_batch_size=FLAGS.eval_batch_size,
      predict_batch_size=FLAGS.predict_batch_size)

  if FLAGS.do_train:
    train_file = os.path.join(FLAGS.output_dir, "train.tf_record")
    file_based_convert_examples_to_features(
        train_examples,FLAGS.max_word_length, FLAGS.max_sen_length, tokenizer, train_file,bert_config.ntags)
    tf.logging.info("***** Running training *****")
    tf.logging.info("  Num examples = %d", len(train_examples))
    tf.logging.info("  Batch size = %d", FLAGS.train_batch_size)
    tf.logging.info("  Num steps = %d", num_train_steps)

    train_input_fn = file_based_input_fn_builder(
        input_file=train_file,
        max_word_length=FLAGS.max_word_length,
        max_sen_length=FLAGS.max_sen_length,
        num_labels=bert_config.ntags,
        is_training=True,
        drop_remainder=True)
    estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)

  if FLAGS.do_eval:
    eval_examples = processor.get_dev_examples(FLAGS.data_dir)
    num_actual_eval_examples = len(eval_examples)


    eval_file = os.path.join(FLAGS.output_dir, "eval.tf_record")
    file_based_convert_examples_to_features(
        eval_examples,  FLAGS.max_word_length,FLAGS.max_sen_length, tokenizer, eval_file,bert_config.ntags)

    tf.logging.info("***** Running evaluation *****")
    tf.logging.info("  Num examples = %d (%d actual, %d padding)",
                    len(eval_examples), num_actual_eval_examples,
                    len(eval_examples) - num_actual_eval_examples)
    tf.logging.info("  Batch size = %d", FLAGS.eval_batch_size)

    # This tells the estimator to run through the entire set.
    eval_steps = None
    # However, if running eval on the TPU, you will need to specify the
    # number of steps.


    eval_drop_remainder =  False
    eval_input_fn = file_based_input_fn_builder(
        input_file=eval_file,
        max_word_length=FLAGS.max_word_length,
        max_sen_length=FLAGS.max_sen_length,
        num_labels=bert_config.ntags,
        is_training=False,
        drop_remainder=eval_drop_remainder)

    result = estimator.evaluate(input_fn=eval_input_fn, steps=eval_steps)

    output_eval_file = os.path.join(FLAGS.output_dir, "eval_results.txt")
    with tf.gfile.GFile(output_eval_file, "w") as writer:
      tf.logging.info("***** Eval results *****")
      for key in sorted(result.keys()):
        tf.logging.info("  %s = %s", key, str(result[key]))
        writer.write("%s = %s\n" % (key, str(result[key])))

  if FLAGS.do_predict:
    predict_examples = processor.get_test_examples(FLAGS.data_dir)
    num_actual_predict_examples = len(predict_examples)

    predict_file = os.path.join(FLAGS.output_dir, "predict.tf_record")
    file_based_convert_examples_to_features(predict_examples,  FLAGS.max_word_length,FLAGS.max_sen_length, tokenizer,
                                            predict_file,bert_config.ntags)

    tf.logging.info("***** Running prediction*****")
    tf.logging.info("  Num examples = %d (%d actual, %d padding)",
                    len(predict_examples), num_actual_predict_examples,
                    len(predict_examples) - num_actual_predict_examples)
    tf.logging.info("  Batch size = %d", FLAGS.predict_batch_size)

    predict_drop_remainder = False
    predict_input_fn = file_based_input_fn_builder(
        input_file=predict_file,
        max_word_length=FLAGS.max_word_length,
        max_sen_length=FLAGS.max_sen_length,
        num_labels=bert_config.ntags,
        is_training=False,
        drop_remainder=predict_drop_remainder)

    result = estimator.predict(input_fn=predict_input_fn)

    output_predict_file = os.path.join(FLAGS.output_dir, "test_results.tsv")


    with tf.gfile.GFile(output_predict_file, "w") as writer:
      num_written_lines = 0
      tf.logging.info("***** Predict results *****")
      for (i, prediction) in enumerate(result):
        pred = prediction["predictions"]
        probabilities=prediction["probabilities"]
        print(probabilities)
        pred_ids=[]

        if i >= num_actual_predict_examples:
           break
        for label_id,v in enumerate(pred):
            if v==1:
                pred_ids.append(label_id)

        output_line = "--None--" + "\n"
        if len(pred_ids)>=1:
            output_line = " ".join(tokenizer.convert_label_ids_to_tokens(pred_ids))+"\n"
        writer.write(output_line)
        num_written_lines += 1
    assert num_written_lines == num_actual_predict_examples



if __name__ == "__main__":
  flags.mark_flag_as_required("data_dir")
  flags.mark_flag_as_required("task_name")
  flags.mark_flag_as_required("vocab_file")
  flags.mark_flag_as_required("bert_config_file")
  flags.mark_flag_as_required("output_dir")
  tf.app.run()
