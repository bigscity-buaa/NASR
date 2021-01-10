import tensorflow as tf
import numpy as np
import random
import pickle
import copy
from collections import deque
import os
import math

os.environ['CUDA_VISIBLE_DEVICES']='0'
# Hyper Parameters for DAN
PRE_TRAIN = False
TEST = True
RESTORE = True
GAMMA = 1.0 # discount factor for target Q
INITIAL_EPSILON = 0.5 # starting value of epsilon
FINAL_EPSILON = 0.01 # final value of epsilon
batch_size = None # size of minibatch
input_steps = None
block_num = 16000#15500    #ofo为47702  出租车为15042
user_num = 15000
user_size = 128
loc_size = 384
time_num = 1000  # dive 24 hour into 96
day_num = 10  # weekday and weekend  7 day
lstm_size = 384
num_layers = 1
TRAIN_BATCH_SIZE = 100 #训练输入的batch 大小
INFERENCE_BATCH_SIZE = 1 #推断的时候输入的batch 大小
PRE_EPISODE = 600
NEG_SAMPLES = 9
NEXT_ACTION_NUM = 3
road_num = 15500
time_size = 56
day_size = 8
his_num = 5
his_length = 50
layers = 6
heads = 6
anchor_num = 64
#class CustomRNN(tf.nn.rnn_cell.BasicLSTMCell):
#    def __init__(self, *args, **kwargs):
#        kwargs['state_is_tuple'] = False # force the use of a concatenated state.
#        returns = super(CustomRNN, self).__init__(*args, **kwargs) # create an lstm cell
#        self._output_size = self._state_size # change the output size to the state size
#        return returns
#    def __call__(self, inputs, state):
#        output, next_state = super(CustomRNN, self).__call__(inputs, state)
#        return next_state, next_state # return two copies of the state, instead of the output and the state


class DAN():
  # DQN Agent
  def __init__(self):
    # init experience replay
    self.train_batches = []
    self.test_batches = []
    # init some parameters
    self.token2cor = {}
    self.sigma = 0.001   #高斯核的系数

    self.gradients= None

    self.create_attn_st_network()
#    self.create_st_network()
#    self.create_pgnn_heuristics_network()
    self.create_heuristics_network()
    self.all_saver = tf.train.Saver(max_to_keep=10)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    self.session = tf.InteractiveSession(config = config)

    self.session.run(tf.global_variables_initializer())

#    self.all_saver = tf.train.import_meta_graph("/data/wuning/AstarRNN/pretrain_test_policity_neural_network_epoch0.ckpt.meta")
# "/data/wuning/AstarRNN/pretrain_test_policity_neural_network_epoch0.ckpt")
    # Init session

    all_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    variables_to_restore = [v for v in all_variables if v.name.split('/')[0]=='st_network']
#    print("variables:", variables_to_restore)
    self.st_saver = tf.train.Saver(variables_to_restore, max_to_keep=10)
#    self.all_saver = tf.train.Saver(max_to_keep=10)

  def build_lstm(self, batch_size):
    lstm = tf.nn.rnn_cell.GRUCell(lstm_size)
#    lstm = CustomRNN(lstm_size)
  # 添加dropout

    drop = tf.nn.rnn_cell.DropoutWrapper(lstm, output_keep_prob=1.0)

  # 堆叠
    cell = tf.nn.rnn_cell.MultiRNNCell([drop for _ in range(num_layers)], state_is_tuple=False)
    initial_state = cell.zero_state(batch_size, tf.float32)
    print("GRU state:", initial_state)
    return cell, initial_state
  
  def create_attn_st_network(self):
    with tf.variable_scope("st_network"):
 #     self.attn_mask = tf.placeholder(tf.float32, shape=(input_steps, input_steps))
      self.st_known_ = tf.placeholder(tf.int64, shape=(batch_size, input_steps), name='st_known')
      self.st_output_ = tf.placeholder(tf.int64, shape=(batch_size, input_steps), name='st_output')
      self.st_time = tf.placeholder(tf.int64, shape=(batch_size, input_steps), name='st_time')
      self.st_time_emb = tf.contrib.layers.embed_sequence(self.st_time, time_num, time_size, scope = "time_embedding")
      self.st_user = tf.placeholder(tf.int64, shape=(batch_size, 1), name='st_user')
      self.st_user_tile = tf.tile(self.st_user, [1, tf.shape(self.st_known_)[1]])
      self.st_user_emb = tf.contrib.layers.embed_sequence(self.st_user_tile, user_num, user_size, scope = "user_embedding")

      self.st_known_embedding = tf.contrib.layers.embed_sequence(self.st_known_, block_num, loc_size, scope = "location_embedding")
      self.st_destination_ = tf.placeholder(tf.int64, shape=(batch_size, 1), name='p_destination')
      self.st_destination_embedding = tf.contrib.layers.embed_sequence(self.st_destination_, block_num, loc_size, scope = "location_embedding", reuse = True)
      cell, initial_state = self.build_lstm(tf.shape(self.st_known_)[0])
      fusion = tf.concat([self.st_known_embedding, self.st_time_emb, self.st_user_emb], 2)
      self.outputs, self.final_state = tf.nn.dynamic_rnn(cell, fusion, initial_state = initial_state, dtype=tf.float32)  #batch * len * dim
      self.rnn_outputs = self.outputs
#      attn = tf.einsum("ijk,ikj->ijj", self.outputs, tf.transpose(self.outputs, [0, 2, 1]))
#      attn = attn * self.attn_mask
#      self.outputs = tf.transpose(tf.matmul(tf.transpose(self.outputs, [0, 2, 1]), attn), [0, 2, 1])

#
################ history module
      self.his_user = tf.placeholder(tf.int64, shape=(batch_size, his_num, his_length), name='his_user')
      self.his_tra = tf.placeholder(tf.int64, shape=(batch_size, his_num, his_length), name='his_tra')
      self.his_time = tf.placeholder(tf.int64, shape=(batch_size, his_num, his_length), name='his_time')
      self.his_padding_mask = tf.placeholder(tf.float32, shape=(batch_size, his_num, his_length), name='his_mask')


      self.his_user_emb = tf.contrib.layers.embed_sequence(self.his_user, user_num, user_size, scope = "user_embedding", reuse = True)
      self.his_tra_emb = tf.contrib.layers.embed_sequence(self.his_tra, block_num, lstm_size, scope = "location_embedding", reuse = True)
      self.his_time_emb = tf.contrib.layers.embed_sequence(self.his_time, time_num, time_size, scope = "time_embedding", reuse = True)
      self.his_tra_emb = tf.concat([self.his_tra_emb, self.his_time_emb, self.his_user_emb], 3)  # batch_size his_num his_length lstm_size
      self.his_tra_emb = tf.reshape(self.his_tra_emb, [-1, his_length, lstm_size + time_size + user_size])

      self.initial_state_1 = cell.zero_state(tf.shape(self.his_tra_emb)[0], tf.float32)
      self.his_outputs, self.his_final_state = tf.nn.dynamic_rnn(cell, self.his_tra_emb, initial_state = self.initial_state_1, dtype=tf.float32)
      self.his_outputs = tf.reshape(self.his_outputs, [tf.shape(self.his_user)[0], tf.shape(self.his_user)[1], tf.shape(self.his_user)[2], lstm_size])
      
# batch_size len lstm_size * batch_size lstm_size his_num his_len
      self.his_attn_1 = tf.einsum("iml,iljk->imjk", self.outputs, tf.transpose(self.his_outputs, [0, 3, 1, 2]))  # batch len his_num his_len 
      self.his_attn_1 = self.his_attn_1 * tf.tile(tf.expand_dims(self.his_padding_mask, 1), [1, tf.shape(self.st_known_)[1], 1, 1])
# batch_size len his_num his_length * batch_size len his_num his_length lstm_size 
      self.his_outputs = tf.einsum("ijkl,ijklm->ijkm", self.his_attn_1, tf.tile(tf.expand_dims(self.his_outputs, 1),[1, tf.shape(self.st_known_)[1], 1, 1, 1])) #batch_size len his_num lstm_size
# batch_size len lstm_size * batch_size len lstm_size his_num   
      self.his_attn_2 = tf.einsum("ijk,ijkl->ijl", self.outputs, tf.transpose(self.his_outputs, [0, 1, 3, 2])) # batch_size len his_num
      self.his_outputs = tf.einsum("ijk,ijkl->ijl", self.his_attn_2, self.his_outputs) #batch_size len lstm_size
      self.outputs = tf.concat([self.outputs, self.his_outputs], 2)
#################
      
      print(self.outputs.shape)
      with tf.variable_scope('st_output'):
        w_p1 = tf.Variable(tf.truncated_normal([3 * lstm_size, block_num], stddev=0.1))
        b_p1 = tf.Variable(tf.zeros(block_num))

      print(self.st_destination_embedding.shape, "--------------------")
      self.st_layer_1 = tf.nn.relu(tf.matmul(tf.reshape(tf.concat([tf.tile(self.st_destination_embedding, [1, tf.shape(self.st_known_)[1], 1]),  self.outputs], 2), [-1, 3 * lstm_size]), w_p1) + b_p1)
      self.st_all = tf.reshape(self.st_layer_1, [tf.shape(self.st_known_)[0], tf.shape(self.st_known_)[1], block_num])
#      self.st_all = tf.transpose(tf.reshape(tf.matmul(self.st_layer_1, w_p2) + b_p2, [tf.shape(self.st_known_)[1], tf.shape(self.st_known_)[0], block_num]), [1, 0, 2])
      self.st_all_prob = tf.nn.softmax(self.st_all)
      action_one_hot = tf.one_hot(self.st_output_, block_num)
      self.st_all_cost = tf.nn.softmax_cross_entropy_with_logits(logits=self.st_all , labels=action_one_hot)
      self.st_all_cost = tf.reduce_mean(self.st_all_cost)
      self.st_all_optimizer = tf.train.AdamOptimizer(0.0001).minimize(self.st_all_cost) 
    

  def create_step_by_step_st_network(self):
    with tf.variable_scope("st_network"):
      self.st_known_ = tf.placeholder(tf.int64, shape=(batch_size, input_steps), name='st_known')
      self.neg_known_ = tf.placeholder(tf.int64, shape=(batch_size, input_steps), name='st_known')
      st_known_embedding = tf.contrib.layers.embed_sequence(self.st_known_, block_num, lstm_size, scope = "location_embedding")
      neg_known_embedding = tf.contrib.layers.embed_sequence(self.neg_known_, block_num, lstm_size, scope = "location_embedding", reuse = True)
      print("--------------", st_known_embedding)
      self.st_destination_ = tf.placeholder(tf.int64, shape=(batch_size), name='p_destination')
      self.st_destination_embedding = tf.contrib.layers.embed_sequence(self.st_destination_, block_num, lstm_size, scope = "location_embedding", reuse = True)
      cell, initial_state = self.build_lstm(tf.shape(self.st_known_)[0])
      self.states, self.final_state = tf.nn.dynamic_rnn(cell, tf.transpose(st_known_embedding, [1, 0, 2]), initial_state = initial_state, dtype=tf.float32, time_major=True)
      print("states:", self.states)
      time = tf.shape(self.st_known_)[1] - 2
      #开始构建负样本的图
      initial_state = self.states[0]
      print(initial_state)
      def compute(i, cur_state, out):
        output, cur_state = cell(neg_known_embedding[:, i + 1], self.states[i])
        return i + 1, cur_state, out.write(i, output)
      _, cur_state, out = tf.while_loop(
        lambda a,b,c: a < time,
        compute,
        (0, initial_state, tf.TensorArray(tf.float32, time))  
      )  
      with tf.variable_scope('st_output'):
        w_p1 = tf.Variable(tf.truncated_normal([lstm_size, 2*lstm_size], stddev=0.1))
        b_p1 = tf.Variable(tf.zeros(2*lstm_size))

        w_p2 = tf.Variable(tf.truncated_normal([2*lstm_size, 1], stddev=0.1))
        b_p2 = tf.Variable(tf.zeros(1))

      self.cutted_states = self.states[1:-1]  #截头去尾
      self.nest_outputs = tf.reshape(tf.add(self.st_destination_embedding, self.cutted_states), [-1, lstm_size])
      
      self.st_layer_1 =tf.nn.relu(tf.matmul(self.nest_outputs, w_p1) + b_p1)
      self.st = tf.matmul(self.st_layer_1, w_p2) + b_p2

      self.infer_outputs = tf.add(self.st_destination_embedding, self.states[-1, :, :])
      self.infer_st_layer_1 =tf.nn.relu(tf.matmul(self.infer_outputs, w_p1) + b_p1)
      self.infer_st = tf.nn.sigmoid(tf.matmul(self.infer_st_layer_1, w_p2) + b_p2)


      self.nest_neg_outputs = tf.reshape(tf.add(self.st_destination_embedding, out.stack()), [-1, lstm_size])
      self.neg_layer_1 = tf.nn.relu(tf.matmul(self.nest_neg_outputs, w_p1) + b_p1)
      self.neg = tf.matmul(self.neg_layer_1, w_p2) + b_p2

      self.st_cost = tf.reduce_mean(- tf.log(tf.nn.sigmoid(self.st)) - tf.log(tf.nn.sigmoid( - self.neg)))

#############margin loss
#      margin = 1.0 - self.st + self.neg#tf.slice(self.st, [0, 0], [TRAIN_BATCH_SIZE, 1]) + tf.slice(self.st, [TRAIN_BATCH_SIZE, 0], [TRAIN_BATCH_SIZE, 1])
#      condition = tf.less(margin, 0.)
#      self.st_cost = tf.reduce_mean(tf.where(condition, tf.zeros_like(margin), margin))
##############

      self.st_optimizer = tf.train.AdamOptimizer(0.0001).minimize(self.st_cost)

  def history_attention(self, query, des):
#  query   batch_size * lstm_size
#  des     batch_size * lstm_size 
    with tf.variable_scope("st_network"):  
#      self.his_tra = tf.placeholder(tf.int64, shape=(batch_size, his_num, his_length))
#      self.his_time = tf.placeholder(tf.int64, shape=(batch_size, his_num, his_length))
#      self.his_day = tf.placeholder(tf.int64, shape=(batch_size, his_num, his_length))
#      self.his_tra_emb = tf.contrib.layers.embed_sequence(self.his_tra, block_num, lstm_size, scope = "location_embedding", reuse = True)
#      self.his_time_emb = tf.contrib.layers.embed_sequence(self.his_time, time_num, time_size, scope = "time_embedding", reuse = True)
#      self.his_day_emb = tf.contrib.layers.embed_sequence(self.his_day, day_num, day_size, scope = "day_embedding", reuse = True)
#      self.his_tra_emb = tf.concat([his_tra_emb, his_time_emb, his_day_emb], 2)  # batch_size his_num his_length lstm_size
#      self.his_tra_emb = tf.contrib.layers.fully_connected(self.his_tra_emb, lstm_size)    #default relu activation
#      print(self.his_tra_emb)
      cell, initial_state = self.build_lstm(tf.shape(self.his_tra)[0])  
      def compute(i, cur_state, out, fin):
        his_state, final_state =  tf.nn.dynamic_rnn(cell, tf.transpose(self.his_tra_emb[:, i, :, :], [1,0,2]), initial_state = initial_state, dtype=tf.float32, time_major=True)
        return i + 1, cur_state, out.write(i, his_state), fin.write(i, final_state)
      _, cur_state, out, fin = tf.while_loop(
          lambda a,b,c,d: a < his_num,
          compute,
          (0, initial_state, tf.TensorArray(tf.float32, his_num), tf.TensorArray(tf.float32, his_num))  
        )
      out = out.stack()
      #out  tra_num * tra_len * batch_size * lstm_size  -> batch_size * tra_num * tra_len * lstm_size  query batch_size * lstm_size
      weight_1 = tf.einsum('ijkl,il->ijk',tf.reshape(out, [tf.shape(self.his_tra_emb)[0], his_num, his_length, lstm_size]), query)  
      #his_tra_emb  batch_size * his_num * his_length * lstm_size  des  batch_size * 1 * lstm_size
      print(self.his_tra_emb, des, tf.squeeze(des, [1]))
      weight_2 = tf.einsum('ijkl,il->ijk',self.his_tra_emb, tf.squeeze(des, [1]))
#      tf.reshape(tf.matmul(tf.reshape(tf.transpose(self.his_tra_emb, [1,2,0,3]), [tf.shape(self.his_tra_emb)[0], -1, lstm_size]), des), [tf.shape(self.his_tra_emb)[1], tf.shape(self.his_tra_emb)[2], tf.shape(self.his_tra_emb)[0]]) 
      local_weight = tf.nn.softmax((weight_1 + weight_2) * tf.cast(self.his_padding_mask, dtype=tf.float32))  #batch_size his_num his_length
      tra_emb = tf.einsum('ijk,ijkl->ijl', local_weight, tf.transpose(out, [2, 0, 1, 3])) #batch_size * his_num  * his_length * lstm_size -> batch_size * his_num *lstm_size
      global_weight = tf.nn.softmax(tf.einsum('ijk,ik->ij', tra_emb, query))  #batch_size his_hum
    return tf.einsum('ij,ijk->ik', global_weight, tra_emb) #batch_size lstm_size
      
  def create_st_network(self):  
    with tf.variable_scope("st_network"):
      self.st_known_ = tf.placeholder(tf.int64, shape=(batch_size, input_steps), name='st_known')
      self.neg_known_ = tf.placeholder(tf.int64, shape=(batch_size, input_steps), name='neg_known')
      self.st_output_ = tf.placeholder(tf.int64, shape=(batch_size, input_steps), name='st_output')
      self.trans_mat = tf.placeholder(tf.int64, shape=(batch_size, input_steps, 70), name='tras_mat')
      self.st_time = tf.placeholder(tf.int64, shape=(batch_size, input_steps), name='st_time')

      self.st_time_emb = tf.contrib.layers.embed_sequence(self.st_time, time_num, time_size, scope = "time_embedding")

      self.st_known_embedding = tf.contrib.layers.embed_sequence(self.st_known_, block_num, lstm_size, scope = "location_embedding")
      self.neg_known_embedding = tf.contrib.layers.embed_sequence(self.neg_known_, block_num, lstm_size, scope = "location_embedding", reuse = True)
      print("--------------", self.st_known_embedding)
      self.st_destination_ = tf.placeholder(tf.int64, shape=(batch_size, 1), name='p_destination')
      self.st_destination_embedding = tf.contrib.layers.embed_sequence(self.st_destination_, block_num, lstm_size, scope = "location_embedding", reuse = True)
      print("--------------", self.st_destination_embedding)
 #     self.padding_mask = tf.placeholder(tf.int64, shape=(batch_size))

################ history module
      self.his_tra = tf.placeholder(tf.int64, shape=(batch_size, his_num, his_length), name='his_tra')
      self.his_time = tf.placeholder(tf.int64, shape=(batch_size, his_num, his_length), name='his_time')
      self.his_day = tf.placeholder(tf.int64, shape=(batch_size, his_num, his_length), name='his_day')
      self.his_padding_mask = tf.placeholder(tf.int64, shape=(batch_size, his_num, his_length), name='his_mask')
      self.his_tra_emb = tf.contrib.layers.embed_sequence(self.his_tra, block_num, lstm_size, scope = "location_embedding", reuse = True)
      self.his_time_emb = tf.contrib.layers.embed_sequence(self.his_time, time_num, time_size, scope = "time_embedding", reuse = True)
      self.his_day_emb = tf.contrib.layers.embed_sequence(self.his_day, day_num, day_size, scope = "day_embedding", reuse = True)
      self.his_tra_emb = tf.concat([self.his_tra_emb, self.his_time_emb, self.his_day_emb], 3)  # batch_size his_num his_length lstm_size
      self.his_tra_emb = tf.contrib.layers.fully_connected(self.his_tra_emb, lstm_size)    #default relu activation
      print("his_tra_emb:", self.his_tra_emb)
#################

      cell, initial_state = self.build_lstm(tf.shape(self.st_known_)[0])
      self.outputs, self.final_state = tf.nn.dynamic_rnn(cell, tf.transpose(self.st_known_embedding, [1, 0, 2]), initial_state = initial_state, dtype=tf.float32, time_major=True)
#      self.neg_outputs, self.neg_final_state = tf.nn.dynamic_rnn(cell, tf.transpose(neg_known_embedding, [1, 0, 2]), initial_state = initial_state, dtype=tf.float32, time_major=True)
################ add historical attention
      tra_len = tf.shape(self.st_known_)[1]
      def compute(i, cur_state, ctx):
        his_state  = self.history_attention(self.outputs[i], self.st_destination_embedding)
        return i + 1, cur_state, ctx.write(i, his_state)
      _, cur_state, ctx = tf.while_loop(
          lambda a,b,c: a < tra_len,
          compute,
          (0, initial_state, tf.TensorArray(tf.float32, tra_len))  
        )
      ctx = ctx.stack()
#      self.outputs = self.outputs + ctx
################
      with tf.variable_scope('st_output'):
        w_p1 = tf.Variable(tf.truncated_normal([lstm_size, 2*lstm_size], stddev=0.1))
        b_p1 = tf.Variable(tf.zeros(2*lstm_size))

        w_p2 = tf.Variable(tf.truncated_normal([2*lstm_size, block_num], stddev=0.1))
        b_p2 = tf.Variable(tf.zeros(1))

      print(self.st_destination_embedding, self.final_state[0][1], self.outputs)
      self.st_layer_1 =tf.nn.relu(tf.matmul(tf.reshape(tf.add(tf.squeeze(self.st_destination_embedding, [1]), self.outputs[:, :, :]), [-1, lstm_size]), w_p1) + b_p1)
      self.st_all =tf.transpose(tf.reshape(tf.matmul(self.st_layer_1, w_p2) + b_p2, [tf.shape(self.st_known_)[1], tf.shape(self.st_known_)[0], block_num]), [1, 0, 2])
      self.st_all_prob = tf.nn.softmax(self.st_all)
      self.mask_code = tf.cast(tf.reduce_sum(tf.one_hot(self.trans_mat, block_num, dtype=tf.uint8), axis=2), dtype=tf.bool)
#      self.mask_code = tf.reduce_sum(tf.one_hot(self.trans_mat, block_num, dtype=tf.uint8), axis=2)
      dummy_scores = tf.ones_like(self.st_all) * -99999.0
      self.st = tf.where(self.mask_code, self.st_all, dummy_scores) 
#      self.st = tf.boolean_mask(self.st, self.mask_code) 
      print("st:", self.st)
#      self.neg_layer_1 =tf.nn.relu(tf.matmul(tf.add(self.st_destination_embedding, self.neg_outputs[-1, :, :]), w_p1) + b_p1)
#      self.neg = tf.matmul(self.neg_layer_1, w_p2) + b_p2

#      print(self.st)
#      margin = 1.0 - self.st + self.neg#tf.slice(self.st, [0, 0], [TRAIN_BATCH_SIZE, 1]) + tf.slice(self.st, [TRAIN_BATCH_SIZE, 0], [TRAIN_BATCH_SIZE, 1])
#      condition = tf.less(margin, 0.)
#      self.st_cost = tf.reduce_mean(tf.where(condition, tf.zeros_like(margin), margin))
#      self.st_optimizer = tf.train.AdamOptimizer(0.0001).minimize(self.st_cost) 

##----------  cross entropy
      self.action = tf.argmax(self.st, axis=1)
      self.st_prob = tf.nn.softmax(self.st)
      action_one_hot = tf.one_hot(self.st_output_, block_num)
      self.st_cost = tf.nn.softmax_cross_entropy_with_logits(logits=self.st_all , labels=action_one_hot)
      self.st_cost = tf.reduce_mean(self.st_cost)
      self.st_optimizer = tf.train.AdamOptimizer(0.0001).minimize(self.st_cost) 
      self.st_all_cost = tf.nn.softmax_cross_entropy_with_logits(logits=self.st_all , labels=action_one_hot)
      self.st_all_cost = tf.reduce_mean(self.st_all_cost)
      self.st_all_optimizer = tf.train.AdamOptimizer(0.0001).minimize(self.st_all_cost) 
  def attn_head(self, inp, bias_mat, context, geo_dist):
#  inp  batch_size * node_num * feature_dim1
#  out  batch_size * node_num * feature_dim2
#  context batch_size * feature_dim3
#  geo distance  batch_size * node_num * node_num
    with tf.variable_scope('attention_weight'):
      w_a1 = tf.Variable(tf.truncated_normal([lstm_size, 64], stddev=0.1))
      b_a1 = tf.Variable(tf.zeros(64))
      w_a2 = tf.Variable(tf.truncated_normal([lstm_size, 64], stddev=0.1))
      b_a2 = tf.Variable(tf.zeros(64))
      w_a3 = tf.Variable(tf.truncated_normal([64], stddev=0.1))
      b_a3 = tf.Variable(tf.zeros(1))
    inp_1 = tf.expand_dims(tf.einsum('ijk,kl->ijl', inp, w_a1) + b_a1, 2)
    inp_2 = tf.expand_dims(tf.einsum('ijk,kl->ijl', inp, w_a1) + b_a1, 1)
#    attn = tf.zeros([inp.shape[0], inp.shape[1], inp.shape[1], inp.shape[2]])
    con = tf.matmul(context, w_a2) + b_a2
    con = tf.expand_dims(con, 1)
    con = tf.expand_dims(con, 1)
    attn = inp_1 + inp_2 + con
    logits = tf.einsum('ijkl,l->ijk', attn, w_a3) + b_a3
    coefs = tf.nn.softmax(tf.nn.relu(logits) + bias_mat)
    vals = tf.einsum('ijk,ikl->ijl', coefs, tf.squeeze(inp_1, 2))
    return tf.nn.elu(vals)

  def geo_pgnn_layer(self, feature, anchor_sets, sub_sets, context, pf_w, pf_b, out_w, out_b):
    self.anchor_dist_mat = tf.placeholder(tf.float32, shape=(block_num, anchor_num), name='p_destination')
    anchor_feature = tf.contrib.layers.embed_sequence(anchor_sets, block_num, loc_size, scope = "st_network/location_embedding", reuse = True)
    self_feature = tf.contrib.layers.embed_sequence(sub_sets, block_num, loc_size, scope = "st_network/location_embedding", reuse = True)
    anchor_feature = tf.stop_gradient(anchor_feature)
    self_feature = tf.stop_gradient(self_feature)

    #anchor_feature: batch_size * anchor_num * dim    self_feature: batch_size * anchor_num + 2 * dim
    anchor_feature = tf.expand_dims(anchor_feature, axis=1)
    anchor_feature = tf.tile(anchor_feature, [1, 2, 1, 1])

    self_feature = tf.expand_dims(self_feature, axis=2)
    self_feature = tf.tile(self_feature, [1, 1, anchor_num, 1])

    context = tf.tile(tf.expand_dims(tf.expand_dims(context, axis=1), axis=1), [1, 2, anchor_num, 1])
    dist_compute = tf.gather(self.anchor_dist_mat, sub_sets)

    messages = anchor_feature * ( 1 / (1+ tf.expand_dims(dist_compute, axis = 3)))
    messages = tf.concat([messages, self_feature], axis=3)
    messages = tf.nn.relu(messages)
    out_position = tf.einsum('ijkl,l->ijk', messages, out_w) + out_b
    out_structure = tf.reduce_mean(messages, axis = 1)
    return out_position, out_structure



  def pf_pgnn_layer(self, feature, anchor_sets, sub_sets, context, pf_w, pf_b, out_w, out_b):
#    print(feature.shape)
#    anchor_features = tf.gather(feature, anchor_sets[0], axis = 0)
#    print("af:", anchor_features.shape, feature.shape, anchor_sets.shape)
#    anchor_features = tf.tile(tf.expand_dims(anchor_features, axis=0), [16500, 1, 1]) 
#    print("af:", anchor_features.shape)
#    anchor_features = tf.reshape(anchor_features, (100, 100, 384))#(anchor_sets.shape[0], anchor_sets.shape[1], feature.shape[2]))

#    self_feature = tf.expand_dims(feature, axis=1)
#    self_feature = tf.tile(self_feature, [1, 100, 1])
   
    anchor_feature = tf.contrib.layers.embed_sequence(anchor_sets, block_num, loc_size, scope = "st_network/location_embedding", reuse = True)
    self_feature = tf.contrib.layers.embed_sequence(sub_sets, block_num, loc_size, scope = "st_network/location_embedding", reuse = True)
    anchor_feature = tf.stop_gradient(anchor_feature)
    self_feature = tf.stop_gradient(self_feature)

    #anchor_feature: batch_size * anchor_num * dim    self_feature: batch_size * anchor_num + 2 * dim
    anchor_feature = tf.expand_dims(anchor_feature, axis=1)
    anchor_feature = tf.tile(anchor_feature, [1, 2, 1, 1])

    self_feature = tf.expand_dims(self_feature, axis=2)
    self_feature = tf.tile(self_feature, [1, 1, anchor_num, 1])

    context = tf.tile(tf.expand_dims(tf.expand_dims(context, axis=1), axis=1), [1, 2, anchor_num, 1])
    dist_compute = tf.einsum('ijkl,l->ijk', tf.concat([self_feature, anchor_feature, context], 3), pf_w) + pf_b #batch_size 2 anchor_num
    messages = anchor_feature * tf.expand_dims(dist_compute, axis = 3)
    messages = tf.concat([messages, self_feature], axis=3)
    messages = tf.nn.relu(messages)
    out_position = tf.einsum('ijkl,l->ijk', messages, out_w) + out_b
    out_structure = tf.reduce_mean(messages, axis = 1)
    return out_position, out_structure

  def create_pgnn_heuristics_network(self):
    with tf.variable_scope("pf_network"):
      pf_w = tf.Variable(tf.truncated_normal([3 * lstm_size], stddev=0.1))
      pf_b = tf.Variable(tf.zeros(1))
    with tf.variable_scope("output_network"):
      out_w = tf.Variable(tf.truncated_normal([2 * lstm_size], stddev=0.1))
      out_b = tf.Variable(tf.zeros(1))
      out_pf_w = tf.Variable(tf.truncated_normal([2 * lstm_size], stddev=0.1))
      out_pf_b = tf.Variable(tf.zeros(1))

    with tf.variable_scope("value_network"):  
      with tf.variable_scope('value_output'):
        w_v1 = tf.Variable(tf.truncated_normal([loc_size + 2 * lstm_size, lstm_size], stddev=0.1))
        b_v1 = tf.Variable(tf.zeros(lstm_size))
        w_v2 = tf.Variable(tf.truncated_normal([4 * anchor_num + lstm_size + 1, 1], stddev=0.1))
        b_v2 = tf.Variable(tf.zeros(1))

    self.loc_embedding = tf.placeholder(tf.float32, shape=(None, lstm_size),name='all_emb') #get tensor by name

    self.anchor_set = tf.placeholder(tf.int32, shape=(batch_size, anchor_num), name='src')
    self.sub_sets = tf.placeholder(tf.int32, shape=(batch_size, 2), name='sub_graph')
    self.start_geo = tf.placeholder(tf.float32, shape=(batch_size, 2), name="start_geo")
    self.end_geo = tf.placeholder(tf.float32, shape=(batch_size, 2), name="end_geo")
    self.sub_geo = self.end_geo - self.start_geo
    self.heu_dist = tf.sqrt(self.sub_geo[:, 0] * self.sub_geo[:, 0] + self.sub_geo[:, 1] * self.sub_geo[:, 1])
    self.heu_dist = tf.expand_dims(self.heu_dist, -1)

#    self.src = self.st_known_[:, -1]
#    self.des = tf.squeeze(self.st_destination_, axis = 1)
 
    self.stop_des_emb = tf.stop_gradient(self.st_destination_embedding)
    self.stop_outputs = tf.stop_gradient(self.outputs[:, -1, :])
#    print(self.stop_des_emb.shape, self.stop_outputs.shape)
    self.value_layer_1 =tf.nn.relu(tf.matmul(tf.concat([self.stop_outputs, tf.squeeze(self.stop_des_emb, axis=1)], 1), w_v1) + b_v1)
#    tf.concat([tf.tile(self.st_destination_embedding, [1, tf.shape(self.st_known_)[1], 1]),  self.outputs], 2)

    x = self.loc_embedding 
    x_position, _ = self.geo_pgnn_layer(x, self.anchor_set, self.sub_sets, self.value_layer_1, pf_w, pf_b, out_w, out_b)
    p_position, _ = self.pf_pgnn_layer(x, self.anchor_set, self.sub_sets, self.value_layer_1, pf_w, pf_b, out_pf_w, out_pf_b)
    self.p_position = p_position
    self.x_position = x_position
#    x = tf.contrib.layers.embed_sequence(self.sub_sets, block_num, lstm_size, scope = "st_network/location_embedding", reuse = True)
#    x = tf.stop_gradient(x)
#    p_position[:, -1, :], p_position[:, -2, :]
    self.heuristics = tf.matmul(tf.concat([x_position[:, -2, :], x_position[:, -1, :], p_position[:, -2, :], p_position[:, -1, :], self.value_layer_1, self.heu_dist], 1), w_v2) + b_v2
#    self.heuristics = tf.matmul(self.value_layer_1, w_v2) + b_v2   
#####  supervised loss
    self.heuristics_input = tf.placeholder(tf.float32, shape=(batch_size), name='heuristics_input')
    self.heuristics_cost = tf.reduce_mean(tf.square(self.heuristics_input - self.heuristics))
#####


#      self.gradients = tf.gradients(self.heuristics_cost, [output_state])

    self.value_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "value_network")

    self.optimizer = tf.train.AdamOptimizer(0.0001).minimize(self.heuristics_cost, var_list=self.value_variables)

#    print("concat shape:", self.st_known_embedding[:, -1, :].shape, tf.squeeze(self.st_destination_embedding).shape, self.value_layer_1.shape)

#    self.pf_output = tf.matmul(tf.concat([tf.stop_gradient(self.st_known_embedding[:, -1, :]), tf.stop_gradient(self.st_destination_embedding[:, 0, :]), self.value_layer_1], 1), tf.expand_dims(pf_w, 1)) + pf_b

#    self.pf_cost = tf.reduce_mean(tf.square(self.heuristics_input - tf.squeeze(self.pf_output)))

#    self.pf_optimizer = tf.train.AdamOptimizer(0.0001).minimize(self.pf_cost, var_list=self.value_variables)


  def create_heuristics_network(self):
    with tf.variable_scope("value_network"):  
      with tf.variable_scope('value_output'):
        w_v1 = tf.Variable(tf.truncated_normal([2*lstm_size, lstm_size], stddev=0.1))
        b_v1 = tf.Variable(tf.zeros(lstm_size))
        w_v2 = tf.Variable(tf.truncated_normal([lstm_size, 1], stddev=0.1))
        b_v2 = tf.Variable(tf.zeros(1))
        
#self.outputs[-1, :, :]
      self.stop_des_emb = tf.stop_gradient(self.st_destination_embedding)
      self.stop_known_emb = tf.stop_gradient(self.st_known_embedding[:, -1, :])
      self.stop_outputs = tf.stop_gradient(self.rnn_outputs[:, -1, :])
      self.value_layer_1 =tf.nn.relu(tf.matmul(tf.concat([self.stop_outputs, tf.squeeze(self.stop_des_emb, axis=1)], 1), w_v1) + b_v1)

      self.src_bias_mat = tf.placeholder(tf.float32, shape=(batch_size, None, None),name='src_adj_mat')
      self.src_embedding = tf.placeholder(tf.float32, shape=(batch_size, None, lstm_size),name='src_emb') #get tensor by name
    
      self.des_bias_mat = tf.placeholder(tf.float32, shape=(batch_size, None, None),name='des_adj_mat')
      self.des_embedding = tf.placeholder(tf.float32, shape=(batch_size, None, lstm_size),name='des_emb') #get tensor by name

      self.src_geo_mat = tf.placeholder(tf.float32, shape=(batch_size, None, None), name='geo_src')
      self.des_geo_mat = tf.placeholder(tf.float32, shape=(batch_size, None, None), name='geo_des')
      h_1 = self.src_embedding
      for it in range(layers):  # layers 层数
        attns = []
        if it == 5:
          self.src_f = h_1
        print('zzzzzzzzzzzzz')
        print(h_1)
        for _ in range(heads):  # head数
          attns.append(self.attn_head(h_1, self.src_bias_mat, self.stop_outputs, self.src_geo_mat))     
        h_1 = tf.concat(attns, axis=-1)  
      # h_1 batch_size node_num lstm_size
      h_1 = h_1[:, 0, :]#tf.einsum('ij,ijk->ik', self.src_mask, h_1)
      print("-------------------------------")
      h_2 = self.des_embedding
      for it in range(layers):  # layers 层数
        attns = []
        if it == 3:
          self.des_f = h_2
        for _ in range(heads):  # head数
          attns.append(self.attn_head(h_2, self.des_bias_mat, self.stop_outputs, self.des_geo_mat))     
        h_2 = tf.concat(attns, axis=-1)  
      # h_1 batch_size node_num lstm_size
      h_2 = h_2[:, 0, :]#tf.einsum('ij,ijk->ik', self.src_mask, h_2)
      print(self.value_layer_1, h_1, h_2)
      self.heuristics = tf.matmul(self.value_layer_1 + h_1 + h_2, w_v2) + b_v2   #self.value_layer_1 + h_1 + h_2
      
#####  margin loss
#      margin = 1.0 - tf.slice(self.heuristics, [0, 0], [TRAIN_BATCH_SIZE, 1]) + tf.slice(self.heuristics, [TRAIN_BATCH_SIZE, 0], [TRAIN_BATCH_SIZE, 1])
#      condition = tf.less(margin, 0.)
#      self.heuristics_cost = tf.reduce_mean(tf.where(condition, tf.zeros_like(margin), margin))
#####

#####  supervised loss
      self.heuristics_input = tf.placeholder(tf.float32, shape=(batch_size), name='heuristics_input')
      self.heuristics_cost = tf.reduce_mean(tf.square(self.heuristics_input - self.heuristics))
#####


#      self.gradients = tf.gradients(self.heuristics_cost, [output_state])

      self.value_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                      "value_network")

      self.optimizer = tf.train.AdamOptimizer(0.0001).minimize(self.heuristics_cost, var_list=self.value_variables)


  def old_create_heuristics_network(self):
    with tf.variable_scope("value_network"):
      self.known_ = tf.placeholder(tf.int64, shape=(batch_size, input_steps), name='known')
      known_embedding = tf.contrib.layers.embed_sequence(self.known_, block_num, lstm_size, scope = "value_location_embedding")
      self.waiting_ = tf.placeholder(tf.int64, shape=(batch_size), name='waiting')
      waiting_embedding = tf.contrib.layers.embed_sequence(self.waiting_, block_num, lstm_size, scope = "value_location_embedding", reuse = True)
      self.destination_ = tf.placeholder(tf.int64, shape=(batch_size), name='destination')
      destination_embedding = tf.contrib.layers.embed_sequence(self.destination_, block_num, lstm_size, scope = "value_location_embedding", reuse = True)
    # network weights

      fw_cell, fw_initial_state = self.build_lstm(tf.shape(self.known_)[0])
      bw_cell, bw_initial_state = self.build_lstm(tf.shape(self.known_)[0])
#    print("-------", tf.concat([known_embedding, tf.expand_dims(waiting_embedding, 1)], 1))
      outputs, state = tf.nn.bidirectional_dynamic_rnn(fw_cell, bw_cell, tf.transpose(tf.concat([known_embedding, tf.expand_dims(waiting_embedding, 1)], 1), [1, 0, 2]), initial_state_fw=fw_initial_state, initial_state_bw=bw_initial_state, dtype=tf.float32, time_major=True)

      initial_state = tf.add(state[0], state[1])
      unstack_state = tf.unstack(initial_state, axis=0)
      tuple_state = tuple([tf.contrib.rnn.LSTMStateTuple(unstack_state[idx][0], unstack_state[idx][1]) for idx in range(num_layers)])

      hidden_states = tf.add(outputs[0], outputs[1])

      distant_embedding = waiting_embedding + destination_embedding
 #   W1 = self.weight_variable([self.state_dim,20])
 #   b1 = self.bias_variable([20])

      with tf.variable_scope('layer1'):
        local_w1 = tf.Variable(tf.truncated_normal([lstm_size, 2*lstm_size], stddev=0.1))
        local_b1 = tf.Variable(tf.zeros(2*lstm_size))

      local_1_layer = tf.nn.relu(tf.matmul(distant_embedding, local_w1) + local_b1)

      with tf.variable_scope('layer2'):
        local_w2 = tf.Variable(tf.truncated_normal([2*lstm_size, lstm_size], stddev=0.1))
        local_b2 = tf.Variable(tf.zeros(lstm_size))

      local_2_layer = tf.nn.relu(tf.matmul(local_1_layer, local_w2) + local_b2)
  
#      output_state = tf.reduce_mean(hidden_states, 0) + local_2_layer

#      print(hidden_states, initial_state.shape, local_2_layer.shape)
      output_state = hidden_states[-1, :, :] + local_2_layer

      with tf.variable_scope('output'):
        w_h = tf.Variable(tf.truncated_normal([lstm_size, 1], stddev=0.1))
        b_h = tf.Variable(tf.zeros(1))
        w_t = tf.Variable(tf.truncated_normal([lstm_size, 1], stddev=0.1))
        b_t = tf.Variable(tf.zeros(1))
      

#      self.heuristics = tf.nn.sigmoid(tf.matmul(output_state, w_h) + b_h)
      self.heuristics = tf.matmul(output_state, w_h) + b_h

      self.time = tf.nn.relu(tf.matmul(output_state, w_t) + b_t)

      self.heuristics_input = tf.placeholder(tf.float32, [batch_size], name = "heuristics_input")

      self.time_input = tf.placeholder(tf.float32, [batch_size], name = "time_input")

#      half_batch_size = tf.div(self.heuristics.shape[0], 2)
      print(tf.slice(self.heuristics, [0, 0], [TRAIN_BATCH_SIZE, 1]).shape, tf.slice(self.heuristics, [TRAIN_BATCH_SIZE, 0], [TRAIN_BATCH_SIZE, 1]).shape)
      margin = 1.0 - tf.slice(self.heuristics, [0, 0], [TRAIN_BATCH_SIZE, 1]) + tf.slice(self.heuristics, [TRAIN_BATCH_SIZE, 0], [TRAIN_BATCH_SIZE, 1])

      condition = tf.less(margin, 0.)

########  supervised learning
      self.heuristics_input = tf.placeholder(tf.float32, shape=(batch_size), name='heuristics_input')
      self.heuristics_cost = tf.reduce_mean(tf.square(self.heuristics_input - self.heuristics))
########

########### margin loss
#      self.heuristics_cost = tf.reduce_mean(tf.where(condition, tf.zeros_like(margin), margin))
###########

#      self.heuristics_cost = tf.reduce_mean(tf.square(self.heuristics_input - self.heuristics))

      self.gradients = tf.gradients(self.heuristics_cost, [output_state])
#      self.heuristics_cost = tf.reduce_mean(tf.square(self.heuristics_input - self.heuristics)) 


  #  y_reshaped = tf.reshape(y_one_hot, logits.get_shape())

    # Softmax cross entropy loss
      self.optimizer = tf.train.AdamOptimizer(0.0001).minimize(self.heuristics_cost)
