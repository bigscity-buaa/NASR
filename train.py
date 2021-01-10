import tensorflow as tf
import numpy as np
import networkx as nx
import random
import pickle
import copy
from collections import deque
import os
import math
from model import *
from utils import *

os.environ['CUDA_VISIBLE_DEVICES']='3'
# Hyper Parameters for DAN
PRE_TRAIN = False
TEST = True
RESTORE = True
GAMMA = 1.0 # discount factor for target Q
INITIAL_EPSILON = 0.5 # starting value of epsilon
FINAL_EPSILON = 0.01 # final value of epsilon
batch_size = None # size of minibatch
input_steps = None
user_size = 128
user_num = 15000
block_num = 16500
lstm_size = 384
num_layers = 2
TRAIN_BATCH_SIZE = 100 #训练输入的batch 大小
INFERENCE_BATCH_SIZE = 1 #推断的时候输入的batch 大小
PRE_EPISODE = 600
NEG_SAMPLES = 9
NEXT_ACTION_NUM = 3
train_set_size = 5000
EPISODE = 100 # Episode limitation
PRE_EPISODE = 300
TRAIN_BATCHES = 300 # Step limitation in an episode
anchor_num = 64

def train_heuristics_network(self):
    self.time_step += 1
# Step 1: obtain random minibatch from replay memory
    minibatch = random.sample(self.replay_buffer,BATCH_SIZE)
    state_batch = [data[0] for data in minibatch]
    action_batch = [data[1] for data in minibatch]
    reward_batch = [data[2] for data in minibatch]
    next_state_batch = [data[3] for data in minibatch]

# Step 2: calculate y
    y_batch = []
    Q_value_batch = self.Q_value.eval(feed_dict={self.state_input:next_state_batch})
    for i in range(0,BATCH_SIZE):
        done = minibatch[i][4]
        if done:
            y_batch.append(reward_batch[i])
        else:
            y_batch.append(reward_batch[i] + GAMMA * np.max(Q_value_batch[i]))

            self.optimizer.run(feed_dict={
                    self.y_input:y_batch,
                    self.action_input:action_batch,
                    self.state_input:state_batch
                    })

def train_st_network(model, PRE_EPISODE):
    trainData = pickle.load(open("/data/wuning/NTLR/beijing/train_loc_set", "rb"))
    trainTimeData = pickle.load(open("/data/wuning/NTLR/beijing/train_time_set", "rb"))
    trainUserData = pickle.load(open("/data/wuning/NTLR/beijing/train_user_set", "rb"))
    trainGeoData = pickle.load(open("/data/wuning/NTLR/beijing/train_user_set", "rb"))
    historyData = pickle.load(open("/data/wuning/NTLR/beijing/userIndexedHistoryAttention", "rb"))
    for episode in range(PRE_EPISODE):
      counter = 0
      for tra_bat, time_bat, user_bat, geo_bat, his_bat, his_time_bat, his_user_bat, his_mask_bat in generate_st_batch(historyData, trainData[:train_set_size], trainTimeData[:train_set_size], trainUserData[:train_set_size], trainGeoData[:train_set_size]):
        if(len(tra_bat[0]) < 15 or len(tra_bat[0]) > 100):
          continue
#        print(tra_bat.shape, tra_mask_bat.shape, hour_bat.shape, day_bat.shape, his_bat.shape, his_hour_bat.shape, his_day_bat.shape, his_mask_bat.shape)
        _, eval_st_loss = model.session.run([model.st_all_optimizer, model.st_all_cost],feed_dict={
          model.st_known_:tra_bat[:, :-1],
          model.st_destination_:tra_bat[:, -1][:, np.newaxis],
          model.st_output_:tra_bat[:, 1:],
          model.st_time:time_bat[:, :-1],
          model.st_user:user_bat[:, np.newaxis],
          model.his_tra:his_bat,
          model.his_time:his_time_bat,
          model.his_user:his_user_bat,
          model.his_padding_mask:his_mask_bat
        })
#        eval_st_loss = model.st_cost.eval(feed_dict={
#          model.st_known_:batch[0],
#          model.st_destination_:batch[2],
#          model.st_output_:batch[1],
#          model.trans_mat:batch[3]
#        })

        if counter % 100 == 0:
          print("epoch:{}...".format(episode),
            "batch:{}...".format(counter),
            "loss:{:.4f}...".format(eval_st_loss))
        counter += 1
      model.all_saver.save(model.session, "/data/wuning/learnAstar/beijingComplete/pre_train_ut_attn_neural_network_epoch{}.ckpt".format(episode))  


def Time_diff_attn(model):
    trainData = pickle.load(open("/data/wuning/NTLR/beijing/train_loc_set", "rb"))
    trainGeoData = pickle.load(open("/data/wuning/NTLR/beijing/train_geo_set", "rb"))
    trainTimeData = pickle.load(open("/data/wuning/NTLR/beijing/train_time_set", "rb"))
    trainUserData = pickle.load(open("/data/wuning/NTLR/beijing/train_user_set", "rb"))
    historyData = pickle.load(open("/data/wuning/NTLR/beijing/userIndexedHistoryAttention", "rb"))

    sub_graph_nodes = pickle.load(open("/data/wuning/NTLR/beijing/sub_graph_nodes", "rb"))
    sub_graph_adjs = pickle.load(open("/data/wuning/NTLR/beijing/sub_graph_adjs", "rb"))
    sub_graph_nodes = np.array(sub_graph_nodes)
    sub_graph_adjs = np.array(sub_graph_adjs)

    model.st_saver.restore(model.session, "/data/wuning/learnAstar/beijingComplete/pre_train_ut_attn_neural_network_epoch20.ckpt")
    graphData = pickle.load(open("/data/wuning/map-matching/allGraph", "rb"))
    #/data/wuning/NTLR/beijing/anchor_dbscan_set_64            /data/wuning/NASR-TKDE/data/beijing_random_anchor_set 
    anchor_set = pickle.load(open("/data/wuning/NTLR/beijing/anchor_dbscan_set_64", "rb"))
    distMat = pickle.load(open("/data/wuning/NTLR/beijing/distMat", "rb"))
#    variable_names = [v.name for v in tf.trainable_variables()]
#    print(variable_names)
#    anchor_set = np.array([])
    location_embeddings = model.session.run(tf.get_default_graph().get_tensor_by_name("st_network/location_embedding/embeddings:0"))
    location_embeddings = np.array(location_embeddings)
#    anchor_set = [random.randint(0, 16499) for i in range(anchor_num)]
    print("zzzzzzzzzzzzzzzzzz")
    print("???", np.array(location_embeddings).shape)

    adj = np.matrix(graphData)[:block_num, :block_num]
    print(adj.shape)
    G = nx.from_numpy_matrix(adj, create_using=nx.DiGraph())
    inv_G = nx.from_numpy_matrix(adj.T, create_using=nx.DiGraph())
    train_set_size = 5000
    losses = []                                    
    for episode in range(PRE_EPISODE):
      counter = 0
      for tra_bat, time_bat, user_bat, geo_bat, his_bat, his_time_bat, his_user_bat, his_mask_bat in generate_st_batch(historyData, trainData[:train_set_size], trainTimeData[:train_set_size], trainUserData[:train_set_size], trainGeoData[:train_set_size]):
        heuristics_batches = []
        if(len(tra_bat[0]) < 15 or len(tra_bat[0]) > 100):
          continue

        _, eval_st_all_prob = model.session.run([model.st_all_optimizer, model.st_all_prob], feed_dict={
            model.st_known_:tra_bat[:, :-1],
            model.st_output_:tra_bat[:, 1:],
            model.st_destination_:tra_bat[:, -1][:, np.newaxis],
            model.st_time:time_bat[:, :-1],
            model.st_user:user_bat[:, np.newaxis],
            model.his_tra:his_bat,
            model.his_time:his_time_bat,
            model.his_user:his_user_bat,
            model.his_padding_mask:his_mask_bat
        })
        idx = tra_bat[:, 1:]
        m_idx, n_idx = idx.shape
        I, J = np.ogrid[:m_idx, :n_idx]
        eval_st_all_prob = eval_st_all_prob[I, J, idx]
        eval_st_all_prob[eval_st_all_prob < 0.05] = 0.05
        for k in range(len(tra_bat[0]) - 6, 0, -1):
          item_heu_batch = - np.log(eval_st_all_prob[:, k])
          for l in range(k, k + 5):
            item_heu_batch += - (0.96)**(l - k) * np.log(eval_st_all_prob[:, l])

          item_heu_batch = np.array(item_heu_batch)
#          eval_policy[0][:, :, ]
          item_known_batch = tra_bat[:, : k + 1]
          item_des_batch = tra_bat[:, -1]
          item_time_batch = time_bat[:, : k + 1]
          item_user_batch = user_bat
          item_des_nodes_batch = sub_graph_nodes[item_des_batch]
          item_des_adj_batch = sub_graph_adjs[item_des_batch]
          item_src_nodes_batch = sub_graph_nodes[tra_bat[:, k]]
          item_src_adj_batch = sub_graph_adjs[tra_bat[:, k]]

          item_heu_future_batch = tra_bat[:,: k + 6]
          item_time_future_batch = time_bat[:, : k + 6]

          heuristics_batches.append([item_known_batch, item_des_batch, item_heu_batch, item_time_batch, item_user_batch, item_time_future_batch, item_heu_future_batch, item_des_nodes_batch, item_src_nodes_batch, item_des_adj_batch, item_src_adj_batch])
        feed_data = {}
        for heu_batch in heuristics_batches:
            heuristics = model.heuristics.eval(feed_dict={
                model.st_known_:heu_batch[-5],
                model.st_destination_:np.array(heu_batch[1])[:, np.newaxis],
                model.heuristics_input:heu_batch[2],
                model.st_time:heu_batch[-6],
                model.st_user:heu_batch[4][:, np.newaxis],
                model.his_tra:his_bat,
                model.his_time:his_time_bat,
                model.his_user:his_user_bat,
                model.his_padding_mask:his_mask_bat,

                model.src_bias_mat:heu_batch[-1],
                model.des_bias_mat:heu_batch[-2],
                model.src_embedding:location_embeddings[heu_batch[-3]],
                model.des_embedding:location_embeddings[heu_batch[-4]],
              })
            #print(np.array(heu_batch[2]).shape, np.array(heuristics).shape)
            heu_batch[2] += 0.82 * heuristics[:, 0]
             
            model.optimizer.run(feed_dict = {
                model.st_known_:heu_batch[0],
                model.st_destination_:np.array(heu_batch[1])[:, np.newaxis],
                model.heuristics_input:heu_batch[2],
                model.st_time:heu_batch[3],
                model.st_user:heu_batch[4][:, np.newaxis],
                model.his_tra:his_bat,
                model.his_time:his_time_bat,
                model.his_user:his_user_bat,
                model.his_padding_mask:his_mask_bat,
                
                model.src_bias_mat:heu_batch[-1],
                model.des_bias_mat:heu_batch[-2],
                model.src_embedding:location_embeddings[heu_batch[-3]],
                model.des_embedding:location_embeddings[heu_batch[-4]]
                }
            )

        heuristics_batches = []
        losses.append(0)
        counter += 1
        if counter % 100 == 0:  
          print("counter:", counter)
      print(losses)
      print("heuristics:", heuristics)
      model.all_saver.save(model.session, "/data/wuning/AstarRNN/train_pgnn_geo_length_dbscan_epoch{}.ckpt".format( episode))





def main():
    AstarRNN = DAN()
#    train_st_network(AstarRNN, 50)
    Time_diff_attn(AstarRNN)
if __name__ == '__main__':
    main()

