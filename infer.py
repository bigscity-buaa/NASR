import tensorflow as tf
import numpy as np
import random 
import pickle
import copy
from collections import deque
import os
import math  
import datetime
from model import *
from utils import *
from metric import *
os.environ['CUDA_VISIBLE_DEVICES']='2'

def reconstruct_path(model, cameFrom, current):
  total_path = [current]
  while current in cameFrom:
    current = cameFrom[current]
    total_path.append(current)
  return total_path

def insert(model, item, the_list, f_score):  #插入 保持从小到大的顺序
  if(len(the_list) == 0):
    return [item]
  for i in range(len(the_list)):
    if(f_score[the_list[i]] > f_score[item]):
      the_list.insert(i, item)
      break
    if i == len(the_list) - 1:
      the_list.append(item)
  return the_list

def move(model, item, the_list, f_score):
  for it in the_list:
    if(f_score[it] == f_score[item]):
      the_list.remove(it)
      break
  return insert(model, item, the_list, f_score)
def greedyTest(model):
  counters = 0
  all_len = 0
  for tra_bat, hour_bat, day_bat, his_bat, his_hour_bat, his_day_bat, his_mask_bat in generate_batch(maskData[0:2000], historyData, trainData[20000:22000], trainTimeData[0:2000], trainUserData[0:2000]):
    result = []
    print("tra_bat_shp:", tra_bat.shape, hour_bat.shape)
    tra_bat = np.array(tra_bat[:, :10])
    starttime = datetime.datetime.now()
    NodeNum = 0
    for item_0, item_1, item_2, item_3, start, unknown, end in zip(his_bat, his_hour_bat, his_day_bat, his_mask_bat, np.array(tra_bat)[:, 0], np.array(tra_bat)[:,1:-1], np.array(tra_bat)[:, -1]):
      path = [start]
      for i in range(len(unknown)):
        NodeNum += 1  
#          print("path:", path)
        st_value = model.st_all.eval(
            feed_dict={
              model.st_known_:np.array(path)[np.newaxis, :],
              model.st_destination_:[[end]],
#              model.his_tra:[item_0],
#              model.his_time:[item_1],
#              model.his_day:[item_2],
#              model.his_padding_mask:[item_3]
          })
        policy = np.argmax(st_value, axis=2)[:, -1]
        if policy[0] == end:
          break    
        path.append(policy[0])
      path.append(end)
      result.append(path)
    endtime = datetime.datetime.now()
    for infer, real in zip(result, tra_bat):
#      counters += edit(infer, real)
      try:  
        print("infer:", len(infer), [locList[ite] for ite in infer])
        print("real:", len(real), [locList[ite] for ite in real])
      except:
        print(infer)
      for item in infer[1:-1]:
        if item in real[1:-1]:
          counters += 1
      all_len += len(infer) - 2
#    all_len += len(result) 
#    all_edt = all_edt / all_len
    print ("time:", (endtime - starttime).seconds)
    print("NodeNum:", NodeNum)
    print(counters, float(counters)/all_len)

def AstarTestSoftmax(model):
  sub_graph_nodes = pickle.load(open("/data/wuning/NTLR/beijing/sub_graph_nodes", "rb"))
  sub_graph_adjs = pickle.load(open("/data/wuning/NTLR/beijing/sub_graph_adjs", "rb"))
  sub_graph_nodes = np.array(sub_graph_nodes)
  sub_graph_adjs = np.array(sub_graph_adjs)
  results = []
  counters = 0
  all_len = 0
  all_search_count = 0
  print(np.array(trainData[6000:]).shape)
  location_embeddings = model.session.run(tf.get_default_graph().get_tensor_by_name("st_network/location_embedding/embeddings:0"))
  location_embeddings = np.array(location_embeddings)
  anchor_set = pickle.load(open("/data/wuning/NASR-TKDE/data/beijing_random_anchor_set", "rb"))
  adj = np.matrix(graphData)[:15208, :15208]
  G = nx.from_numpy_matrix(adj, create_using=nx.DiGraph())
  inv_G = nx.from_numpy_matrix(adj.T, create_using=nx.DiGraph())
  complex_count = 0
  count = 0
  for tra_bat, time_bat, user_bat, geo_bat, his_bat, his_time_bat, his_user_bat, his_mask_bat in generate_st_batch(historyData, trainData[5000:], trainTimeData[5000:], trainUserData[5000:], trainGeoData[5000:]):
#  for batch in testData:
    result = []
    if len(tra_bat[0]) < 40:
      continue    
    tra_bat = np.array(tra_bat[:, :20])
    start_times = time_bat[:, 0]
    users = user_bat
    for start, unknown, end, start_time, user, geo, his_tra, his_time, his_user, his_mask in zip(np.array(tra_bat)[:, 0], np.array(tra_bat)[:,1:-1], np.array(tra_bat)[:, -1], start_times, users, geo_bat, his_bat, his_time_bat, his_user_bat, his_mask_bat):
#      if not count + 1 in complexSamples:
#        count += 1
#        continue  
      complex_count += 1    
      allNode = []
      closedSet = []
      openSet = [start]
      cameFrom = {}
      pathFounded = {start: [start]}
      gScore = {}
      fScore = {}
      gScore[start] = 0
      fScore[start] = 0
      waitingTra = 0
      bestScore = 10000000
      bestTra = []
      searchCount = 0
      while len(openSet) > 0:
        searchCount += 1
        current = openSet[0]
        allNode.append(current)
        openSet.remove(current)
        closedSet.append(current)
        if gScore[current] < bestScore and len(pathFounded[current]) == (len(unknown) + 1):
          bestScore = gScore[current]
          bestTra = copy.deepcopy(pathFounded[current])
          bestTra.append(end)
#        if len(pathFounded[current]) > (len(unknown) + 1) or len(pathFounded[current]) == (len(unknown) + 1):
#          continue
        if current == end:
          bestTra = copy.deepcopy(pathFounded[current])
          bestTra.append(end)
#          result.append(bestTra)  
          print("advance")  
          break
#        print(user, [(start_time + i // 8) % 576 for i in range(len(pathFounded[current]))])      
        st_value = model.st_all_prob.eval(
          feed_dict={
            model.st_known_:[pathFounded[current]],
            model.st_destination_:[[end]],
            model.st_time:[[int((start_time + i // 8) % 576) for i in range(len(pathFounded[current]))]],
            model.st_user:[[int(user)]],
            model.his_tra:[his_tra],
            model.his_time:[his_time],
            model.his_user:[his_user],
            model.his_padding_mask:[his_mask]
        })

#        st_value = np.exp(st_value)/np.sum(np.exp(st_value))
#        policy_value = np.exp(policy_value)/np.sum(np.exp(policy_value), axis=2)[:, :, np.newaxis]
#        st_arg = np.argsort(-st_value, axis=2)[0, -1][:2]
        st_arg = G[current]
        for waiting in st_arg:
          if (waiting in closedSet):
            continue
          prob = st_value[0][-1][waiting]  
          if prob == 0.0:
            prob = 0.00000000001    
          one_step_value = -math.log(prob)

#          if one_step_value == 0:
#              one_step_value = 0.01
          g_score = one_step_value + gScore[current]#((1 - len(pathFounded[current]) / 20)) * st_value[waiting_count][0] + gScore[current]
#            f_score = np.array(policy_value)[-1, -1, waiting] + fScore[current]           
          temp = copy.deepcopy(pathFounded[current])
          temp.append(waiting)
#          src_adj, des_adj, des_emb, src_emb, des_mask, src_mask, src_node, des_node = generate_sub_graph(location_embeddings, G, inv_G, 500, src=waiting, des=end)
          src_bias_mat = sub_graph_adjs[waiting]
          des_bias_mat = sub_graph_adjs[end]
          src_nodes = sub_graph_nodes[waiting]
          des_nodes = sub_graph_nodes[end]
          src_embs = location_embeddings[src_nodes]
          des_embs = location_embeddings[des_nodes]
          h_score =  model.heuristics.eval(
            feed_dict={
              model.st_known_:np.array(temp)[np.newaxis, :],
              model.st_destination_:[[end]],
              model.st_time:[[int((start_time + i // 8) % 576) for i in range(len(temp))]],
              model.st_user:[[int(user)]],#[int(user)],
              model.his_tra:[his_tra],
              model.his_time:[his_time],
              model.his_user:[his_user],
              model.his_padding_mask:[his_mask],

              model.src_bias_mat:[src_bias_mat],
              model.des_bias_mat:[des_bias_mat],
              model.src_embedding:[src_embs],
              model.des_embedding:[des_embs],

            }
          )

          h_score /= 10.0
          sub_geo = np.array(id2geo[waiting]) - np.array(id2geo[end])
          d_score = math.sqrt(sub_geo[0]**2 + sub_geo[1]**2) * 100 
          h_score += d_score 

          if (waiting in gScore) and (g_score < gScore[waiting]):
            continue
          gScore[waiting] = g_score
          fScore[waiting] = gScore[waiting] + h_score
          if waiting not in openSet:
            openSet = insert(model, waiting, openSet, fScore)
          else:
            openSet = move(model, waiting, openSet,fScore)
          pathFounded[waiting] =  temp

        if(searchCount >= 1000):
          openSet = []
      count += 1
      print("count:", count, "complex_count:", complex_count, "searchCount:",  searchCount, "best_score:",  bestScore, "bestTra:", bestTra, "allNode:", len(allNode))
      all_search_count += searchCount

      result.append(bestTra)
    results.append(result)
    for infer, real in zip(result, tra_bat):
      for item in infer[1:-2]:
        if item in real[1:-1]:
          counters += 1
      all_len += len(real) - 2
    print(all_search_count, counters, all_len, float(counters)/all_len)
  return float(counters)/all_len


graphData = pickle.load(open("/data/wuning/NTLR/beijing/CompleteAllGraph", "rb"))

locList = pickle.load(open("/data/wuning/NTLR/beijing/locList", "rb"))
trainData = pickle.load(open("/data/wuning/NTLR/beijing/train_loc_set", "rb"))
trainTimeData = pickle.load(open("/data/wuning/NTLR/beijing/train_time_set", "rb"))
trainUserData = pickle.load(open("/data/wuning/NTLR/beijing/train_user_set", "rb"))
trainGeoData = pickle.load(open("/data/wuning/NTLR/beijing/train_geo_set", "rb"))
historyData = pickle.load(open("/data/wuning/NTLR/beijing/userIndexedHistoryAttention", "rb"))
id2geo = pickle.load(open("/data/wuning/NTLR/beijing/id2geo", "rb"))
distMat = pickle.load(open("/data/wuning/NTLR/beijing/distMat", "rb"))
complexSamples = pickle.load(open("/data/wuning/NASR-TKDE/data/complex_samples", "rb"))


def main():
    AstarRNN = DAN()
    AstarRNN.all_saver.restore(tf.get_default_session(), "/data/wuning/AstarRNN/train_pgnn_geo_length_dbscan_epoch1.ckpt")

    AstarTestSoftmax(AstarRNN)
if __name__ == '__main__':
    main()

