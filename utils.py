import numpy as np
import networkx as nx
#testGraph = [
#[0,0,0,0,1,0,0,0],
#[0,0,0,1,0,0,0,0],
#[0,0,0,0,0,1,0,0],
#[0,0,1,0,0,0,0,0],
#[0,0,0,1,0,1,1,0],
#[0,0,0,0,0,0,1,1],
#[0,0,0,0,0,0,0,1],
#[0,0,0,0,0,0,0,0]
#                ]
#adj = np.matrix(testGraph)
def knbrs(G, start, k):
  all_node = []
  nbrs = set([start])
  all_node.extend(list(nbrs))
  for l in range(k):
    try:  
      nbrs = set((nbr for n in nbrs for nbr in G[n]))
    except:
      nbrs = set([])
    all_node.extend(list(nbrs))
  for i in range(len(all_node)):
    if all_node[i] >= 15200:
      all_node[i] = i    
  return_node = []
  for i in all_node:
    if i not in return_node:
      return_node.append(i)
  return return_node
def generate_one_graph(road_embedding, G, road_num, cutoff=10, src=0):
  
  node_set = set()
  src_node = []
  src_emb = []
  src_node = knbrs(G, src, cutoff)
  while len(src_node) < road_num + 15:
    src_node.append(len(src_node))
#  if(len(src_node)<10):
#    print(len(src_node))    
  src_emb = [road_embedding[n] for n in src_node]
  src_adj = nx.adjacency_matrix(G.subgraph(src_node)).todense()
  src_adj = np.array(src_adj[:road_num, :road_num])
  adj = np.eye(road_num)   #self-loop
  adj[:src_adj.shape[0],:src_adj.shape[1]] += src_adj

  adj = (adj - 1) * (9999999.0)
 
  mask = [0 for i in range(road_num)]
  mask[0] = 1
  return adj, src_emb[:road_num], mask    
def generate_sub_graph(road_embedding, G, inv_G, road_num, cutoff=10, src=0, des=7):
# input:
# road_embedding feature matrix of roads  [road_num * feature_dim]
# road_network  road adj matrix   [road_num * road_num]
# road_num the size of sub graph
# cutoff max length of path between src and des
# src  source node  (int)
# def  destination node (int)
# return:
# src_adj  the adj matrix of subgraph
# src_emb  the corresponding emb matrix of subgraph
# des_adj  
# des_emb
  node_set = set()
#  des_G = nx.DiGraph()
#  src_G = nx.DiGraph()
  src_node = []
  des_node = []
  src_emb = []
  des_emb = []
#  print(np.matrix(road_network).shape)
#  adj = np.matrix(road_network)[:15500, :15500]
#  G = nx.from_numpy_matrix(adj, create_using=nx.DiGraph())
#  inv_G = nx.from_numpy_matrix(adj.T, create_using=nx.DiGraph())

#  paths_between_generator = nx.all_simple_paths(G,source=src,target=des,cutoff=cutoff)
#  for path in reversed(list(paths_between_generator)):
#    print(path)  
#    sub_G.add_path(path) 
#    for item in path:
#      if not item in node_set:
#        sub_emb.append(road_embedding[item])  
#        node_set.append(item)
#    if len(sub_emb) >= road_num:
#      break 
  src_node = knbrs(G, src, cutoff)
  des_node = knbrs(inv_G, des, cutoff)
  while len(src_node) < road_num + 15:
    src_node.append(len(src_node))
  while len(des_node) < road_num + 15:
    des_node.append(len(des_node))   
  src_emb = [road_embedding[n] for n in src_node]
  des_emb = [road_embedding[n] for n in des_node]
  des_adj =  nx.adjacency_matrix(inv_G.subgraph(des_node)).todense()
  src_adj = nx.adjacency_matrix(G.subgraph(src_node)).todense()

  src_adj = np.array(src_adj[:road_num, :road_num])
  des_adj = np.array(des_adj[:road_num, :road_num])

  s_adj = np.eye(road_num)   #self-loop
  s_adj[:src_adj.shape[0],:src_adj.shape[1]] += src_adj

  s_adj = (s_adj - 1) * (9999999.0)


  d_adj = np.eye(road_num)   #self-loop
  d_adj[:des_adj.shape[0],:des_adj.shape[1]] += des_adj

  d_adj = (d_adj - 1) * (9999999.0)

  mask = [0 for i in range(road_num)]
  mask[0] = 1
  return d_adj, s_adj, des_emb[:road_num], src_emb[:road_num], mask, mask, src_node[:road_num], des_node[:road_num]      

def generate_batch(maskData, historyData, trainData, trainTimeData, trainUser):
  for mask_bat, tra_bat, time_bat, user_bat in zip(maskData, trainData, trainTimeData, trainUser):
    hour_bat = np.array(time_bat)[:, :, 0] 
    day_bat = np.array(time_bat)[:, :, 1]
    his_bat = []
    his_hour_bat = []
    his_day_bat = []
    his_mask_bat = []
    for user in user_bat:
      his_bat.append([item[0] for item in historyData[user]])
      his_hour_bat.append([[time[0] for time in item[1]] for item in historyData[user]])
      his_day_bat.append([[time[1] for time in item[1]] for item in historyData[user]])
      his_mask_bat.append([item[2] for item in historyData[user]])

#    mask_bat = np.sum(np.array(mask_bat)[:, 1:], 1)
#    des = [tra[ind] for tra, ind in zip(tra_bat, mask_bat)]
    yield np.array(tra_bat), np.array(hour_bat), np.array(day_bat), np.array(his_bat)[:, :, :-1], np.array(his_hour_bat)[:, :, :-1], np.array(his_day_bat)[:, :, :-1], np.array(his_mask_bat)[:, :, :-1]
#    for i in range(2):
#      yield np.array(des[i*50: (i+1)*50]), np.array(tra_bat)[i*50: (i+1)*50], np.array(mask_bat)[i*50: (i+1)*50], np.array(hour_bat)[i*50: (i+1)*50, :-1], np.array(day_bat)[i*50: (i+1)*50, :-1], np.array(his_bat)[i*50: (i+1)*50, :, :-1], np.array(his_hour_bat)[i*50: (i+1)*50, :, :-1], np.array(his_day_bat)[i*50: (i+1)*50, :, :-1], np.array(his_mask_bat)[i*50: (i+1)*50, :, :-1]   


def generate_st_batch(historyData, trainData, trainTimeData, trainUser, trainGeo):
  for tra_bat, time_bat, user_bat, geo_bat in zip(trainData, trainTimeData, trainUser, trainGeo):
    time_bat = np.array(time_bat)[:, :, 0] * np.array(time_bat)[:, :, 1] 
    his_bat = []
    his_time_bat = []
    his_user_bat = []
    his_mask_bat = []
    for user in user_bat:
      his_bat.append([item[0] for item in historyData[user][:5]])
      his_time_bat.append([[np.array(time[0]) * np.array(time[1]) for time in item[1]] for item in historyData[user][:5]])
      his_user_bat.append([[user for item_ in range(50)] for item in historyData[user][:5]])
      his_mask_bat.append([[0.0 for item_ in range(50)] for item in historyData[user][:5]])
    for i in range(len(his_bat)):
      for j in range(len(his_bat[i])):
        if len(his_bat[i][j]) > 50:
          his_bat[i][j] = his_bat[i][j][:50]
          his_time_bat[i][j] = his_time_bat[i][j][:50]
          for k in range(len(his_bat[i][j])):
            his_mask_bat[i][j][k] = 1.0                
        else:
          for k in range(len(his_bat[i][j])):
            his_mask_bat[i][j][k] = 1.0                
          while len(his_bat[i][j]) < 50:
            his_bat[i][j].append(0)
          while len(his_time_bat[i][j]) < 50:
            his_time_bat[i][j].append(0)    
    yield np.array(tra_bat), np.array(time_bat), np.array(user_bat), np.array(geo_bat), np.array(his_bat), np.array(his_time_bat), np.array(his_user_bat), np.array(his_mask_bat)
#






