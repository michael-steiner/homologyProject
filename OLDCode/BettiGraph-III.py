import networkx as nx
import matplotlib.pyplot as plt
import graphviz
import numpy as np
import pandas as pd
import itertools as it

#####################definitions###########################

def findPathsNoLC(G,u,n):
    if n==0:
        return [[u]]
    paths = []
    for neighbor in G.neighbors(u):
        for path in findPathsNoLC(G,neighbor,n-1):
            if u not in path:
                paths.append([u]+path)
    return paths

def create_directed_graph(number_of_nodes, std_of_dist=1,center_of_dist = 0,list_of_edges = [],):

    num_nodes = number_of_nodes
    edge_list = list_of_edges
    cent_dist = center_of_dist
    std_dist = std_of_dist
    node_values_G = []
    G = nx.DiGraph()
    list_of_nodes = [n for n in range(num_nodes)]

    #for i in range(num_nodes):
       # node_values_G.append(np.random.normal(loc=0.0, scale=1.0))

    if(edge_list == []):
        edge_list_iterator = it.combinations(list_of_nodes, 2)
        for i in edge_list_iterator:
            edge_list.append(i)

    for ii in range(num_nodes):
        node_values_G.append(np.random.normal(cent_dist, std_dist))


    for i in range(len(list_of_nodes)):
        G.add_node(list_of_nodes[i], node_value=node_values_G[i])

    G.add_edges_from(edge_list)

    return G

def create_cloud_of_points(graph, dimension_of_cloud):
    cloud_dim = dimension_of_cloud
    G = graph
    list_of_paths = []
    list_of_points_in_n_space = []
    for g_node in G.nodes():
        list_of_paths.extend(findPathsNoLC(G, g_node,cloud_dim ))

    for n_walk in list_of_paths:
        temp_point_list = []
        for n_node in n_walk:
            temp_point_list.append(G.node[n_node]['node_value'])
        list_of_points_in_n_space.append(temp_point_list)

    df_points = pd.DataFrame( list_of_points_in_n_space)

    return df_points

def generate_save_point_clouds(cloud_name,number_of_clouds, number_of_nodes, dim_of_cloud, save_path = "C:\\Users\micha\PycharmProjects\\betti\cloud_data\\",stDev=1, center_of_dist = 0 ):

    cloudName = cloud_name
    numClouds = number_of_clouds
    numNodes = number_of_nodes
    dimOfCloud = (dim_of_cloud -1)
    savePath = save_path
    standardDev = stDev
    centerOfDist = center_of_dist

    for cloudNumber in range(numClouds):
        G_temp = create_directed_graph(numNodes,standardDev,centerOfDist)
        tempCloud = create_cloud_of_points(G_temp,dimOfCloud)
        tempCloud.to_csv(savePath + cloudName + str(cloudNumber) + ".csv", index=False)


##############################################



generate_save_point_clouds("funCloud",500,10,3,"C:\\Users\\micha\\PycharmProjects\\betti\\cloud_data\\Class1\\",1,0)


#"C:\\Users\\micha\\PycharmProjects\\betti\\cloud_data\\Class1\\"
