def findPathsNoLC(G,u,n):
    if n==0:
        return [[u]]
    paths = []
    for neighbor in G.neighbors(u):
        for path in findPathsNoLC(G,neighbor,n-1):
            if u not in path:
                paths.append([u]+path)
    return paths

def create_diected_graph(number_of_nodes, list_of_edges = [], center_of_dist = 0, std_of_dist=1):

    num_nodes = number_of_nodes
    edge_list = list_of_edges
    cent_dist = center_of_dist
    std_dist = std_of_dist
    node_values_G = []
    G = nx.DiGraph()
    list_of_nodes = [n for n in range(num_nodes)]

    for i in range(num_nodes):
        node_values_G.append(np.random.normal(loc=0.0, scale=1.0))

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




