# -*- coding: utf-8 -*-
"""
Created on Tue Jan 18 14:24:18 2022

@author: Sarah Li
"""
import numpy as np
import pickle


def augment_graph(flight, full_graph, flat_graph, fl_min=0, fl_max=500):
    # create edge list
    edges = []
    
    flight_max = 500 # maximum flight_level
    flight_min = 0 # minimum flight_level
    # fl_num = int((flight_max -flight_min)/50 + 1)
    # flight_levels = [i*50 + flight_min for i in range(fl_num)]


    for (wp1, wp2) in zip(flight[:-1], flight[1:]):
        n1 = wp1[1]
        n2 = wp2[1]
        fl_1 = wp1[4]
        fl_2 = wp2[4]
        if n1 in flat_graph.nodes() and n2 in flat_graph.nodes():
            o_name = n1 + '_'+str(fl_1)
            found_fl = False
            for d_fl in [-50, 0, 50]:
                targ_fl = d_fl+fl_1
                if targ_fl >= fl_min and targ_fl <= fl_max:
                    d_name = n2 + '_'+str(targ_fl)
                    edges.append((o_name, d_name))
                    found_fl = found_fl or (targ_fl == fl_2) 
                
            if not found_fl:
                print(f' target altitude {fl_2} not added for origin: {n1}, {fl_1} and destination {n2}, {fl_2}')
    full_graph.add_edges_from(edges)
    
    
def find_smallest_ind(num_list, thres):
    """ Find the smallest index i where for all j < i, x_j <= thresh
    

    Parameters
    ----------
    num_list : list
        ordered list of numbers x_1...x_N
    thres : float
        cutoff value.

    Returns
    -------
    None.

    """
    i = 0
    while i < len(num_list) and num_list[i] < thres:
        i +=1
    return i # num_list[:i], num_list[i:]


def get_file(file_name):
    obj_file = open(file_name, 'rb')
    obj = pickle.load(obj_file)
    obj_file.close()
    return obj

def density_2_expected_traj(state_action, ordered_nodes):
    S, A, T = state_action.shape
    density = np.sum(state_action, axis=1)
    trajectory = []
    for t in range(T):
        max_state = np.argmax(density[:, t])
        trajectory.append(ordered_nodes[max_state])
    return trajectory

def compute_expected_traj(o_state, policy, graph, T, A, ordered_nodes):
    # given policy over a graph and the original ordered nodes, return a list
    # of node names corresponding the expected trajectory under policy, 
    # starting at o_state. o_state should be name of origin state.
    # compute expected trajectory starting from origin
    trajectory = [o_state]
    ordered_nodes = list(graph.nodes())
    for t in range(T-1):
        # print(f' time step {t}')
        s_ind = ordered_nodes.index(trajectory[-1])
        neighbors_list = list(graph.neighbors(trajectory[-1]))
        # print(f' {s_ind} {t}')
        # print(f' next state = {trajectory[-1]}')
        next_state_ind = int(policy[s_ind, t])
        if next_state_ind == A - 1:
            next_state = trajectory[-1]
        else:
            # print(f' neighbors ind {next_state_ind}/{len(neighbors_list)}')
            next_state = neighbors_list[next_state_ind] 
        trajectory.append(next_state)
        
    return trajectory 

def total_density(x, graph, ordered_nodes):
    # given x, a state-action density over subgraph graph, compute the 
    # density over the original ordered nodes. 
    # x should have shape (S, A, T), where S matches the number of nodes in 
    # graph.
    sub_density = np.sum(x, axis=1)
    _, T = sub_density.shape
    if np.sum(sub_density, axis=0).all() != np.ones(T).all():
        print('util.total_density: input density != 1 at each time.')
    sub_nodes = list(graph.nodes())
    density_ind = 0
    density = np.zeros((len(ordered_nodes), T))
    for node in sub_nodes:
        final_ind = ordered_nodes.index(node)
        density[final_ind] = sub_density[density_ind]
        density_ind += 1
    return density

def subgraph(states, flight, graph, depth=2):
    nodes = states.copy()
    for d in range(depth):
        neighbors = []
        for s in nodes:
            if s not in graph:
                states.remove(s)
                sep_strs = s.split('_')
                print(f' flight to remove is {(sep_strs[0], int(sep_strs[1]))}')
                for f in flight:
                    if f[1] == sep_strs[0] and f[2] == int(sep_strs[1]):
                        flight.remove(f)
            else:
                for s_neighbor in graph.neighbors(s):
                    if s_neighbor not in states:
                        neighbors.append(s_neighbor)
        # print(f' found neighbors {len(neighbors)}')
        states = states + neighbors.copy() # add neighbors to relevant states list
        nodes = neighbors.copy() # choose all neighbors and iterate again
    sub_graph = graph.subgraph(states)
    return sub_graph

def policy_to_traj(sub_graph, o_state, policy, T, A, flight):
    ordered_nodes = list(sub_graph.nodes())
    # o_ind = ordered_nodes.index(o_state)
    # print(f'total number of steps to get to get to destination is {V[o_ind, 0]}')
    
    # generate way points
    waypoint_seq = compute_expected_traj(
        o_state, policy, sub_graph, T, A, ordered_nodes)
    expected_traj = []
    for wp_ind in range(len(flight)):
        wp = waypoint_seq[wp_ind].split('_')
        expected_traj.append((wp[0], wp[1], flight[wp_ind][2]))
    return expected_traj

def insert_density(net_density, flight, flight_density, min_time, t_int):
    _, T = flight_density.shape
    # print(f'total density shape {net_density.shape}')
    for t_ind in range(T):
        # print(f'time is {flight[t_ind][-1]}')
        
        t = int((flight[t_ind][-1] - min_time) / t_int)
        # print(f't is {t}')
        net_density[:, t] += flight_density[:, t_ind]
    return net_density
        