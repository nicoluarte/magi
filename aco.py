import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from itertools import groupby
import networkx as nx
import time

class ant_colony:
    """ ant colony optimization algorithm
    n_ant = number of ants
    ph_evap = pheremone evaporation rate
    ph_deposit = pheremone deposit
    mat = matrix to optimize; 0 = not a possible neighbour
    ph_mat = the pheromone matrix"""
    def __init__(self, n_ants, n_iterations, ph_evap, mat, alpha = 1, beta = 1):
        self.best_ant = None
        self.n_ants = n_ants
        self.ants = pd.DataFrame(np.zeros((n_ants, n_iterations))).astype('object')
        self.ants_cost = pd.DataFrame(np.zeros((n_ants, n_iterations)))
        self.n_iterations = n_iterations
        self.mat = np.ma.masked_less(np.asarray(mat), np.inf)
        self.ph_mat = np.ones(self.mat.shape)*self.mat.mask
        self.probability_mat = np.zeros(self.mat.shape)
        # constants
        self.ph_evap = ph_evap
        self.alpha = alpha
        self.beta = beta

    def update_probability_matrix(self):
        # calculates the cost of each edge
        # x = pheremone of given edge
        # y = 1/distance
        p = lambda x, y: x * np.divide(1, y, out=np.zeros_like(x), where=y!=0)
        # pheromone(of edge) * distance(of edge) /
        # the sum of all (pheromone(of edge) * distance(of edge))
        # it returns a probability distribution over possible neighbours
        # aux is the sum all over all posible neoghbours
        aux = np.sum(np.array(list(map(p, self.ph_mat, self.mat))), axis = 1, keepdims = True)
        self.probability_mat = np.array(list(map(p, self.ph_mat, self.mat))) / aux

    def pick_node(self):
        # set current node as the starting node
        curr_node = 0
        # var to save the paths
        path = []
        no_retrace_mat = np.copy(self.probability_mat)
        while curr_node != self.mat.data.shape[0]-1:
            # set the last step node in memory
            last_node = curr_node
            # choose a node to visit
            step = np.random.choice(no_retrace_mat[curr_node,], size = 1, replace = True,
                                    p = no_retrace_mat[curr_node,])
            # select that node and "go" there
            curr_node = np.where(no_retrace_mat[curr_node,] == step)[0][-1]
            path.append([last_node, curr_node])
        # clean path from duplicates
        for i in range(len(path)-1):
            if np.array_equal(path[i], np.flip(path[i+1], axis = 0)):
                path[i+1] = None
        path = [e for e in path if e != None]
        path = [x[0] for x in groupby(path)]
        index = [tuple(l) for l in path]
        values = [self.mat.data[i] for i in index]
        cost = sum(values)
        return path, cost

    def generate_path(self):
        self.update_probability_matrix()
        for iterations in range(self.n_iterations):
            for ant in range(self.n_ants):
                self.ants.iat[ant, iterations], self.ants_cost.iat[ant, iterations] = self.pick_node()
            # updates
            self.update_pheromone(iterations)
            self.update_probability_matrix()

    def update_pheromone(self, iterations):
        # get best ant
        self.best_ant = self.ants_cost.idxmin(axis = 0)[iterations]
        # update pheromone matrix
        # get the path tuples
        tt = [tuple(l) for l in self.ants.iat[self.best_ant, iterations]]
        values = [self.mat.data[i] for i in tt]
        delta_t = list(map(lambda x: 1/x, values))
        # deposit pheromone
        self.ph_mat *= (1 - self.ph_evap)
        for idx, deltas in zip(tt, delta_t):
            self.ph_mat[idx] += deltas*self.ph_evap


## minitest
dist_mat = np.array([[np.inf, 5, np.inf, 5, np.inf, np.inf],
                    [5, np.inf, 3, 5, 3, np.inf],
                    [np.inf, 3, np.inf, 3, np.inf, np.inf],
                    [5, 5, 3, np.inf, np.inf, np.inf],
                    [np.inf, 3, np.inf, np.inf, np.inf, 3],
                    [np.inf, np.inf, np.inf, np.inf, 3, np.inf]])

ant = ant_colony(5, 100, 0.01, dist_mat)

ant.generate_path()
x, y = np.where(ant.ants_cost == ant.ants_cost.min().min())
print(ant.ants.iat[x[0],y[0]])
