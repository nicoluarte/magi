import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
import networkx as nx


class AntColony:
    """
    Ant colony optimization algorithm
    finds the shortest distance in a 2d TSP
    """
    def __init__(self, mat, n_ants=100, n_iterations=1000, ph_evap=0.6, exp=1):
        # masks imposible transitions
        self.mat = np.ma.masked_less(np.asarray(mat), np.inf)
        # initialize the number of ants per iteration
        self.n_ants = n_ants
        # initialize the number of iterations
        self.n_iterations = n_iterations
        # initialize the rate of pheromone evaporation
        self.ph_evap = ph_evap
        # the best ant of each iteration, the one that is going
        # to deposit pheromone
        self.best_ant = None
        # data frame to save each ant path, on every iteration
        self.ants = pd.DataFrame(np.zeros((n_ants, n_iterations))).astype('object')
        # saves the cost of each path, on every iteration
        self.ants_cost = pd.DataFrame(np.zeros((n_ants, n_iterations)))
        # sets the initial pheromone quantity, higher values promote more
        # exploration, lower values exploitation
        self.ph_mat = (np.ones(self.mat.shape)*exp)*self.mat.mask
        # the transition probability matrix
        self.probability_mat = np.zeros(self.mat.shape)
        # not considered for the moment
        self.alpha = 1
        self.beta = 1

    def update_probability_matrix(self):
        """ updates the transition probability based on pheromone deposit"""
        node_prob = lambda x, y: x * np.divide(1, y, out=np.zeros_like(x), where=y != 0)
        # pheromone(of edge) * distance(of edge) /
        # the sum of all (pheromone(of edge) * distance(of edge))
        # it returns a probability distribution over possible neighbours
        # aux is the sum all over all posible neoghbours
        aux = np.sum(np.array(list(map(node_prob, self.ph_mat, self.mat))),
                     axis=1, keepdims=True)
        # here the actual update ocurrs
        # each row sums up to 1
        self.probability_mat = np.array(list(map(node_prob, self.ph_mat, self.mat))) / aux

    def pick_node(self):
        """ based on transition probability the ant chooses the next node"""
        # set current node as the starting node
        curr_node = 0
        # var to save the paths
        path = []
        # until it gets to the last node
        while curr_node != self.mat.data.shape[0]-1:
            # set the last step node in memory
            last_node = curr_node
            # choose a node to visit
            step = np.random.choice(self.probability_mat[curr_node,], size=1, replace=True,
                                    p=self.probability_mat[curr_node,])
            # select that node and "go" there
            curr_node = np.where(self.probability_mat[curr_node,] == step)[0][-1]
            path.append([last_node, curr_node])
        index = [tuple(l) for l in path]
        values = [self.mat.data[i] for i in index]
        cost = sum(values)
        return path, cost

    def generate_path(self):
        """ generates the path for each ant on each iteration"""
        self.update_probability_matrix()
        for iterations in range(self.n_iterations):
            for ant in range(self.n_ants):
                self.ants.iat[ant, iterations], \
                    self.ants_cost.iat[ant, iterations] = self.pick_node()
            # updates
            self.update_pheromone(iterations)
            self.update_probability_matrix()

    def update_pheromone(self, iterations):
        """ updates the pheromone matrix based on the best ant path"""
        # get best ant
        self.best_ant = self.ants_cost.idxmin(axis=0)[iterations]
        # update pheromone matrix
        # get the path tuples
        path_tuples = [tuple(l) for l in self.ants.iat[self.best_ant, iterations]]
        values = [self.mat.data[i] for i in path_tuples]
        delta_t = list(map(lambda x: 1/x, values))
        # deposit pheromone
        self.ph_mat *= (1 - self.ph_evap)
        for idx, deltas in zip(path_tuples, delta_t):
            self.ph_mat[idx] += deltas*self.ph_evap



## minitest
DIST_MAT = np.array([[np.inf, 5, np.inf, 5, np.inf, np.inf],
                     [5, np.inf, 3, 5, 3, np.inf],
                     [np.inf, 3, np.inf, 3, np.inf, np.inf],
                     [5, 5, 3, np.inf, np.inf, np.inf],
                     [np.inf, 3, np.inf, np.inf, np.inf, 3],
                     [np.inf, np.inf, np.inf, np.inf, 3, np.inf]])

ANT = AntColony(DIST_MAT, 5, 100, 0.01)

ANT.generate_path()
X, Y = np.where(ANT.ants_cost == ANT.ants_cost.min().min())
print(ANT.ants.iat[X[0],Y[0]])

def get_elem(elem):
    plt.plot(*zip(*elem))
    plt.gcf()
    plt.show()

plt.ion()
elem = ANT.ants[0][0]
plt.plot(*zip(*elem))
plt.show()


elem = ANT.ants[0][1]
plt.plot(*zip(*elem))
plt.show()

G2 = nx.from_numpy_matrix(DIST_MAT)
nx.draw_circular(G2)
plt.axis('equal')

plt.show()
axes = plt.gca()
axes.set_xlim(0,5)
axes.set_ylim(0,5)
X = []
Y = []
for iteration in range(0, ANT.ants.shape[1]-1):
    line, = axes.plot(X, Y, 'r-')
    elem = ANT.ants.loc[0, iteration]
    X = [x[0] for x in elem]
    Y = [y[1] for y in elem]
    points = list(zip(X,Y))
    plt.draw()
    plt.pause(1e-17)
    time.sleep(0.1)
