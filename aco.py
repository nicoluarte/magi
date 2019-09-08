import numpy as np

class ant_colony:
    """ ant colony optimization algorithm
    n_ant = number of ants
    ph_evap = pheremone evaporation rate
    ph_deposit = pheremone deposit
    mat = matrix to optimize; 0 = not a possible neighbour
    ph_mat = the pheromone matrix"""
    def __init__(self, n_ants, ph_evap, ph_deposit, mat):
        self.n_ants = n_ants
        self.n_ants = n_ants
        self.ph_evap = ph_evap
        self.ph_deposit = ph_deposit
        self.mat = mat
        self.ph_mat = mat
        self.cost_mat = None

    def update_cost_matrix(self):
        # calculates the cost of each edge
        # x = pheremone of given edge
        # y = 1/distance
        p = lambda x, y: x * np.divide(1, y, out=np.zeros_like(x), where=y!=0)
        # pheromone(of edge) * distance(of edge) /
        # the sum of all (pheromone(of edge) * distance(of edge))
        # it returns a probability distribution over possible neighbours
        self.cost_mat = np.array(list(map(p, self.ph_mat, self.mat)))
        aux = np.sum(np.array(list(map(p, self.ph_mat, self.mat))), axis = 1, keepdims = True)


## minitest
ant = ant_colony(0, 0, 0, distances)

