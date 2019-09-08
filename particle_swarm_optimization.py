import numpy as np
from matplotlib import pyplot as plt

class particle:
    """ initialize the swarm"""
    def __init__(self, particle_number, n_dimension, lower, upper, omega, phi_p, fun):
        self.particle_number = particle_number
        self.fun = fun
        self.n_dimension = n_dimension
        self.lower = lower
        self.upper = upper
        self.omega = omega
        self.phi_p = phi_p
        self.phi_g = 1 - phi_p
        """ initialize the particle's best known position to its initial position"""
        self.particle_position = np.random.uniform(low = self.lower,
                                                        high = self.upper,
                                                        size = (self.particle_number, self.n_dimension))
        """calculate the cost of initial positions"""
        self.particle_cost = [self.fun(particle) for particle in self.particle_position]
        """at first iteration the local best is equal to the starting point"""
        self.particle_best = self.particle_position
        """global best is equal to the lowest cost in particle initial position"""
        self.swarm_best = self.particle_position[np.argmin(self.particle_cost)]
        self.swarm_best_cost = self.fun(self.swarm_best)
        self.particle_velocity = np.random.uniform(low = -abs(self.upper - self.lower),
                                                   high = abs(self.upper - self.lower),
                                                   size = (self.particle_number, self.n_dimension))
    """update the cost of particle and swarm candidate solutions"""
    def update_particle_cost(self):
        self.particle_cost = [self.fun(particle) for particle in self.particle_position]
    def update_swarm_cost(self):
        self.swarm_best_cost = self.fun(self.swarm_best)
    """update swarm and particle best"""
    def update_particle_best(self):
        self.particle_best = [self.particle_position[i] \
                              if self.particle_cost[i] < self.fun(self.particle_best[i]) \
                              else self.particle_best[i] \
                              for i in range(self.particle_number)]
    def update_swarm_best(self):
        if min(self.particle_cost) < self.swarm_best_cost:
            self.swarm_best = self.particle_position[np.argmin(self.particle_cost)]
    """function to update each particle velocity"""
    def update_velocity(self):
        self.particle_velocity = [self.omega*self.particle_velocity[i] + \
                                  self.phi_p*np.random.uniform(size = self.n_dimension)* \
                                  (self.particle_best[i] - self.particle_position[i]) + \
                                  self.phi_g*np.random.uniform(size = self.n_dimension)* \
                                  (self.swarm_best - self.particle_position[i]) \
                                  for i in range(self.particle_number)]
    """function to update each particle position"""
    def update_position(self):
        self.particle_position = [self.particle_position[i] + self.particle_velocity[i] \
                                  for i in range(self.particle_number)]

"""solution is (3, 0.5)"""
def test_function(x):
    out = (1.5 - x[0] + x[0] * x[1])**2 + \
    (2.25 - x[0] + x[0] * x[1]**2)**2 + \
    (2.625 - x[0] + x[0] * x[1]**3)**2
    return out

def run_spo():
    p = particle(100, 2, -10, 10, 0.7, 0.4, test_function)
    plt.ion()
    fig = plt.gcf()
    fig.show()
    fig.canvas.draw()
    colors = np.random.rand(p.particle_number)
    for i in range(100):
        p.update_velocity()
        p.update_position()
        p.update_particle_cost()
        p.update_particle_best()
        p.update_swarm_best()
        p.update_swarm_cost()
        x = [p[0] for p in p.particle_position]
        y = [p[1] for p in p.particle_position]
        plt.scatter(x, y, s=p.particle_cost, c=colors, alpha = 0.5)
        plt.scatter([3, -2.8, -3.7, 3.5], [2, 3.13, -3.2, -1.8],
                    alpha = 0.3, marker = 'o', c='blue',
                    s=300,
                    linewidths=10)
        plt.xlim([-10,10])
        plt.ylim([-10,10])
        fig.canvas.draw()
        plt.clf()
