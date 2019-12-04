"""
Template for implementing QLearner  (c) 2015 Tucker Balch

Code written by Karan Achtani in CS 4646 (ML4T)
and modified for the purposes of this project.
"""

import numpy as np
import random as rand


class QLearner(object):
    def __init__(self,
                 num_states=100,
                 num_actions=4,
                 alpha=0.2,
                 gamma=0.9,
                 rar=0.8,
                 radr=0.80,
                 verbose=False):

        self.verbose = verbose
        self.num_actions = num_actions
        self.alpha = alpha
        self.gamma = gamma
        self.rar = rar
        self.radr = radr
        self.s = 0
        self.a = 0
        self.Q = np.empty((num_states,num_actions))
        self.Q[:,:] = 0.0
        np.random.seed(0)

    def querysetstate(self, s):
        """
        @summary: Update the state without updating the Q-table
        @param s: The new state
        @returns: The selected action
        """
        self.s = s
        random = np.random.rand(1)
        if random < self.rar:
            action = rand.randint(0, self.num_actions - 1)
        else:
            action = np.argmax(self.Q[s])
        if self.verbose: print("s =", s, "a =", action)
        self.a = action
        return action

    def query(self, s_prime, r):
        """
        @summary: Update the Q table and return an action
        @param s_prime: The new state
        @param r: The ne state
        @returns: The selected action
        """
        random = np.random.rand(1)
        if random < self.rar:
            action = rand.randint(0, self.num_actions - 1)
            self.rar *= self.radr
        else:
            action = np.argmax(self.Q[s_prime])


        self.Q[self.s, self.a] = (1 - self.alpha)\
                                 * self.Q[self.s, self.a]\
                                 + self.alpha\
                                   * (r + self.gamma * self.Q[s_prime, action])


        self.s = s_prime
        self.a = action

        if self.verbose: print("s =", s_prime, "a =", action, "r =", r)

        return action
