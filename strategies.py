"""
This file defines strategies for bandits and ways to simulate them
"""
import numpy as np
import hashlib
import bandit_lp
import os
import markovianbandit

if not os.path.exists('computed_values/'):
    os.makedirs('computed_values/')

class AbstractStrategy:
    """
    Describes the Interface of a strategy
    """
    def next_y(self, state_x, N=None):
        """
        N is optional and only used for some strategies like FTVA
        """
        pass

    def hashname(self):
        """
        Empty hash if the class has nothing.
        """
        assert False, "abstract function not implemented"

    def __init__(self):
        self.X = None
        self.reward = None

class PriorityStrategy(AbstractStrategy):
    """
    A strategy that gives priority to some states
    """
    def __init__(self, order_of_states, alpha):
        """
        Initialize a priority strategy.
        
        Parameters:
        - order_of_states = priority list (order_of_sates[0] has highest priority)
        - alpha = budget
        """
        AbstractStrategy.__init__(self)
        self.S = len(order_of_states)
        assert np.all(np.sort(order_of_states) == np.arange(self.S)), "order_of_states should be a permutation of 0...(S-1)"
        self.order_of_states = order_of_states
        self.A = 2
        self.alpha = alpha

    def hashname(self):
        """
        Return a hash of the bandit
        """
        h = hashlib.new('sha256')
        h.update(b'priority')
        h.update(np.array(self.S).view())
        h.update(np.array(self.order_of_states).view())
        h.update(str(self.A).encode())
        h.update(str(self.alpha).encode())
        return h.hexdigest()[0:10]

    def next_y(self, state_x, N=None):
        """
        Compute the next Y as a function of "state_x" (according to the priority given)
        """
        assert len(state_x) == self.S, "size mismatch between state_x and the priority list"
        next_y = np.zeros(shape=(self.S, self.A))
        budget = self.alpha
        for s in self.order_of_states:
            next_y[s, 1] = min(state_x[s], budget)
            budget -= next_y[s, 1]
            next_y[s, 0] = state_x[s] - next_y[s, 1]
        return next_y

class LPupdateStrategy(AbstractStrategy):
    """
    return the "LP-update" policy for a budget alpha and a planing horizon "time"
    """
    def __init__(self, bandit: bandit_lp.BanditInstance, alpha, time):
        """
        Initialize the class with the bandit. 

        Inputs:
        - bandit = BanditInstance
        - alpha = budget (betwee 0 and 1)
        - time = rolling-horizon for the instance
        """
        AbstractStrategy.__init__(self)
        self.bandit = bandit
        self.alpha = alpha
        self.time = time
        self.computed_y = dict([])

    def hashname(self):
        """
        Return a hash of the bandit
        """
        h = hashlib.new('sha256')
        h.update(b'lp-update')
        h.update(self.bandit.hashname().encode())
        h.update(str(self.alpha).encode())
        h.update(str(self.time).encode())
        return h.hexdigest()[0:10]

    def next_y(self, state_x, N=None):
        """
        Solves the LP and returns the first value.

        The use of a dictionary fasten the computation for small N and small state-space.
        """
        if tuple(state_x) not in self.computed_y:
            y_0 = self.bandit.relaxed_lp_finite_time(self.alpha, state_x, self.time)[1][0]
            self.computed_y[tuple(state_x)] = y_0
            return y_0
        return self.computed_y[tuple(state_x)]

class LPPriorityStrategy(PriorityStrategy):
    """
    Class corresponding to the LP-priority strategy
    """
    def __init__(self, bandit : bandit_lp.BanditInstance, alpha):
        """
        Computes the LP-priority strategy for the specific instance of bandit. 
        """
        multiplier_alpha = bandit.relaxed_lp_average_reward(alpha)[3]
        q_table = np.copy(bandit.R)
        reward_plus_penalty = np.copy(bandit.R)
        reward_plus_penalty[:,1] -= multiplier_alpha
        for i in range(1000):
            q_table = reward_plus_penalty + bandit.P@np.max(q_table, 1)
        self.lp_index = q_table[:,1]-q_table[:,0]
        PriorityStrategy.__init__(self, np.flip(np.argsort(self.lp_index)), alpha)

class WhittleIndexStrategy(PriorityStrategy):
    """
    Class corresponding to the Whittle index strategy
    """
    def __init__(self, bandit : bandit_lp.BanditInstance, alpha):
        """
        Priority according to Whittle index for the specific instance of bandit. 
        """
        bandit_instance = markovianbandit.RestlessBandit(bandit.P, bandit.R)
        self.whittle_indices = bandit_instance.whittle_indices()
        PriorityStrategy.__init__(self, np.flip(np.argsort(self.whittle_indices)), alpha)


class FTVA_Strategy(AbstractStrategy):
    """
    Provide a way to implement the FTVA strategy of https://arxiv.org/pdf/2306.00196
    """
    def __init__(self, bandit: bandit_lp.BanditInstance, alpha):
        """
        The function needs to alpha
        """
        AbstractStrategy.__init__(self)
        self.bandit = bandit
        self.alpha = alpha
        _ , y_star, _, _ = bandit.relaxed_lp_average_reward(alpha)
        self.pi_star = y_star[:,1] / np.sum(y_star, 1) # optimal policy for the relaxed problem
        self.S = None
        self.S_virtual = None
        self.Y = None

    def hashname(self):
        """
        Return a hash of the bandit
        """
        h = hashlib.new('sha256')
        h.update(b'ftvl')
        h.update(self.bandit.hashname().encode())
        h.update(str(self.alpha).encode())
        return h.hexdigest()[0:10]

    def next_y(self, state_x, N=None):
        """
        !!!This returns Y but also simulate the next X!!!
        """
        assert isinstance(N, int), "this strategy is not defined for infinite N"
        if self.S is None or len(self.S) != N:
            #This means that we never called this function: we need to initialize
            self.S = np.zeros(N, dtype=int) # states of all bandits
            n=0
            for i in range(self.bandit.S):
                for j in range(int(N*state_x[i])):
                    self.S[n] = i
                    n+=1
            for n in range(n, N):
                self.S[n] = np.random.choice(len(state_x), p = state_x)
            self.S_virtual = np.copy(self.S)

        budget = int(self.alpha*N)

        A_virtual = np.array([np.random.rand() <= self.pi_star[self.S_virtual[i]] for i in range(N)], dtype=int)
        A = np.copy(A_virtual)
        truncate_to_budget(A, budget)

        self.Y = Y_from_S(self.S, A, self.bandit.S, 2)/N
        self.reward = np.tensordot(self.Y, self.bandit.R)

        for i in range(N):
            if self.S[i] == self.S_virtual[i] and A[i] == A_virtual[i]:
                self.S[i] = np.random.choice(self.bandit.S, p = self.bandit.P[self.S[i], A[i], :])
                self.S_virtual[i] = self.S[i]
            else:
                self.S[i] = np.random.choice(self.bandit.S, p = self.bandit.P[self.S[i], A[i], :])
                self.S_virtual[i] = np.random.choice(self.bandit.S, p = self.bandit.P[self.S_virtual[i], A_virtual[i], :])
        
        self.X = X_from_S(self.S, self.bandit.S)/N
        return self.Y

def truncate_to_budget(A, budget):
    """
    Truncate the vector of actions "A" so that sum(A) == budget
    """
    remaining_budget = budget - np.sum(A)
    for i, a in enumerate(A):
        if remaining_budget > 0 and a == 0:
            A[i] = 1
            remaining_budget -= 1
        elif remaining_budget < 0 and a == 1:
            A[i] = 0
            remaining_budget += 1
        
def X_from_S(S, number_of_states):
    """
    Computes the empirical measure X from the state 'S'
    """
    X = np.zeros(number_of_states)
    for s in S:
        X[s] += 1
    return X

def Y_from_S(S, A, number_of_states, number_of_actions):
    """
    Computes the empirical measure Y from the state-actions 'S,A'
    """
    Y = np.zeros(shape=(number_of_states, number_of_actions))
    for s, a in zip(S, A):
        Y[s, a] += 1
    return Y




def hashname(bandit, strategy, initial_state, N, time, seed):
    """
    Provide a name for the parameter of 'simulate'
    """
    h = hashlib.new('sha256')
    h.update(np.array(initial_state).view())
    return 'computed_values/{}_{}_{}_N{}_T{}_seed{}.npz'.format(bandit.hashname(), strategy.hashname(),
                                                        h.hexdigest()[0:10], N, time, seed)

def round_state_to_integer(vector, N):
    """
    Returns an array close to 'vector' so that array[i]*N is an integer
    """
    array = np.array([int(N*x_0_i)/N for x_0_i in vector])
    if np.sum(array) < 1:
        array[0] += 1- np.sum(array)
    else:
        for i in range(len(array)):
            array[i] = max(0, array[i] - np.sum(array) +1)
    return array

def simulate(bandit: bandit_lp.BanditInstance, strategy: AbstractStrategy, initial_state, N, time, verbose=False, seed=None):
    """
    This simulates a bandit with a given strategy. 

    Inputs:
    - bandit
    - strategy 
    - initial_state
    - N : can be finite or np.inf
    - time = time-horizon (integer)
    """
    filename = hashname(bandit, strategy, initial_state, N , time, seed)
    try:
        if seed is None:
            assert False
        file = np.load(filename)
        reward_values, x_values, y_values = file['arr_0'], file['arr_1'], file['arr_2']
        if np.max(np.abs(np.sum(x_values, 1)-1)) > 1e-6:
            print("There is a problem with the number of bandits:{}", np.max(np.abs(np.sum(x_values, 1)-1)))
            assert False
    except Exception as err:
        if verbose:
            print('we need to recompute', filename, 'because', err)
        np.random.seed(seed)
        state_x = round_state_to_integer(initial_state, N)
        x_values = np.zeros(shape=(time, bandit.S))
        y_values = np.zeros(shape=(time, bandit.S, bandit.A))
        reward_values = np.zeros(shape=time)
        cumulative_reward = 0
        for t in range(time):
            x_values[t] = state_x
            next_y = strategy.next_y(state_x, N)
            y_values[t] = next_y
            if strategy.X is None:
                next_x, reward = bandit.next_x_from_y(next_y, N)
            else:
                next_x = strategy.X
                reward = strategy.reward
            reward_values[t] = reward
            if verbose:
                print(state_x, reward, cumulative_reward)
            state_x = next_x
        if seed is not None:
            np.savez_compressed(filename, reward_values, x_values, y_values)
    return np.mean(reward_values), x_values, reward_values, y_values

