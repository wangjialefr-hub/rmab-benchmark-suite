"""
This file defines the class "BanditInstance", used to define bandit and solve the corresponding LPs
"""
import hashlib
import pulp
import numpy as np

class BanditInstance:
    """
    This class represents the parameter of a restless bandit
    """
    def __init__(self, P, R):
        """
        Initialize a model from the matrices P and R
        """
        assert len(R.shape) and len(P.shape)==3, "R should be SxA and P should be SxAxS"
        self.S, self.A = R.shape
        assert P.shape == (self.S, self.A, self.S), "R should be SxA and P should be SxAxS"
        self.P = P
        self.R = R
        # to store values for a given alpha
        self.alpha, self.x_star, self.g_star, self.multiplier_x, self.y_star = None, None, None, None, None
        
    def hashname(self):
        """
        Return a hash of the bandit
        """
        h = hashlib.new('sha256')
        h.update(self.P.view())
        h.update(self.R.view())
        return h.hexdigest()[0:10]

    def print_latex(self):
        """
        Print the transition matrix and reward vector
        """
        for action in range(self.A):
            print("\\begin{align*}\n    P^{",action,"}=\\left(\n    \\begin{array}{", end='')
            for i in range(self.P.shape[-1]):
                print('c', end='')
            print('}')
            for line in self.P[:, action, :]:
                print('       ', end='')
                for i in line:
                    print(float_to_str(i), end=' &')
                print('\b\\\\')
            print("    \\end{array}\\right)\n\\end{align*}")
            print("\nR^{", action,'} =', end=' ')
            for i in self.R[:, action]:
                print(float_to_str(i), end=', ')
            print('\b\n')

    def next_x_from_y(self, Y, N):
        """
        Simulate the stochastic system with N arms or the N=inf system.

        Inputs:
        - Y = S x 2 array
        - N = np.inf or integer. 
        
        Return (x, r), where:
        - X is the next state (array of size S)
        - r is the total reward
        """
        assert N == np.inf or isinstance(N, int), "N should be np.inf or an integer"
        assert self.A == 2, "only implemented for two actions because of rounding"
        reward = np.tensordot(Y, self.R)
        if N == np.inf:
            next_x = np.tensordot(Y, self.P)
        else:
            next_x = np.zeros(self.S)
            for s in range(self.S): # the main part is to treat the rounding problem.
                int_y_s_1 = int(np.floor(N*Y[s, 1])+1e-6) # we add 1e-6 to avoid rounding errors
                int_y_s_0 = int(np.round(N*(Y[s,0]+Y[s, 1])-int_y_s_1 ))
                int_y_s = [int_y_s_0, int_y_s_1]
                for a in range(self.A):
                    next_x += np.random.multinomial(int_y_s[a], self.P[s, a, :])/N
        return next_x, reward

    def relaxed_lp_average_reward(self, alpha):
        """
        Provides the solution of the infinite-horizon LP

        Inputs: 
        - alpha = resource constraint (for now, we restrict our self to two action bandits)
        
        Outputs: (gain, y_star, multipliers)
        """
        assert self.A == 2, "this is only implemented for two actions"
        actions = range(0, self.A)
        states = range(0, self.S)
        prob = pulp.LpProblem("LP1", pulp.LpMaximize)
        variables = pulp.LpVariable.dicts("Y",(states, actions),lowBound=0., upBound=1.)
        # resource constraints
        prob += pulp.lpSum([variables[s][1] for s in states]) == alpha
        # Markov state evolution
        for s in states:
            prob += pulp.lpSum(variables[s][a] for a in actions) == \
                pulp.lpSum([variables[ss][a]*self.P[ss,a,s] for a in actions for ss 
                                                                          in states])
        # initial condition is replaced by "MUST SUM TO 1"
        prob += pulp.lpSum(variables[s][a] for a in actions for s in states) == 1

        # objective    
        prob += pulp.lpSum([variables[s][a]*self.R[s, a] for a in actions for s in states])

        prob.solve(pulp.PULP_CBC_CMD(msg=0))

        gain = pulp.value(prob.objective)

        y_star = np.zeros((self.S, self.A))
        for s in states:
            for a in actions:
                y_star[s,a] = variables[s][a].varValue

        multipliers = []
        for (id, c) in prob.constraints.items():
            if id == '_C1':
                multiplier_alpha = c.pi
            else:
                multipliers.append(c.pi)
        multipliers = np.array(multipliers[:-1])
        return gain, y_star, multipliers, multiplier_alpha

    def relaxed_lp_finite_time(self, alpha, x_init, time, compute_with_rotated_cost=False):
        """
        Provides the solution of the infinite-horizon LP

        Inputs: 
        - alpha = resource constraint (for now, we restrict our self to two action bandits)
        
        Outputs: (gain, y_star, multipliers)
        """
        assert self.A == 2, "this is only implemented for two actions"
        actions = range(0, self.A)
        states = range(0, self.S)
        times = range(0, time)
        prob = pulp.LpProblem("LP1", pulp.LpMaximize)
        variables = pulp.LpVariable.dicts("Y",(times, states, actions),lowBound=0., upBound=1.)
        for t in times[:-1]:
            # resource constraints
            prob += pulp.lpSum([variables[t][s][1] for s in states]) == alpha
            # Markov state evolution
            for s in states:
                prob += pulp.lpSum(variables[t+1][s][a] for a in actions) == \
                    pulp.lpSum([variables[t][ss][a]*self.P[ss,a,s] for a in actions for ss 
                                                                              in states])
        prob += pulp.lpSum([variables[time-1][s][1] for s in states]) == alpha
                                                                              
        # initial condition
        for s in states:
            prob += pulp.lpSum([variables[0][s][a] for a in actions]) == x_init[s]

        # objective
        if compute_with_rotated_cost:
            _, _, multipliers, _ = self.relaxed_lp_average_reward(alpha)
            prob += pulp.lpSum([variables[t][s][a]*self.R[s, a] for a in actions for s in states for t in times])  - pulp.lpSum(
                    [variables[0][s][a]*multipliers[s] for a in actions for s in states]
                )+ pulp.lpSum(
                    [variables[time-1][s][a]*self.P[s,a,ss]*multipliers[ss] for a in actions for ss in states for s in states]
                )

        else:
            prob += pulp.lpSum([variables[t][s][a]*self.R[s, a] for a in actions for s in states for t in times])

        prob.solve(pulp.PULP_CBC_CMD(msg=0))

        gain = pulp.value(prob.objective)

        y_star = np.zeros((time, self.S, self.A))
        for t in times:
            for s in states:
                for a in actions:
                    y_star[t,s,a] = variables[t][s][a].varValue

        return gain, y_star

    def compute_and_store_lp_upper_bound(self, alpha, verbose=False):
        """
        Compute the LP average upper bound and store it
        """
        if self.alpha is None or np.abs(self.alpha - alpha) > 1e-6:
            if verbose:
                print("first time function called...")
            self.alpha = alpha
            self.g_star, self.y_star, self.multiplier_x, _ = self.relaxed_lp_average_reward(self.alpha)
            self.x_star = np.sum(self.y_star, 1)

    def rotated_cost(self, y, verbose=False):
        """
        Return the rotated cost, equal to g^*-r(y) + lambda np.sum(P@Y, 1) - np.sum(Y,1)
        """
        if len(np.array(y).shape) == 2:
            self.compute_and_store_lp_upper_bound(np.sum(y, 0)[1], verbose)
            return self.g_star - np.tensordot(y, self.R) - np.dot(self.multiplier_x, np.tensordot(y, self.P) - np.sum(y, 1))
        assert len(np.array(y).shape)==3, "y must be of size SxA (or TxSxA)"
        return np.array([self.rotated_cost(y_t) for y_t in y])

    def distance_to_x_star(self, x=None, alpha=None, y=None, verbose=False):
        """
        Return the distance to x_star

        Inputs: 
        - x, alpha where x is S or TxS vector
        - y, alpha=None where y is SxA or TxSxA vector 
        """
        if x is None:
            x = np.sum(y, len(y.shape)-1)
        if y is not None: 
            if len(y.shape) == 3:
                alpha = np.sum(y[0,:,1])
            else:
                alpha = np.sum(y[:,1])
        if len(np.array(x).shape) == 1:
            self.compute_and_store_lp_upper_bound(alpha, verbose)
            return np.sum(np.abs(x - self.x_star))
        assert len(np.array(x).shape)==2, "y must be of size SxA (or TxSxA)"
        return np.array([self.distance_to_x_star(x_t, alpha) for x_t in x])

class BanditCounterExample(BanditInstance):
    """
    This is the counter-example from Hong et al's paper https://arxiv.org/abs/2306.00196
    """
    def __init__(self):
        PSR = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
        PSL = [1.0, 1.0, 0.48, 0.47, 0.46, 0.45, 0.44, 0.43]
        P0 = np.zeros([8, 8])
        P1 = np.zeros([8, 8])

        P0[0,0] = PSL[0]
        P1[0, 1] = PSR[0]
        P1[0, 0] = 1- PSR[0]
        for i in range(1, 8):
            if i < 4:
                P0[i, i-1] = PSL[i]
                P0[i,i] = 1 - PSL[i]
                P1[i, (i + 1)%8] = PSR[i]
                P1[i, i] = 1 - PSR[i]
            else:
                P0[i, (i + 1)%8] = PSR[i]
                P0[i, i] =1 - PSR[i]
                P1[i, i - 1] = PSL[i]
                P1[i, i] = 1 - PSL[i]

        R0 = np.zeros(8)
        R0[7] = PSR[7]
        R1 = np.zeros(8)

        P = np.zeros(shape=(8, 2, 8))
        R = np.zeros(shape=(8, 2))
        P[:,0,:] = P0
        P[:,1,:] = P1
        R[:,0] = R0
        R[:,1] = R1
        BanditInstance.__init__(self, P, R)

class BanditCounterExampleYan1(BanditInstance):
    """
    First example of Section E.2 of https://arxiv.org/pdf/2012.09064
    (3 dimensional with cycle of length 2)
    """
    def __init__(self):
        P = np.zeros((3,2,3))
        P[:,0,:] = [[0.5214073, 0.40392496, 0.07466774],
                    [0.0158415, 0.21455666, 0.76960184],
                    [0.53722329, 0.37651148, 0.08626522]]
        P[:,1,:] = [[0.24639364, 0.23402385, 0.51958251],
                    [0.49681581, 0.49509821, 0.00808597],
                    [0.37826553, 0.15469252, 0.46704195]]
        R = np.zeros((3,2))
        R[:, 1] = [0.72232506, 0.18844869, 0.25752477]
        self.alpha = 0.4
        BanditInstance.__init__(self, P, R)

class BanditCounterExampleYan2(BanditInstance):
    """
    Second example of Section E.2 of https://arxiv.org/pdf/2012.09064
    (3 dimensional with cycle of length 2)
    """
    def __init__(self):
        P = np.zeros((3,2,3))
        P[:,0,:] = [[0.02232142, 0.10229283, 0.87538575],
                    [0.03426605, 0.17175704, 0.79397691],
                    [0.52324756, 0.45523298, 0.02151947]]
        P[:,1,:] = [[0.14874601, 0.30435809, 0.54689589],
                    [0.56845754, 0.41117331, 0.02036915],
                    [0.25265570, 0.27310439, 0.47423991]]
        R = np.zeros((3,2))
        R[:, 1] = [0.37401552, 0.11740814, 0.07866135]
        self.alpha = 0.4
        BanditInstance.__init__(self, P, R)




class BanditRandom(BanditInstance):
    """
    This generates a random bandit
    """
    def __init__(self, number_of_states, number_of_actions=2, seed=None):
        np.random.seed(seed)
        P = np.random.exponential(size=(number_of_states, number_of_actions, number_of_states))
        R = np.random.exponential(size=(number_of_states, number_of_actions))
        for s in range(number_of_states):
            for a in range(number_of_actions):
                P[s, a] /= sum(P[s,a])
        BanditInstance.__init__(self, P, R)

def is_almost_integer(a, precision=1e-8):
    """
    Return true if a is almost an integer
    """
    return np.abs(a - int(a)) <= precision
def float_to_str(a):
    """
    Pretty print of float
    """
    if np.abs(a) <= 1e-8: # we do not print zeros.
        return ""
    if is_almost_integer(a):
        return str(int(a))
    else:
        if is_almost_integer(10*a):
            return '{:.1f}'.format(a)
        elif is_almost_integer(100*a):
            return '{:.2f}'.format(a)
    return '{:.3f}'.format(a)

