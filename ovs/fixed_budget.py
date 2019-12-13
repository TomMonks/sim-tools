'''

Fixed budget alpirhtms

1. OCBA - optimal computing budget allocation
2. OCBA-m - optimal computering budget allocation top m.

'''

import numpy as np

def ocba_m(dataset, k, allocations, T, delta, m):
    
    while allocations.sum() < T:
        
        #simulate systems using new allocation of budget
        reps = simulate(dataset, k, allocations) 
        
        #calculate sample means and standard errors
        means, ses = summary_statistics(reps, allocations)
        
        #calculate parameter c and deltas
        c = parameter_c(means, ses, k, m)
        deltas = means - c
        
        #allocate
        for i in range(delta):
            values = np.divide(allocations, np.square(np.divide(ses, deltas)))
            ranks = get_ranks(values)
            allocations[ranks.argmin()] += 1
            
    return means, ses, allocations


def get_ranks(array):
    """
    Returns a numpy array containing ranks of numbers within a input numpy array
    e.g. [3, 2, 1] returns [2, 1, 0]
    e.g. [3, 2, 1, 4] return [2, 1, 0, 3]
        
    Parameters:
    --------
    array - np.ndarray, numpy array (only tested with 1d)

    Returns:
    ---------
    np.ndarray, ranks
        
    """
    temp = array.argsort()
    ranks = np.empty_like(temp)
    ranks[temp] = np.arange(len(array))
    return ranks


class OCBA(object):
    def __init__(self, model, n_designs, budget, delta, n_0=5):
        '''
        Constructor method for Optimal Budget Computer Allocation

        Parameters:
        ---------

        model - object, simulation model that implements 
        method simulate(design, replications)

        n_designs - int, (k) the number of competing system designs 

        budget - int, (T) the total simulation budget available i.e. the 
        total number of replications available to allocate between systems

        delta - int, incremental budget to allocate.  Recommendation is
        > 5 and smaller than 10 percent of budget.  When simulation is expensive
        then this number could be set to 1.

        n_0 - int, the total number of initial replications.  Minimum allowed is 5
        (default=5)

        '''
        if n_0 < 5:
            raise ValueError('n_0 must be >= 5')

        if (budget - (n_designs * n_0)) % delta != 0:
            raise ValueError('(budget - (n_designs * n_0)) must be multiple of delta')

        model.register_observer(self)
        self._env = model
        self._k = n_designs
        self._T  = budget
        self._delta = delta
        self._n_0 = n_0

        self._actions = np.zeros(n_designs, np.int32)
        self._means = np.zeros(n_designs, np.float64)
        self._init_obs = np.zeros((n_designs, n_0), np.float64)

        #used when calculating running standard deviation across designs
        self._mu = np.zeros(n_designs, np.float64)
        self._sq = np.zeros(n_designs, np.float64)


    def solve(self):
        l = 0
        self._initialise()

        while self._actions.sum() < T:
            pass

    
    def _initialise(self):
        for design in range(self.k):
            for replication in self._n0:
                self._env.action(design)


    def _sequential_replication(self, systems):
        for design in systems:
            self._env.action(design)
        
    def feedback(self, *args, **kwargs):
            '''
        Feedback from the environment
        Recieves a reward and updates understanding
        of an arm

        Keyword arguments:
        ------
        *args -- list of argument
                 0  sender object
                 1. arm index to update
                 2. reward

        *kwards -- dict of keyword arguments:
                   None expected!

        '''
        design_index = args[1]
        reward = args[2]
        self._actions[design_index] +=1 #+= number of replicaions
        mu = self._mu[design_index]
        self._means[design_index] = self.updated_mean_estimate(design_index, reward)
        
        #probably should check what is happening here...
        mu_new = mu + (reward - mu) / self._actions[design_index]
        self._sq[design_index] += (reward - mu) * (reward - mu_new)
        mu = muNew
        #if self._r < self._n_0:
        #    self._init_obs[design_index][self._r] = reward

        

    def updated_mean_estimate(self, design_index, reward):
        '''
        Calculate the new running average of the design

        Keyword arguments:
        ------
        design_index -- int, index of the array to update
        reward -- float, reward recieved from the last action

        Returns:
        ------
        float, the new mean estimate for the selected arm
        '''
        n = self._actions[design_index]
        current_value = self._means[design_index]
        new_value = ((n - 1) / float(n)) * current_value + (1 / float(n)) * reward
        return new_value 

