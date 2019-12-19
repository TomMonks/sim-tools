'''

Fixed budget alpirhtms

1. OCBA - optimal computing budget allocation
2. OCBA-m - optimal computering budget allocation top m.

'''

import numpy as np

from toy_models import BanditCasino, GaussianBandit, guassian_bandit_sequence

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
    '''
    Optimal Computer Budget Allocation (OCBA)

    Given a total replication budget T allocate
    replications across designs in order to maximise the 
    probability of correct selections.

    Assumes each system design has similar run time.

    '''
    def __init__(self, model, n_designs, budget, delta, n_0=5, min=True):
        '''
        Constructor method for Optimal Budget Computer Allocation

        Parameters:
        ---------

        model - object, simulation model that implements 
        interface action(design)

        n_designs - int, (k) the number of competing system designs 

        budget - int, (T) the total simulation budget available i.e. the 
        total number of replications available to allocate between systems

        delta - int, incremental budget to allocate.  Recommendation is
        > 5 and smaller than 10 percent of budget.  When simulation is expensive
        then this number could be set to 1.

        n_0 - int, the total number of initial replications.  Minimum allowed is 5
        (default=5)

        min - bool, True if minimisation; False if maximisation.  (default=True)

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

        self._allocations = np.zeros(n_designs, np.int32)
        self._means = np.zeros(n_designs, np.float64)
        self._vars = np.zeros(n_designs, np.float64)
        self._ratios = np.zeros(n_designs, np.float64)

        #used when calculating running standard deviation across designs
        #sum of squares
        self._sq = np.zeros(n_designs, np.float64)

        if min:
            self._negate = 1.0
        else:
            self._negage = -1.0

    def solve(self):
        '''
        run the ranking and selection procedure
        Vanilla OCBA 
        '''
        self._initialise()

        while self._allocations.sum() < self._T:
            new_allocations = self._allocate()
            
            for design in range(self._k):
                for replication in new_allocations[design]:
                    self._env.action(design)

        best = np.argmin(self._means)
        self._means *= self._negate
        return best


    def _initialise(self):
        '''
        For each design run n_0 initial replications
        '''
        for design in range(self._k):
            for replication in range(self._n_0):
                self._env.action(design)

    
    def _allocate(self):
        '''
        Allocate the incremental budget across 
        designs
        '''
        #total allocated + delta
        total_allocated = self._allocations.sum() + self._delta

        #get indicies of best and second best designs so far
        #note treated as minimisation problem.  Means are negated if maximisation
        ranks = get_ranks(self._means) 
        best_index, s_best_index = np.argpartition(ranks, 2)[:2]

        self._ratios[s_best_index] = 1.0

        #Part 1: Ratio N_i / N_s
        #all 'select' does is exclude best and second best from arraywise calcs
        select = [i for i in range(self._k) if i not in [best_index, s_best_index]]
        temp = self._means[best_index] - self._means[s_best_index] \
                / self._means[best_index] - self._means[select]
        self._ratios[select] = np.square(temp) * self._vars[select] / self._vars[s_best_index]

        #Part 2: N_b
        #exclude best
        select = [i for i in range(self._k) if i not in [best_index]]
        temp = (np.square(self._ratios[select]) / self._vars[select]).sum()
        self._ratios[best_index] = np.sqrt(self._vars[best_index] * temp)

        #got all of the ratios now...
        more_runs = np.full(self._k, 1, dtype=np.int8)

        more_alloc = True
        #additional_runs = np.fill(self._k, 1, dytpe=np.int16)

        while(more_alloc):
            ratio_s = (more_runs * self._ratios).sum()
            additional_runs = (total_allocated / (ratio_s * self._ratios)).astype(int)

            mask = additional_runs < self._allocations
            additional_runs[mask] = self._allocations[mask]
            more_runs[mask] = 0  # not sure I need this...
            if mask.sum() > 0: more_alloc = True

            if more_alloc:
                budget_remaining = total_allocated ## ?
                total_allocated -= (additional_runs * more_runs).sum()

        total_additional = additional_runs.sum()
        additional_runs[best_index] = total_allocated - total_additional

        return additional_runs - self._allocations


    def feedback(self, *args, **kwargs):
        '''
        Feedback from the environment
        Recieves a reward and updates understanding
        of an arm

        Parameters:
        ------
        *args -- list of argument
                 0  sender object
                 1. arm index to update
                 2. observation

        *kwargs -- dict of keyword arguments

        '''
        design_index = args[1]
        observation = args[2]
        self._allocations[design_index] +=1 
                
        #update running mean and standard deviation
        self._update_moments(design_index, observation)
        

        
    def _update_moments(self, design_index, observation):
        '''
        Updates the running average, var of the design

        Parameters:
        ------
        design_index -- int, index of the array to update

        observation -- float, observation recieved from the last replication
        '''
        n = self._allocations[design_index]
        current_mean = self._means[design_index]
        new_mean = ((n - 1) / float(n)) * current_mean + (1 / float(n)) * observation * self._negate

        if n > 1:
            self._sq[design_index] += (observation - current_mean) * (observation - new_mean)
            self._vars[design_index] = self._sq[design_index] / (n - 1)

        self._means[design_index] = new_mean




if __name__ == '__main__':
    designs = guassian_bandit_sequence(1, 11)
    
    environment = BanditCasino(designs)

    ocba = OCBA(environment, len(designs), 500, 10)

    results = ocba.solve()
    print(results)