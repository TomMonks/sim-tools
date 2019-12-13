'''
Contains algorithms for Optimisation via Simulation
with a fixed budget (fixed total number of replications)

1. OCBA - optimal computing budget allocation
2. OCBA-m - optimal computering budget allocation top m.
3. KN - KN (Kim-Nelson) sequential Ranking and Selection Algorithm

'''

import numpy as np

from toy_models import BanditCasino, GaussianBandit, guassian_bandit_sequence

class KN(object):
    def __init__(self, model, n_designs, delta, alpha=0.05, n_0=2):
        '''
        Constructor method for KN  Ranking and Selection Procedure.

        This works well for up to 20 competing designs

        Parameters:
        ----------

        model - object, simulation model that implements 
        method simulate(design, replications)

        n_designs - int, the number of designs (k)

        delta - float, the indifference zone

        alpha - float, between 0 and 1.  Used to calculate the 1-alpha 
        probability of correct selection

        n_0 - int, the number of initial observations (default = 2)

        '''

        model.register_observer(self)
        self._env = model
        #self._total_rounds = budget
        self._total_reward = 0
        self._current_round = 0
        self._actions = np.zeros(n_designs, np.int32)
        self._means = np.zeros(n_designs, np.float64)
        self._init_obs = np.zeros((n_designs, n_0), np.float64)
        
        self._k = n_designs
        self._contenders = np.arange(self._k)
        self._delta = delta
        self._alpha = alpha
        self._n_0 = n_0
        self._r = 0

        self._eta = 0.5 * (np.power((2 * alpha) / (self._k - 1), -2/(self._n_0-1)) - 1)
    
    def solve(self):
        self._initialisation()

        while not self._stopping():
            self._screening()
            self._sequential_replication()

        return self._contenders

    def _initialisation(self):  
        '''
        Initialise KN
        
        Estimate initial sample means and variances of the differences

        '''      
        self._h_squared = 2 * self._eta * (self._n_0 - 1)
        
        for observation in range(self._n_0):
            self._sequential_replication()
                
        self._variance_of_diffs = self._calc_variance_of_differences()


    def _sequential_replication(self):
        for design in self._contenders:
            self._env.action(design)
        
        self._r += 1
        print(self._r)


    def _calc_variance_of_differences(self):
        pairwise_diffs = self._init_obs[:, None] - self._init_obs
        variance_of_diffs = pairwise_diffs.var(axis=-1, ddof=1)
        #flattens array and drops differences with same design
        return variance_of_diffs[~np.eye(variance_of_diffs.shape[0],dtype=bool)]

    def _screening(self):
        self._contenders_old = np.asarray(self._contenders).copy()
        contenders_mask = np.full(self._contenders.shape, True, dtype=bool)
        #terrible way to code this...!

        #designs in contention for this round
        for i in range(len(self._contenders_old)):
            for l in range(len(self._contenders_old)):
                if i != l:
                    design_i, design_l = self._contenders_old[i], self._contenders_old[l]
                    if not self._design_still_in_contention(i, l):
                        contenders_mask[i] = False
                        break
                

        self._contenders = self._contenders[contenders_mask]


    def _design_still_in_contention(self, design_i, design_l):
        w_il = self._limit_to_distance_from_sample_means(design_i, design_l)

        mean_i = self._means[design_i]
        mean_l = self._means[design_l]

        return mean_i >= mean_l - w_il

    def _limit_to_distance_from_sample_means(self, design_i, design_l):
        '''
        calculates W_li(r), 
        which determines how far the sample mean from system i can 
        drop below the sample means of system l without being eliminated
        '''
        index = design_i * (self._k - 1) + (design_l - 1)

        w_il = (self._delta / (2 * self._r)) \
            * (((self._h_squared * self._variance_of_diffs[index]) / self._delta**2) - self._r)
        

        self

        return w_il

    def _stopping(self):
        '''
        If |I| = 1 then stop (and select the system whose index is in I)
        else take one further sample from each system

        '''
        if len(self._contenders) == 1:
            return True
        
        return False



    def update(self):
        pass



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
        self._means[design_index] = self.updated_mean_estimate(design_index, reward)
        if self._r < self._n_0:
            self._init_obs[design_index][self._r] = reward
        #calculate running standard deviation.
        

        #UCB specific to remove...
        #first run through divides by zero.  In numpy this operation yields inf.
        #the with np.errstate() call/context avoid warning user of the operation 
        #with np.errstate(divide='ignore', invalid='ignore'):
        #    deltas = np.sqrt(3/2 * (np.log(self._current_round + 1) / self._actions))
        
        #self._upper_bounds = self._means + deltas
        

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








if __name__ == '__main__':
    designs = guassian_bandit_sequence(1, 11)
    
    environment = BanditCasino(designs)

    kn = KN(environment, len(designs), 
            delta=0.05, alpha=0.1, n_0=2)

    results = kn.solve()
    print(results)