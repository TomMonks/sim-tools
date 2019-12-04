'''
Contains algorithms for Optimisation via Simulation
with a fixed budget (fixed total number of replications)

1. OCBA - optimal computing budget allocation
2. OCBA-m - optimal computering budget allocation top m.
3. KNPlusPlus - KN++ sequential Ranking and Selection Algorithm

'''

import numpy as np

class KNPlusPlus(object):
    def __init__(self, model, n_designs, delta, alpha=0.05, n_0=2, m_0='auto'):
        '''
        Constructor method for KN++ Ranking and Selection Procedure

        Parameters:
        ----------

        model - object, simulation model that implements 
        method simulate(design, replications)

        n_designs - int, the number of designs (k)

        delta - float, the indifference zone

        alpha - float, between 0 and 1.  Used to calculate the 1-alpha 
        probability of correct selection

        n_0 - int, the number of initial replications (default = 2)

        m_0 - int or str, the batch size where m_0 < n_0.  If integer value
        used then this is set batch size used.  If 'auto' is used then m_0 is set
        to n_0 - 1.  (default='auto')
        '''

        model.register_observer(self)
        #self._validate_budget(budget)
        self._env = model
        #self._total_rounds = budget
        self._total_reward = 0
        self._current_round = 0
        self._actions = np.zeros(n_designs, np.int32)
        self._means = np.zeros(n_designs, np.float64)
        
        self._k = n_designs
        self._delta = delta
        self._alpha = alpha
        self._n_0 = n_0

        if m_0 != 'auto'
            if m_0 >= n_0 or m_0 =< 0:
                raise ValueError('m_0 must be < n_0 and m_0 > 0')

        if m_0 == 'auto':
            self._m_0 = n_0 - 1
        else:
            self._m_0 = m_0

        self._r_0 = n_0
    
        eta = 0.5 * (np.power(2 * (1 -np.power(1- alpha, 1/(self._k-1))), -2/f) - 1)

    def initialisation(self):
        designs = [for i in range(self._k)]
        self._model.simulate(candidates=designs, 
                             replications=self.n_0)



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
        self._total_reward += reward
        self._actions[design_index] +=1 #+= number of replicaions
        self._means[design_index] = self.updated_mean_estimate(arm_index, reward)
        

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



class UpperConfidenceBound(object):
    
    def __init__(self, budget, environment):
        '''
        Constructor method
        '''
        environment.register_observer(self)
        self._validate_budget(budget)
        self._env = environment
        self._total_rounds = budget
        self._total_reward = 0
        self._current_round = 0
        self._actions = np.zeros(environment.number_of_arms, np.int32)
        self._means = np.zeros(environment.number_of_arms, np.float64)
        self._upper_bounds = np.zeros(environment.number_of_arms, np.float64)
    
        
    def _validate_budget(self, budget):
        if budget < 0:
            msg = 'budget argument must be a int > 0'
            raise ValueError(msg)
            
    def reset(self):
        self._total_reward = 0
        self._current_round = 0
        self._actions = np.zeros(self._env.number_of_arms, np.int32)
        self._means = np.zeros(self._env.number_of_arms, np.float64)
        self._upper_bounds = np.zeros(self._env.number_of_arms, np.float64)

    def _get_total_reward(self):
        return self._total_reward

    def _get_action_history(self):
        return self._actions
    
    def _get_best_arm(self):
        '''
        Return the index of the arm 
        with the highest expected value

        Returns:
        ------
        int, Index of the best arm
        '''
        return np.argmax(self._means)
    
    def solve(self):
        '''
        Run the epsilon greedy algorithm in the 
        environment to find the best arm 
        '''
        for i in range(self._total_rounds):
            
            max_upper_bound_index = np.argmax(self._upper_bounds)
            self._env.action(max_upper_bound_index)            
            
            self._current_round += 1
    
        
    
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
        arm_index = args[1]
        reward = args[2]
        self._total_reward += reward
        self._actions[arm_index] +=1
        self._means[arm_index] = self.updated_reward_estimate(arm_index, reward)
        
        #first run through divides by zero.  In numpy this operation yields inf.
        #the with np.errstate() call/context avoid warning user of the operation 
        with np.errstate(divide='ignore', invalid='ignore'):
            deltas = np.sqrt(3/2 * (np.log(self._current_round + 1) / self._actions))
        
        self._upper_bounds = self._means + deltas
        

    def updated_reward_estimate(self, arm_index, reward):
        '''
        Calculate the new running average of the arm

        Keyword arguments:
        ------
        arm_index -- int, index of the array to update
        reward -- float, reward recieved from the last action

        Returns:
        ------
        float, the new mean estimate for the selected arm
        '''
        n = self._actions[arm_index]
        current_value = self._means[arm_index]
        new_value = ((n - 1) / float(n)) * current_value + (1 / float(n)) * reward
        return new_value

    total_reward = property(_get_total_reward)
    actions = property(_get_action_history)
    best_arm = property(_get_best_arm)