from numpy import np

def guassian_bandit_sequence(start, end, step=1):
    '''
    Sequence of GuassianBandit Arms.  Assumes unit variance. 
    e.g. start = 1, end = 10 return 10 bandits arms with means 1 through 10.

    Parameters:
    -------
    start - int, start of sequence of means (inclusive)
    end - int, end of sequence of means (exclusive)
    step - int, step size for sequence (default = 1)

    Returns:
    -------
    list of GaussianBandit objects ordered by means

    '''    
    return [GaussianBandit(mean) for mean in range(start, end, step)]





def custom_guass_bandit_problem(*means):
    '''
    Creates a list of BernouliBandit objects with
    user specified means

    Keyword arguments:
    ------
    *means - variable size list of means

    Returns:
    ------
    list, BernoulliBandits size = len(means)
    '''
    return [GaussianBandit(mean) for mean in means]


class GaussianBandit(object):
    '''
    Classic one armed bandit gambling machine.

    A user plays the bandit by pulling its arm.

    The bandit returns a reward normally distribution 
    with mean mu and stdev sigma
    '''

    def __init__(self, mu, sigma=1.0):
        '''
        Constructor method for Gaussian Bandit

        Keyword arguments:
        -----
        mu -- float, mean of the normal distribution
        sigma -- float, stdev of the normal distribution (default = 1.0)

        '''
        self._mu = mu
        self._sigma = sigma
        self._number_of_plays = 0
        self._total_reward = 0
        self._observers = []

    def play(self, replications):
        '''
        Pull the arm on the bernoulli bandit

        Parameters:
        ---------
        replications - int, number of replications to return

        Returns:
        -----
        reward -- int with value = 0 when no reward
                  and 1 when the pull results in a win
        '''      
        reward = np.random.normal(self._mu, self._sigma)

        self._total_reward += reward
        self._number_of_plays += 1

        return reward

    def reset(self):
        '''
        Reset the number of plays and 
        total rewards to zero.
        '''
        self._number_of_plays = 0
        self._total_reward = 0


class StandardRankingSelectionProblem(object):
    def __init__(self, designs, random_state=None):
        self._designs = designs
        self._env = BanditCasino(guassian_bandit_sequence(1, designs))
        
        if random_state is not None:
            np.random.seed(random_state)

    def simulate(designs, replications):
        for bandit in self._bandits[designs]:
            bandit.play(replications)


class BanditCasino(object):
    
    def __init__(self, bandits):
        '''
        Casino constructor method

        Keyword arguments:
        ------
        bandits -- list, of BernoulliBandits objects
        '''
        self._bandits = bandits
        self._current_index = 0
        self._observers = []
    
    def __getitem__(self, index):
        return self._bandits[index]

    def __get_number_of_arms(self):
        return len(self._bandits)
    
    def action(self, bandit_index):
        '''
        Play a specific bandit machine.

        Notifies all observers of the outcome 

        Keyword arguments:
        -----
        bandit_index -- int, index of bandit to play 
        '''
        reward = self._bandits[bandit_index].play()
        self.notify_observers(bandit_index, reward)

    def random_action(self):
        '''
        Selects a bandit index at random and plays it.
        '''
        bandit_index = np.random.choice(len(self._bandits))
        self.action(bandit_index)

       
    def __iter__(self):
        return self

    def __next__(self):
        self._current_index += 1
        if self._current_index > len(self._bandits):
            raise StopIteration
        else:
            return self._bandits[self._current_index - 1]

    def register_observer(self, observer):
        self._observers.append(observer)
 
    def notify_observers(self, *args, **kwargs):
        for observer in self._observers:
            observer.feedback(self, *args, **kwargs) 

    number_of_arms = property(__get_number_of_arms)