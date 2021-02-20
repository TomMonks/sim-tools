import numpy as np

from sim_tools.ovs.toy_models import BanditCasino, GaussianBandit, guassian_bandit_sequence
from sim_tools.ovs.fixed_budget import OCBAM

if __name__ == '__main__':
    designs = guassian_bandit_sequence(1, 11)
    
    environment = BanditCasino(designs)

    ocba = OCBAM(environment, len(designs), 200, 10, n_0=5, m=2, min=False)

    np.random.seed(101)

    results = ocba.solve()
    print('best design:\t{}'.format(results))
    print('allocations:\t{}'.format(ocba._allocations))
    print('total reps:\t{}'.format(ocba._allocations.sum()))

    np.set_printoptions(precision=2)
    print('means:\t\t{0}'.format(ocba._means))
    print('vars:\t\t{0}'.format(ocba._vars))