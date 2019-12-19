import numpy as np

from ovs.toy_models import BanditCasino, GaussianBandit, guassian_bandit_sequence
from ovs.fixed_budget import OCBA

if __name__ == '__main__':
    designs = guassian_bandit_sequence(1, 11)
    
    environment = BanditCasino(designs)

    ocba = OCBA(environment, len(designs), 200, 10, n_0=10, min=False)

    results = ocba.solve()
    print('best design:\t{}'.format(results))
    print('allocations:\t{}'.format(ocba._allocations))
    print('total reps:\t{}'.format(ocba._allocations.sum()))

    np.set_printoptions(precision=2)
    print('means:\t\t{0}'.format(ocba._means))
    print('vars:\t\t{0}'.format(ocba._vars))