import numpy as np

from sim_tools.ovs.toy_models import (BanditCasino, GaussianBandit, 
                                        guassian_bandit_sequence, 
                                        custom_guassian_model, 
                                        guassian_sequence_model, 
                                        random_guassian_model)
from sim_tools.ovs.fixed_budget import OCBA
from sim_tools.ovs.indifference_zone import KN, KNPlusPlus

if __name__ == '__main__':
    
    n_designs = 10
    model = guassian_sequence_model(1, n_designs)

    #something not quite right with KN.  
    #quite often is incorrect with random model, but does a lot better with 
    #sequence.  This suggests taht something wrong with contenders screening...
    model = random_guassian_model(mean_low=1, mean_high=15, 
                                  var_low=1, var_high=1,
                                  n_designs=10)

    means = [4, 4.1, 4.2, 4, 4.1, 4.3, 4, 4.1, 4.2, 4.2]
    variances = [1, 1, 1, 0.1, 0.1, 10, 10, 10, 10, 0.1]
    SEED = 999
    N_0 = 10
    np.random.seed(SEED)
    guass_model = custom_guassian_model(means, variances)

    kn = KN(model=guass_model, 
            n_designs=10, 
            delta=0.15, 
            alpha=0.05, 
            n_0=N_0)

    kn.reset()
    best_design = kn.solve()
    print('best design\t{0}'.format(best_design))
    print('allocations\t{0}'.format(kn._allocations))
    print('total reps\t{0}'.format(kn._allocations.sum()))
    print('means\t\t{0}'.format(kn._means))


    knpp = KNPlusPlus(model=guass_model, 
            n_designs=10, 
            delta=0.15, 
            alpha=0.05, 
            n_0=N_0)

    np.random.seed(SEED)
    best_design = knpp.solve()
    print('best design\t{0}'.format(best_design))
    print('allocations\t{0}'.format(knpp._allocations))
    print('total reps\t{0}'.format(knpp._allocations.sum()))
    print('means\t\t{0}'.format(knpp._means))
    
    model[best_design[0]]._mu