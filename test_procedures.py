import numpy as np

from ovs.toy_models import BanditCasino, GaussianBandit, guassian_bandit_sequence, guassian_sequence_model, random_guassian_model
from ovs.fixed_budget import OCBA
from ovs.indifference_zone import KN

if __name__ == '__main__':
    n_designs = 10
    model = guassian_sequence_model(1, n_designs)

    #something not quite right with KN.  
    #quite often is incorrect with random model, but does a lot better with 
    #sequence.  This suggests taht something wrong with contenders screening...
    model = random_guassian_model(mean_low=1, mean_high=15, 
                                  var_low=1, var_high=1,
                                  n_designs=10)

    kn = KN(model=model, 
            n_designs=10, 
            delta=1.0, 
            alpha=0.05, 
            n_0=5)

    kn.reset()
    best_design = kn.solve()
    print('best design\t{0}'.format(best_design))
    print('allocations\t{0}'.format(kn._allocations))
    print('total reps\t{0}'.format(kn._allocations.sum()))
    print('means\t\t{0}'.format(kn._means))
    model[best_design[0]]._mu