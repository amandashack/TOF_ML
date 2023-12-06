import numpy as np

# number of trees T = 1 - 3
# paths traversed in tree (beam search) P = 10
# max no of labels in leaf M = 100
# misclass. penality for classifiers C = 10 (1) log loss (squared hinge loss)

# P = 5, 10, 15, 20, 50, 100 ?

#space = {
#    'n_trees': [1, 2, 3, 4, 5],                            # T in Parabel paper
#    'min_branch_size': [10, 25, 50, 100, 250, 500, 1000],  # M
#    'max_depth': [10, 15, 20, 25, 30, 40, 50],
#    'linear.C': np.geomspace(0.01, 100, num=5),            # C
#    'linear.loss': ['hinge', 'log'],
#}

# Note: you can also specify probability distributions, e.g.,
# import scipy
#     'C': scipy.stats.expon(scale=100),
#     'gamma': scipy.stats.expon(scale=.1),

# a couple of notes about how to construct this:
# dropout typically should not go above ~0.3
# The ML community seems to indicate that the layer size
# for hidden layers should be about the mean number of
# input + output layers


space = {
    'dropout': [0, 0.1, 0.15, 0.2, 0.25, 0.3],
    'layer_size': [3, 4, 5, 6, 7, 8, 9],
    'alpha' : [0.001, 0.05, 0.02, 0.1],
    'batch_size' : np.exp2(np.arange(5, 9)),
    'epochs' : np.arange(5, 150, step=5)
}
