from q_nn_learning_agent import evaluate_q_nn
from q_tabular_agent import evaluate_q_tabular
from policy_search_reinforce import evaluate_policy_search
from deterministic_agent import evaluate_deterministic_model
from random_agent import evaluate_random
import matplotlib.pyplot as plt
plt.ioff() #remove if you want to interactively see the grahs (will be a lot of graphs)
#evaluate performance of differebt agent and save graphs in the directories, if not exist, the programs will create them.

evaluate_random(stochastic = True, noisy = True) #Random/

evaluate_deterministic_model(stochastic = False,noisy = True) #Deterministic/

rewards_q_tabular = evaluate_q_tabular(500,stochastic_train = False,
                                       noisy_train = False,
                                       stochastic_test = False,
                                       noisy_test = False) #Q_tabular/
rewards_q_nn = evaluate_q_nn(250,
                             stochastic_train=False,
                             noisy_train=True,
                             stochastic_test=False,
                             noisy_test=True) #Q_nn_plots/

rewards_policy_nn = evaluate_policy_search(2000,
                                           stochastic_train=False,
                                           noisy_train=False,
                                           stochastic_test=False,
                                           noisy_test=False) #Policy_Search/