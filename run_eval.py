from q_nn_learning_agent import evaluate_q_nn
from q_tabular_agent import evaluate_q_tabular
from policy_search_reinforce import evaluate_policy_search
from deterministic_agent import evaluate_deterministic_model
from random_agent import evaluate_random
import matplotlib.pyplot as plt
plt.ioff() #remove if you want to interactively see the grahs (will be a lot of graphs)
#evaluate performance of differebt agent and save graphs in the directories, if not exist, the programs will create them.

# evaluate_random(True,True) #Random/
# evaluate_deterministic_model(False,True) #Deterministic/
# rewards_q_tabular = evaluate_q_tabular(500,True,True,True,True) #Q_tabular/
rewards_q_nn = evaluate_q_nn(250,False,True,False,True) #Q_nn_plots/
# rewards_policy_nn = evaluate_policy_search(2000) #Policy_Search/