print("...starting simulations")
import cliffwalking_mc
print("...mc finished...")
import cliffwalking_sarsa
print("...sarsa finished...")
import cliffwalking_qlearning
print("...qlearning finished...")

import numpy as np
import matplotlib.pyplot as plt

N = 3000
def polyfit(list_y, deg=3):
    poly = np.polyfit(np.arange(len(list_y)), list_y, deg)
    poly_y = np.poly1d(poly)(np.arange(len(list_y)))
    return poly_y

def running_mean(x, N=20):
    cumsum = np.cumsum(np.insert(x, 0, 0)) 
    return (cumsum[N:] - cumsum[:-N]) / float(N)

mc = -np.log(-np.array(cliffwalking_mc.history[:N]))
sarsa = -np.log(-np.array(cliffwalking_sarsa.history[:N]))
qlearning = -np.log(-np.array(cliffwalking_qlearning.history[:N]))

plt.plot(np.arange(len(mc)), mc, 'o', label='mc', markersize=1, alpha=0.4)
plt.plot(running_mean(mc), label='mc_interp')
plt.plot(np.arange(len(sarsa)), sarsa, 'o', label='sarsa', markersize=1, alpha=0.4)
plt.plot(running_mean(sarsa), label='sarsa_interp')
plt.plot(np.arange(len(qlearning)), qlearning, 'o', label='qlearning', markersize=1, alpha=0.4)
plt.plot(running_mean(qlearning), label='qlearning_interp')

plt.legend()
plt.xlabel('Episodes')
plt.ylabel('reward (log scale) per episode')
plt.title('Comparison of different RL methods reward during training')
plt.savefig('cliffwalking_learning_plot.png')
