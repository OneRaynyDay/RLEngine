"""
Example in CliffWalking, using the FiniteMCModel.
"""
import sys
sys.path.append("..")

import gym
import matplotlib.pyplot as plt
import numpy as np

from mc import FiniteMCModel as MC

env = gym.make("CliffWalking-v0")

# WARNING: If you try to set eps to a very low value,
# And you attempt to get the m.score() of m.pi, there may not
# be guarranteed convergence.
eps = 10000
S = 4*12
A = 4
START_EPS = 0.7
m = MC(S, A, epsilon=START_EPS)
for i in range(1, eps+1):
    ep = []
    observation = env.reset()
    while True:
        # Choosing behavior policy
        action = m.choose_action(m.b, observation)

        # Run simulation
        next_observation, reward, done, _ = env.step(action)
        ep.append((observation, action, reward))
        observation = next_observation
        if done:
            break
    m.update_Q(ep)
    # Decaying epsilon, reach optimal policy
    m.epsilon = START_EPS*(eps-i)/eps

print("Final expected returns : {}".format(m.score(env, m.pi, n_samples=10)))

X = 12
Y = 4
Fx = np.zeros((Y, X))
Fy = np.zeros((Y, X))
for y in range(Y):
    for x in range(X):
        amax = np.argmax(m.Q[x+y*12])
        if amax == 0: # UP
            Fy[y, x] = -1
        elif amax == 1: # RIGHT
            Fx[y, x] = 1
        elif amax == 2: # DOWN
            Fy[y, x] = 1
        elif amax == 3: # LEFT
            Fx[y, x] = -1
plt.quiver(Fx,Fy)
plt.savefig("cliffwalking.png")
