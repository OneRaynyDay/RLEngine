"""
Example in CliffWalking, using the FiniteMCModel.
"""
import sys
sys.path.append("..")

import gym
import utils
from mc import FiniteMCModel as MC

env = gym.make("CliffWalking-v0")

# WARNING: If you try to set eps to a very low value,
# And you attempt to get the m.score() of m.pi, there may not
# be guarranteed convergence.
eps = 3000
S = 4*12
A = 4
START_EPS = 0.7
m = MC(S, A, epsilon=START_EPS)

# For plotting history
history = []
SAVE_FIG = False

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

    # Add total reward to history:
    history.append(sum([x[2] for x in ep]))

    # Decaying epsilon, reach optimal policy
    m.epsilon = START_EPS*(eps-i)/eps

if SAVE_FIG:
    print("Final expected returns : {}".format(m.score(env, m.pi, n_samples=10)))
    utils.render_cliffwalking(m.Q, "cliffwalking.png")
