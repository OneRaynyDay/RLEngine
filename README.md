# RL Engine

An RL engine (monte carlo, SARSA, Qlearning included) for solving reinforcement learning problems.

We get back a 4-tuple from openai gyms, which is of the form: `(observation, reward, done, info)`. We will use observations, reward, and done to generate episodes via an off-policy algorithm.

And then, we will feed the episodes to a target policy that will run improvements upon it. Simple idea.

Details about implementation is covered in the [blog about Monte Carlo](https://oneraynyday.github.io/ml/2018/05/24/Reinforcement-Learning-Monte-Carlo/), and the [blog about TD methods](https://oneraynyday.github.io/ml/2018/09/30/Reinforcement-Learning-TD/).

Try running the python files in examples. They are examples on how to use the MC model.
