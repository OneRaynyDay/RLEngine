# Monte Carlo Engine

A general purpose monte carlo engine for solving reinforcement learning problems.

We get back a 4-tuple from openai gyms, which is of the form: `(observation, reward, done, info)`. We will use observations, reward, and done to generate episodes via an off-policy algorithm.

And then, we will feed the episodes to a master policy that will run improvements upon it. Simple idea.
