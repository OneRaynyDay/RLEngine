# Monte Carlo Engine

A monte carlo engine for solving reinforcement learning problems.

We get back a 4-tuple from openai gyms, which is of the form: `(observation, reward, done, info)`. We will use observations, reward, and done to generate episodes via an off-policy algorithm.

And then, we will feed the episodes to a target policy that will run improvements upon it. Simple idea.

Details about implementation is covered in the [blog](https://oneraynyday.github.io/ml/2018/05/24/Reinforcement-Learning-Monte-Carlo/).
