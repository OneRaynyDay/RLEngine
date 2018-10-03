"""
General purpose Temporal Difference model for training on-policy methods.
"""
from base import FiniteModel
import numpy as np

class FiniteSarsaModel(FiniteModel):
    def __init__(self, state_space, action_space, gamma=1.0, epsilon=0.1, alpha=0.01):
        """SarsaModel takes in state_space and action_space (finite) 
        Arguments
        ---------
        
        state_space: int OR list[observation], where observation is any hashable type from env's obs.
        action_space: int OR list[action], where action is any hashable type from env's actions.
        gamma: float, discounting factor.
        epsilon: float, epsilon-greedy parameter.
        
        If the parameter is an int, then we generate a list, and otherwise we generate a dictionary.
        >>> m = FiniteSarsaModel(2,3,epsilon=0)
        >>> m.Q
        [[0, 0, 0], [0, 0, 0]]
        >>> m.Q[0][1] = 1
        >>> m.Q
        [[0, 1, 0], [0, 0, 0]]
        >>> m.pi(1, 0)
        1
        >>> m.pi(1, 1)
        0
        """
        super(FiniteSarsaModel, self).__init__(state_space, action_space, gamma, epsilon) 
        self.alpha = alpha
       

    def update_Q(self, sarsa):
        """Performs a TD(0) action-value update using a single step.
        Arguments
        ---------
        
        sarsa: (state, action, reward, state, action), an event in an episode.
        """
        # Generate returns, return ratio
        p_state, p_action, reward, n_state, n_action = sarsa
        q = self.Q[p_state][p_action]
        self.Q[p_state][p_action] = q + self.alpha * \
            (reward + self.gamma * self.Q[n_state][n_action] - q)
   

    def score(self, env, policy, n_samples=1000):
        """Evaluates a specific policy with regards to the env.
        Arguments
        ---------
        
        env: an openai gym env, or anything that follows the api.
        policy: a function, could be self.pi, self.b, etc.
        """
        rewards = []
        for _ in range(n_samples):
            observation = env.reset()
            cum_rewards = 0
            while True:
                action = self.choose_action(policy, observation)
                observation, reward, done, _ = env.step(action)
                cum_rewards += reward
                if done:
                    rewards.append(cum_rewards)
                    break
        return np.mean(rewards)


class FiniteQLearningModel(FiniteModel):
    def __init__(self, state_space, action_space, gamma=1.0, epsilon=0.1, alpha=0.01):
        """FiniteQLearningModel takes in state_space and action_space (finite) 
        Arguments
        ---------
        
        state_space: int OR list[observation], where observation is any hashable type from env's obs.
        action_space: int OR list[action], where action is any hashable type from env's actions.
        gamma: float, discounting factor.
        epsilon: float, epsilon-greedy parameter.
        
        If the parameter is an int, then we generate a list, and otherwise we generate a dictionary.
        >>> m = FiniteQLearningModel(2,3,epsilon=0)
        >>> m.Q
        [[0, 0, 0], [0, 0, 0]]
        >>> m.Q[0][1] = 1
        >>> m.Q
        [[0, 1, 0], [0, 0, 0]]
        >>> m.pi(1, 0)
        1
        >>> m.pi(1, 1)
        0
        """
        super(FiniteQLearningModel, self).__init__(state_space, action_space, gamma, epsilon) 
        self.alpha = alpha
       

    def update_Q(self, sars):
        """Performs a TD(0) action-value update using a single step.
        Arguments
        ---------
        
        sars: (state, action, reward, state, action) or (state, action, reward, state), 
            an event in an episode.
        
        NOTE: For Q-Learning, we don't actually use the next action, since we argmax.
        """
        # Generate returns, return ratio
        if len(sars) > 4:
            sars = sars[:4]

        p_state, p_action, reward, n_state = sars
        q = self.Q[p_state][p_action]
        max_q = max(self.Q[n_state].values()) if isinstance(self.Q[n_state], dict) else max(self.Q[n_state])
        self.Q[p_state][p_action] = q + self.alpha * \
            (reward + self.gamma * max_q - q)
   

    def score(self, env, policy, n_samples=1000):
        """Evaluates a specific policy with regards to the env.
        Arguments
        ---------
        
        env: an openai gym env, or anything that follows the api.
        policy: a function, could be self.pi, self.b, etc.
        """
        rewards = []
        for _ in range(n_samples):
            observation = env.reset()
            cum_rewards = 0
            while True:
                action = self.choose_action(policy, observation)
                observation, reward, done, _ = env.step(action)
                cum_rewards += reward
                if done:
                    rewards.append(cum_rewards)
                    break
        return np.mean(rewards)


if __name__ == "__main__":
    import doctest
    doctest.testmod()
