"""
General purpose Monte Carlo model for training on-policy methods.
"""
from base import FiniteModel
import numpy as np

class FiniteMCModel(FiniteModel):
    def __init__(self, state_space, action_space, gamma=1.0, epsilon=0.1):
        """MCModel takes in state_space and action_space (finite) 
        Arguments
        ---------
        
        state_space: int OR list[observation], where observation is any hashable type from env's obs.
        action_space: int OR list[action], where action is any hashable type from env's actions.
        gamma: float, discounting factor.
        epsilon: float, epsilon-greedy parameter.
        
        If the parameter is an int, then we generate a list, and otherwise we generate a dictionary.
        >>> m = FiniteMCModel(2,3,epsilon=0)
        >>> m.Q
        [[0, 0, 0], [0, 0, 0]]
        >>> m.Q[0][1] = 1
        >>> m.Q
        [[0, 1, 0], [0, 0, 0]]
        >>> m.pi(1, 0)
        1
        >>> m.pi(1, 1)
        0
        >>> d = m.generate_returns([(0,0,0), (0,1,1), (1,0,1)])
        >>> assert(d == {(1, 0): 1, (0, 1): 2, (0, 0): 2})
        >>> m.choose_action(m.pi, 1)
        0
        """
        super(FiniteMCModel, self).__init__(state_space, action_space, gamma, epsilon) 
        
    
    def generate_returns(self, ep):
        """Backup on returns per time period in an epoch
        Arguments
        ---------
        
        ep: [(observation, action, reward)], an episode trajectory in chronological order.
        """
        G = {} # return on state
        C = 0 # cumulative reward
        for tpl in reversed(ep):
            observation, action, reward = tpl
            G[(observation, action)] = C = reward + self.gamma*C
        return G
    
    
    def update_Q(self, ep):
        """Performs a action-value update.
        Arguments
        ---------
        
        ep: [(observation, action, reward)], an episode trajectory in chronological order.
        """
        # Generate returns, return ratio
        G = self.generate_returns(ep)
        for s in G:
            state, action = s
            q = self.Q[state][action]
            self.Ql[state][action] += 1
            N = self.Ql[state][action]
            self.Q[state][action] = q * N/(N+1) + G[s]/(N+1)
   

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
