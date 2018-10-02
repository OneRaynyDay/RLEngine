"""
General purpose finite model baseclass that requires some functions to be implemented.
"""
import numpy as np
from abc import ABCMeta, abstractmethod
from copy import deepcopy

class FiniteModel(object, metaclass=ABCMeta):
    def __init__(self, state_space, action_space, gamma=1.0, epsilon=0.1):
        """FiniteModel takes in state_space and action_space (finite) 
        Arguments
        ---------
        
        state_space: int OR list[observation], where observation is any hashable type from env's obs.
        action_space: int OR list[action], where action is any hashable type from env's actions.
        gamma: float, discounting factor.
        epsilon: float, epsilon-greedy parameter.
        
        If the parameter is an int, then we generate a list, and otherwise we generate a dictionary.
        """
        self.gamma = gamma
        self.epsilon = epsilon
        self.Q = None
        if isinstance(action_space, int):
            self.action_space = np.arange(action_space)
            actions = [0]*action_space
            # Action representation
            self._act_rep = "list"
        else:
            self.action_space = action_space
            actions = {k:0 for k in action_space}
            self._act_rep = "dict"
        if isinstance(state_space, int):
            self.state_space = np.arange(state_space)
            self.Q = [deepcopy(actions) for _ in range(state_space)]
        else:
            self.state_space = state_space
            self.Q = {k:deepcopy(actions) for k in state_space}
            
        # Frequency of state/action.
        self.Ql = deepcopy(self.Q)

       
    def pi(self, action, state):
        """pi(a,s,A,V) := pi(a|s)
        We take the argmax_a of Q(s,a).
        q[s] = [q(s,0), q(s,1), ...]
        """
        if self._act_rep == "list":
            if action == np.argmax(self.Q[state]):
                return 1
            return 0
        elif self._act_rep == "dict":
            if action == max(self.Q[state], key=self.Q[state].get):
                return 1
            return 0
    
   
    def b(self, action, state):
        """b(a,s,A) := b(a|s) 
        Sometimes you can only use a subset of the action space
        given the state.

        Randomly selects an action from a uniform distribution.
        """
        return self.epsilon/len(self.action_space) + (1-self.epsilon) * self.pi(action, state)


    def choose_action(self, policy, state):
        """Uses specified policy to select an action randomly given the state.
        Arguments
        ---------
        
        policy: function, can be self.pi, or self.b, or another custom policy.
        state: observation of the environment.
        """
        probs = [policy(a, state) for a in self.action_space]
        return np.random.choice(self.action_space, p=probs)

   
    @abstractmethod
    def score(self, env, policy, n_samples=1000):
        pass


    @abstractmethod
    def update_Q(self, sequence):
        pass
