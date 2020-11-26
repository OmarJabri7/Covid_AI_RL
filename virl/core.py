import gym
import numpy as np

class Agent(gym.Env):
    """ Implements the gym.Env interface. 
    
    Each agent in a Closed Loop Data Science simulation acts on the union 
    of observations made available by other agents, and returns an object 
    representing what can be observed of this agent by others.
    
    Example: Alice and Bob play No Limit Holdem poker and Bob observes Alice
    raising on the flop. Bob's action involves deciding whether to fold, call
    or raise based on his hand, the cards on the flop, his belief on Alice's 
    hand cards, etc. As a CLDSAgent, Bob's 'step' method would receive 
    {'Alice': Raise(amount)} as his environment's action, and return 'fold' 
    as the observable result of his own action.
    
    Most documentation below is copied, for convenience, from 
    https://github.com/openai/gym/blob/master/gym/core.py
    
    The main API methods that users of this class need to know are:
        step
        reset
        render
        close
        seed
    """
    
    def reset(self):
        """Resets the state of the agent and returns an initial observable.
        Returns:
            observation (object): the initial observation.
        """
        raise NotImplementedError
        
    def step(self, action):
        """Run one timestep of the environment's dynamics. When end of
        episode is reached, you are responsible for calling `reset()`
        to reset this environment's state.
        
        Accepts an action and returns a tuple (observation, reward, done, info).
        
        Args:
            action (dict): a dictionary of observations made available by other agents.
            
        Returns:
            observation (object): information the agent makes observable to others
            reward (float) : amount of reward returned after previous action
            done (bool): whether the episode has ended, in which case further step() calls will return undefined results
            info (dict): contains auxiliary diagnostic information (helpful for debugging, and sometimes learning)
        """
        raise NotImplementedError
        
    @staticmethod
    def _get_input(var, action):
        """ Facilitates the implementation of agents that support both
        (a) parameters of the step function remain constant from __init__, and 
        (b) parameters are read from the `action` input.
        
        If var is a string, then action[var]. 
        If var is a function, the function is applied to the action.
        By default, var is returned.
        """
        if isinstance(var, str):
            return action[var]
        elif callable(var):
            return var(action)
        else: 
            return var  
			
			
class Composite(Agent):

    def __init__(self, order='concurrent'):
        """
        Composite agent that calls child agents' step, provides a global store 
        of observables, applies postprocessing to agent outputs and 
        preprocessing of agent inputs.
        
        Args:
            order (str): Execution order of child agents. 
                With 'concurrent' order all agents observe outputs of all other
                agents from the previous timestep.
                With 'sequential' order agent i observes outputs of agents 
                0,..i-1 from the current time step and outputs of agents 
                i,..N-1 from the previous time step.
        """
        self.order = order
        self.agents = []
        self.observable = None
        
        
        # TODO: set the following as required by gym.Env
        #action_space: The Space object corresponding to valid actions
        #observation_space: The Space object corresponding to valid observations
        #reward_range: A tuple corresponding to the min and max possible rewards
        self.metadata = {'render.modes': []}
        self.reward_range = (-float('inf'), float('inf'))
        self.spec = None
    
    class Internal(Agent):
        """ Agent wrapper augmenting instances with input preprocessing and output 
           postprocessing.
        
        Args:
            agent (Agent): agent to be wrapped.
            out (str): agent output channel name.
            pre (callable): function applied to observable input before `step`
                is called.
            post (callable): function applied to agent output before sending it
                to `out`.
        """
        
        def __init__(self, agent, out, pre=None, post=None):
            assert isinstance(agent, Agent)
            assert isinstance(out, str)
            self.agent = agent
            self.out = out
            self.pre = pre
            self.post = post
            
        def reset(self):
            return self._wrap_output(self.agent.reset())
            
        def step(self, observable):
            o, r, d, i = self.agent.step(self._wrap_input(observable))
            return self._wrap_output(o), r, d, i
        
        def _wrap_input(self, observable):
            return observable if self.pre is None else self.pre(observable)
        
        def _wrap_output(self, output):
            return {self.out: output if self.post is None else self.post(output)}  
    
    def add(self, agent, out, pre=None, post=None):
        self.agents.append(self.Internal(agent=agent, out=out, pre=pre, post=post))
    
    def reset(self):
        """ Resets observable by resetting all agents. """
        self.observable = {}
        for agent in self.agents:
            o = agent.reset()
            for k, v in o.items():
                self.observable[k] = v
            
        return self.observable
        
    
    def step(self, action=None):
        """ Updates joint observable by stepping all agents once.
        
        Args:
            action (dict): (key, value) pairs added to observable before 
                agent.step is called.
        
        Returns:
            observation (object): dictionary of agent observations.
            reward (float) : dictionary of rewards (if any).
            done (bool): dictionary of done indicators.
            info (dict): dictionary of auxiliary diagnostic information returned 
                from all agents.
        """
        if action is not None:
            for k, v in action.items():
                self.observable[k] = v
                
        if self.order == 'concurrent':
            observable, reward, done, info = {}, 0, False, None
            for agent in self.agents:
                o, r, d, i = agent.step(self.observable)
                for k, v in o.items():
                    observable[k] = v
                    
        if self.order == 'sequential':
            # initialize current step's output with previous step's output.
            observable, reward, done, info = dict(self.observable), 0, False, None
            for agent in self.agents:
                # agents receive latest outputs from all agents
                o, r, d, i = agent.step(observable) 
                for k, v in o.items():
                    observable[k] = v
            
        self.observable = observable
        return observable, reward, done, info