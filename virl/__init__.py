import gym
import numpy as np

from .core import Agent
from .siqr import BatchSIQR

class Epidemic(Agent):
	# wrapper environment around BatchSIQR
	
	def __init__(self, stochastic=False, noisy=False, problem_id=0):
		"""
		Mass-action SIQR epidemics model.
		
		Args:
			stochastic (bool): Is the infection rate sampled from some distribution at the beginning of each episode (default: False)?
			noisy (bool): Is the state a noisy estimate of the true state (default: False)?
			problem_id (int): Deterministic parameterization of the epidemic (default: 0).
		"""
		assert(problem_id >= 0 and problem_id < 10)
		self.is_stochastic = stochastic
		self.is_noisy = noisy
		self.problem_id = problem_id
		
		# action space is set of interventions on beta
		self.actions = np.array([1, .0175, 0.5, 0.65]) # beta coeffs
		self.action_space = gym.spaces.Discrete(self.actions.shape[0])
		self.beta_bounds = [0.19, 0.56] # default infectivity
		self.problems = [0.35, 0.19, 0.23, 0.27, 0.31, 0.40, 0.44, 0.48, 0.52, 0.56]
		self.N = 6e8 # population size
		self.I0= 2e4# initial infectious and recovereds
		self.noise_level = 0.2 # weighted average of noisy observation and true state
		self.action_repeat = 7 # number of days between actions
		self.steps_total = int(365/self.action_repeat) # episode length
		
		self.env = BatchSIQR(beta='beta', N=self.N, epsilon=self.I0/self.N)
		self.observation_space = self.env.observation_space
		
	def reset(self):
		self.steps = 0
		self.beta = self.problems[self.problem_id]
		if (self.is_stochastic):
			self.beta = self.beta_bounds[0] + np.random.uniform() * self.beta_bounds[1]
		s = self.env.reset().reshape(-1)
		return self._observe(s)
		
	def step(self, action):
		# map action to beta
		assert(self.steps is not None) # step called before reset
		assert(action >= 0) # action out of bounds
		assert(action < self.actions.shape[0]) # action out of bounds
		
		c = self.actions[action]
		beta = self.beta * c
		r = 0
		for _ in range(self.action_repeat):
			s, _, d, info = self.env.step({'beta': beta})
			s = s.reshape(-1)
			r += self._reward(s/self.N, c)
		self.steps += 1
		
		# corrupt observation
		self._observe(s)
		
		# check done
		if self.steps >= self.steps_total:
			d = True
			self.steps = None
		
		return self._observe(s), r/self.action_repeat, d, info
		
	def _reward(self, s, c):
		a = s[1] + s[2]
		# s: epidemic state (normalized)
		# c: policy severity
		b = 1-c
		return (-30*a - 30*a**2 - b - b**2)/62
		
	def _observe(self, s):
		if not self.is_noisy:
			return s
		else:
			noise = np.random.uniform(size=4)
			noise = noise/np.sum(noise)
			o = s * ((1-self.noise_level) + self.noise_level * noise)
			o = o/np.sum(o)*self.N
			return o
			
		
		
		

    
