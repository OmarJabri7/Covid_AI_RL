from gym import spaces
import numpy as np
import numpy.matlib

from .core import Agent

# clock-driven SIQR model that solves batches of Monte Carlo samples at once
class BatchSIQR(Agent):
    def __init__(self,
                 batch_size=1,
                 beta=.373,
                 alpha=0.067,
                 eta=0.067,
                 delta=0.036,
                 epsilon=0.01,
                 N=6e7,
                 round_state=False, step_size=0.1):
        """Class for SIQR dynamics of the environment

        This represents the 'disease dynamics' box in the hello-world graph
        At each step, it receives as input the population behavior `beta`, and
        given the true state of the system `(gamma, S, I, R)`, it transitions
        to new values `(S_, I_, R_)`

        Parameters
        ----------
        beta (float, default=0.373): the infection rate
        alpha (float, default=0.067): the transition rate from I to R (recovery or death rate for non-quarantined population)
        eta (float, default=0.067): the transition rate from I to Q
        delta (float, default=0.036): the transition rate from Q to R (recovery or death rate for quarantined population)
        N (int, default=10000): the total siwe of the population
        epsilon (float, default=0.01): initial fraction of infected and recovered members of the
            population.
        round_state(bool): round state to nearest integer after each step.

        Attributes
        ----------
        observation_space (gym.spaces.Box, shape=(3,)): at each step, the
            environment only returns the true values S, I, R
        action_space (gym.spaces.Box, shape=(1)): the value beta

        #TODO necessary to wrap gym.Env?

        """
    
        self.batch_size = batch_size  
        self.beta = self._to_batch(beta)
        self.alpha = self._to_batch(alpha)
        self.eta = self._to_batch(eta)
        self.delta = self._to_batch(delta)
        self.epsilon = self._to_batch(epsilon)
        self.N = self._to_batch(N)
        self.round_state = round_state
        self.step_size = step_size

        self.observation_space = spaces.Box(
            0, np.inf, shape=(4,), dtype=np.float64)  # check dtype
        self.action_space = spaces.Box(
            0, np.inf, shape=(1,), dtype=np.float64)

    def reset(self):
        """returns initial state (s0,  i0, r0)"""
        I0 = (self.epsilon * self.N).astype(np.int32)
        self.state = np.array([
            self._to_batch(self.N - 2*I0), # S
            self._to_batch(I0), # I
            self._to_batch(0), # Q
            self._to_batch(I0)]).T # R
        if self.round_state:
            self.state = self._pround(self.state)

        return self.state

    def step(self, action=None):
        """performs integration step"""
        beta = self._get_input(self.beta, action)
        beta_batch = self._to_batch(beta)
        
        self.state = self.euler_step(self.state, dt=1, beta=beta_batch)
        if self.round_state:
            self.state = self._pround(self.state)
            
        return self.state, 0, False, None

    @staticmethod
    def _pround(x):
        dx = np.random.uniform(size=x.shape) < (x-x.astype(np.int32))
        return x + dx
        
    # variable should have shape (batch, ) + shape
    def _to_batch(self, x, shape=()):
        # return placeholder key or callable
        if isinstance(x, str) or callable(x):
            return x
        
        x_arr = np.array(x)
        target_shape = (self.batch_size, ) + shape

        if x_arr.shape == target_shape:
            return x_arr
        elif (x_arr.shape == shape):
            return np.matlib.repmat(x_arr.reshape(shape), self.batch_size,1).reshape(target_shape)
        elif len(x_arr.shape) > 0 and x_arr.shape[0] == target_shape:
            return x_arr.reshape(target_shape)
        else:
            print("Warning: unable to convert to target shape", x, target_shape)
            return x
    
    def euler_step(self, X, dt, beta):
        
        X_ = np.array(X)
        n_steps = int(1/self.step_size)
        for _ in range(n_steps):
            dxdt = self._ode(X_, dt/n_steps, beta, self.alpha, self.eta, self.delta, self.N)
            X_ = X_ + dxdt
        return X_

    @staticmethod
    def _ode(Y, dt, beta, alpha, eta, delta, N, f=0):
        """Y = (S, I, R)^T """
        S, I, Q, R = Y[:,0], Y[:,1], Y[:,2], Y[:,3]
        
        dS = - beta * (1/N) * I * S
        dI = beta * (1/N) * I * S - (alpha + eta) * I
        dQ = eta * I - delta * Q
        dR = delta * Q + alpha * I

        return np.array([dS, dI, dQ, dR]).T * dt

    def render(self, mode='human'):
        pass

    def close(self):
        pass
		

