import numpy as np
import random
from collections import deque

import gym
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers.normalization import BatchNormalization 
from keras.optimizers import RMSprop

from utils.live_plot import LivePlot


class Brain(object):
    def __init__(self, env,
                       nn = None,
                       **config):
        """
        Based on:
            https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf

        Parameters
        """

        # if not isinstance(env.action_space, discrete.Discrete):
        #     raise UnsupportedSpace('Action space {} incompatible with {}. (Only supports Discrete action spaces.)'.format(action_space, self))
        
        self.env = env
        self.config = {
            'eps': 0.05,
            'gamma': 0.95,
            'exploration_period': 1000,
            'store_every':5,
            'train_every':5,
            'minibatch_size': 1,
            'discount_rate': 0.95, 
            'max_experience': 5000,
            'target_nn_update_rate': 0.01,
            'maxepoch': 100,
            'maxstep': 100,
            'outdir': '/tmp/brainresults',
            'plot': True,
            'render': True,
        }
        self.config.update(config)
        self.plotter = LivePlot(self.config['outdir'])

        # Deep Q Agent State
        self._action_ctr = 0 # actions excuted so far
        self._iter_ctr = 0
        self._store_ctr = 0
        self._train_ctr = 0
        self._experience = deque()

        if not nn:
            self._init() # Initialise the neural networks
        else:
            self.nn = nn
            # self.target_nn = nn.copy()


    def _init_network(self):
        # Init the learning network
        self.nn = Sequential()
        self.nn.add(Dense(output_dim=self.config['nn_hidden_size'][0], input_dim=self.env.observation_space.n))
        self.nn.add(Activation('relu'))
        self.nn.add(Dense(output_dim=self.env.action_space.n))
        self.nn.add(Activation('linear')) # Q values
        # Init the target network
        self.target_nn = self.nn.copy()

        # Compile both networks
        self.nn.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy']) # To be determined
        self.target_nn.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])


    def _act(self, state):
        self._action_ctr += 1
        # eps = self._linear_annealing(self._action_ctr,
        #                                       self.config['exploration_period'],
        #                                       1.0,
        #                                       self.config['eps'])
        state = np.reshape(state, newshape=(1, self.env.observation_space.shape[-1]))
        out = self.nn.predict(state, batch_size=1)
        a = out.argmax() if np.random.random() > self.config['eps'] else self.env.action_space.sample()
        return a, out


    def _store(self, experience):
        if self._store_ctr % self.config['store_every'] is 0:
            self._experience.append(experience)
            if len(self._experience) > self.config['max_experience']:
                self._experience.popleft()
        self._store_ctr += 1


    def _update_nn(self):
        # Starting with one step learning with no memory 

        # sample experience.
        samples   = [self._experience[i] for i in random.sample(range(len(self._experience)), self.config['minibatch_size'])]

        # Make batch
        S0 = np.empty((len(samples), self.env.observation_space.shape[0]))
        S1 = np.empty((len(samples), self.env.observation_space.shape[0]))
        R = np.empty((len(samples),))
        A = np.empty((len(samples),), dtype=int)
        tdtarget = np.empty((len(samples), self.env.action_space.n))

        for i, (s0, a0, r, s1, out0) in enumerate(samples):
            S0[i] = s0
            R[i] = r          
            S1[i] = s1 if s1 is not None else 0 
            tdtarget[i] = out0
            A[i] = a0
        
        tdtarget[:, A] = R + self.config['gamma'] * self.nn.predict(S1).max(axis=1) # TODO: Should use the target network
        self.nn.train_on_batch(S0, tdtarget)


    def learn(self):
        config = self.config
        env.monitor.start(config['outdir'], force=True, seed=0)

        for e in range(config['maxepoch']):
            s0 = self.env.reset()

            for i in range(config['maxstep']):
                a0, out0 = self._act(s0)
                q0 = out0.max()
                s1, r0, done, _ = env.step(a0)
                _, out1 = self._act(s1)
                q1 = out1.max()

                self._store((s0, a0, r0, s1, out0)) # Saving out0 for batch training

                if len(self._experience) > self.config['minibatch_size']:
                    self._update_nn()

                # Shift transition 
                s0 = s1
                if done:
                    break

                if self.config['plot']: self.plotter.plot()
                if self.config['render']: env.render()

        env.monitor.close()


    def _linear_annealing(self, n, total, p_initial, p_final):
        """Linear annealing between p_initial and p_final
        over total steps - computes value at step n"""
        if n >= total:
            return p_final
        else:
            return p_initial - (n * (p_initial - p_final)) / (total)


if __name__ == '__main__':
    env = gym.make('LunarLander-v2')

    # Init the learning network
    nn = Sequential()
    print('nstate:', env.observation_space.shape[0]) 
    nn.add(Dense(output_dim=100, input_dim=env.observation_space.shape[0]))
    nn.add(Activation('tanh'))
    nn.add(BatchNormalization(axis=-1))
    nn.add(Dense(output_dim=50))
    nn.add(Activation('tanh'))
    nn.add(BatchNormalization(axis=-1))
    nn.add(Dense(output_dim=env.action_space.n))
    nn.add(Activation('linear')) # Q values
    nn.compile(loss='mse', 
               optimizer=RMSprop(lr=0.01, rho=0.9, epsilon=1e-08), 
               metrics=['accuracy'])

    b = Brain(env, nn, maxepoch=20000, maxstep=env.spec.timestep_limit, eps=0.05, gamma = 0.95, minibatch_size=2000, render=False, max_experience=50000)
    b.learn()