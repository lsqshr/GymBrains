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
            'store_every':5,
            'train_every':5,
            'minibatch_size': 1,
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

        self.nn = nn # The keras network should be compiled outside
        self.tnn = Sequential.from_config(nn.get_config()) # Init target NN
        self.tnn.set_weights(self.nn.get_weights())


    def _act(self, state):
        self._action_ctr += 1
        state = np.reshape(state, newshape=(1, self.env.observation_space.shape[-1]))
        out = self.nn.predict(state)
        a = out.argmax() if np.random.random() > self.config['eps'] else self.env.action_space.sample()
        return a, out


    def _store(self, experience):
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
            A[i] = a0
            tdtarget[i] = out0
        
        q1 = self.tnn.predict(S1).max(axis=1)
        for i in range(tdtarget.shape[0]):
            tdtarget[i, A[i]] = R[i] + self.config['gamma'] * q1[i]

        self.nn.train_on_batch(S0, tdtarget)


    def _update_target_nn():
        rho = self.config['target_nn_update_rate']
        new = []

        # Soft copy the weights to target NN
        for i, l, tl in enumerate(zip(self.nn.get_weights(), self.tnn.get_weights())):
            new.append((1 - rho) * tl + rho * l)

        self.tnn.set_weights(new)


    def learn(self):
        config = self.config
        env.monitor.start(config['outdir'], force=True, seed=0)

        for e in range(config['maxepoch']): # Episode
            s0 = self.env.reset()
            score = 0
            done = False

            for i in range(config['maxstep']): # Step
            # while not done:
                print('step: %d score:%d' % (i, score), end='\r')
                a0, out0 = self._act(s0)
                q0 = out0.max()
                s1, r0, done, _ = env.step(a0)
                _, out1 = self._act(s1)
                q1 = out1.max()

                if done and i < env.spec.timestep_limit:
                    r0 = 1000 # If make it, send a big reward
                score += r0
                r0 += score / 100 # Reward will be the accumulative score divied by 100

                if self._store_ctr % self.config['store_every'] is 0:
                    self._store((s0, a0, r0, s1, out0)) # Saving out0 for batch training

                if len(self._experience) > self.config['minibatch_size']:
                    self._update_nn()
                    self._update_target_nn()

                # Shift transition 
                s0 = s1
                if done: break

                if self.config['plot']: self.plotter.plot()
                if self.config['render']: env.render()

        env.monitor.close()


if __name__ == '__main__':
    env = gym.make('Acrobot-v0')
    env.spec.timestep_limit = 500 

    # Init the learning network
    nn = Sequential()
    print('nstate:', env.observation_space.shape[0]) 
    nn.add(Dense(output_dim=20, input_dim=env.observation_space.shape[0]))
    nn.add(Activation('tanh'))
    nn.add(BatchNormalization(axis=-1))
    nn.add(Dense(output_dim=20))
    nn.add(Activation('tanh'))
    nn.add(BatchNormalization(axis=-1))
    nn.add(Dense(output_dim=env.action_space.n))
    nn.add(Activation('linear')) # Q values
    nn.compile(loss='mse', 
               optimizer=RMSprop(lr=0.01, rho=0.5, epsilon=1e-08), 
               metrics=['accuracy'])

    b = Brain(env, nn, maxepoch=20000, maxstep=env.spec.timestep_limit, 
              eps=0.05, gamma = 0.9, minibatch_size=512, render=False, 
              max_experience=50000, target_nn_update_rate=0.01)
    b.learn()