import sys
import gym
import pylab
import numpy as np
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import Adam
from keras.optimizers import RMSprop
from keras import backend as K
from file_writer import open_file_and_save

##### EPISODES = 1000

# A2C agent of cartpole example
class A2CAgent:
    def __init__(self, state_size, action_size):
        self.render = False
        self.load_model = False
        # 상태와 행동의 크기 정의
        self.state_size = state_size
        self.action_size = action_size
        self.value_size = 1

        # a2c hyperparameter
        self.discount_factor = 0.99
        self.actor_lr = 0.001
        self.critic_lr = 0.005

        # Generation of policy NN and value NN 
        self.actor = self.build_actor()
        self.critic = self.build_critic()
        self.actor_updater = self.actor_optimizer()
        self.critic_updater = self.critic_optimizer()

        # The list for saving samples durin the specified time step
        self.states, self.actions, self.rewards = [], [], []

        if self.load_model:
            self.actor.load_weights("./save_model/cartpole_actor_trained.h5")
            self.critic.load_weights("./save_model/cartpole_critic_trained.h5")

    # The calculation of probability for each act receiving the satus
    def build_actor(self):
        actor = Sequential()
        '''
        actor.add(Dense(24, input_dim=self.state_size, activation='relu',
                        kernel_initializer='he_uniform'))
        actor.add(Dense(self.action_size, activation='softmax',
                        kernel_initializer='he_uniform'))
        '''
        actor.add(Dense(64, input_dim=self.state_size, activation='tanh',
                        kernel_initializer='he_uniform')) ### glorot_uniform
        actor.add(Dense(64, activation='tanh',
                        kernel_initializer='he_uniform'))
        actor.add(Dense(64, activation='tanh',
                        kernel_initializer='he_uniform'))
        actor.add(Dense(4, activation='tanh',
                        kernel_initializer='he_uniform'))
        actor.add(Dense(self.action_size, activation='softmax',
                        kernel_initializer='he_uniform'))

        actor.summary()
        return actor

    # critic: the calculation of the value of satus after receiving the status     
    def build_critic(self):
        critic = Sequential()
        '''
        critic.add(Dense(24, input_dim=self.state_size, activation='relu',
                         kernel_initializer='he_uniform'))
        critic.add(Dense(24, input_dim=self.state_size, activation='relu',
                         kernel_initializer='he_uniform'))
        critic.add(Dense(self.value_size, activation='linear',
                         kernel_initializer='he_uniform'))
        '''
        critic.add(Dense(64, input_dim=self.state_size, activation='tanh',
                         kernel_initializer='he_uniform'))
        critic.add(Dense(64,  activation='tanh',
                         kernel_initializer='he_uniform'))
        critic.add(Dense(30,  activation='tanh',
                         kernel_initializer='he_uniform'))
        critic.add(Dense(self.value_size, activation=None,
                         kernel_initializer='he_uniform'))

        critic.summary()
        return critic

    # The choice of act stochastically after receiving the policy output
    def get_action(self, history):
        ##### history = np.float32(history / 255.)
        ##### policy = self.local_actor.predict(history)[0]
        policy = self.actor.predict(history)[0]
        action_index = np.random.choice(self.action_size, 1, p=policy)[0]
        ##### return action_index, policy
        return action_index

    # Storing samples
    def append_sample(self, history, action, reward):

        self.states.append(history)
        act = np.zeros(self.action_size)
        act[action] = 1
        self.actions.append(act)
        self.rewards.append(reward)

    # The function updating the policy NN
    def actor_optimizer(self):
        action = K.placeholder(shape=[None, self.action_size])
        advantages = K.placeholder(shape=[None, ])

        policy = self.actor.output

        # Policy cross entropy error function
        action_prob = K.sum(action * policy, axis=1)
        cross_entropy = K.log(action_prob + 1e-10) * advantages
        cross_entropy = -K.sum(cross_entropy)

        # Entropy error for exploring steadily
        entropy = K.sum(policy * K.log(policy + 1e-10), axis=1)
        entropy = K.sum(entropy)

        # Generation of final error function by adding two error functions
        loss = cross_entropy + 0.01 * entropy

        optimizer = RMSprop(lr=self.actor_lr, rho=0.99, epsilon=0.01)
        updates = optimizer.get_updates(self.actor.trainable_weights, [],loss)
        train = K.function([self.actor.input, action, advantages],
                           [loss], updates=updates)
        return train

    # The function updating the value NN
    def critic_optimizer(self):
        discounted_prediction = K.placeholder(shape=(None,))

        value = self.critic.output

        # [반환값 - 가치]의 제곱을 오류함수로 함
        loss = K.mean(K.square(discounted_prediction - value))

        optimizer = RMSprop(lr=self.critic_lr, rho=0.99, epsilon=0.01)
        updates = optimizer.get_updates(self.critic.trainable_weights, [],loss)
        train = K.function([self.critic.input, discounted_prediction],
                           [loss], updates=updates)
        return train

    # The calculation of prediction during k step
    def discounted_prediction(self, rewards, done):
        discounted_prediction = np.zeros_like(rewards)
        running_add = 0

        if not done:
            # states = np.float32(
            #      self.states[-1] / 255.)
            # print(states)

            ##### running_add = self.critic.predict(np.float32(
            #####    self.states[-1] / 255.))[0]
            running_add = self.critic.predict(np.float32(
                self.states[-1]))[0]

        for t in reversed(range(0, len(rewards))):
            running_add = running_add * self.discount_factor + rewards[t]
            discounted_prediction[t] = running_add
        return discounted_prediction

    # Update of the policy NN and the value NN
    def train_model(self, done, episodes):
        discounted_prediction = self.discounted_prediction(self.rewards, done)

        ##### states = np.zeros((len(self.states), 84, 84, 4))
        states = np.zeros((len(self.states), 2),  dtype=np.float32)
        for i in range(len(self.states)):
            ##### states[i] = self.states[i]
            states[i] = np.array(self.states[i][0])

        ##### states = np.float32(states / 255.)

        values = self.critic.predict(states)
        values = np.reshape(values, len(values))

        advantages = discounted_prediction - values

        '''
        self.optimizer[0]([states, self.actions, advantages])
        self.optimizer[1]([states, discounted_prediction])
        '''
        self.actor_updater([states, self.actions, advantages])
        self.critic_updater([states, discounted_prediction])

        print(sum(self.rewards), episodes)
        ##### open_file_and_save('./reward.csv', [sum(self.rewards)])
        self.states, self.actions, self.rewards = [], [], []