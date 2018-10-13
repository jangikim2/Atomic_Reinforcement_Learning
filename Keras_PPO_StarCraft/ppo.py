import numpy as np
import copy
from keras.layers import Dense
from keras.models import Sequential
from keras import backend as K
from keras.optimizers import Adam

class PPOIQN:
    def __init__(self):
        self.state_size = 2
        self.action_size = 4
        self.value_size = 1

        self.discount_factor = 0.99
        self.actor_lr = 0.001
        self.critic_lr = 0.005
        self.learning_rate = 0.001

        self.gamma = 0.99

        self.act_probs = self.build_actor()
        self.v_preds = self.build_value()
        self.act_probs_old = self.build_actor()
        self.v_preds_old = self.build_value()

        self.act_updater = self.act_optimizer()
        self.value_updater = self.value_optimizer()

    def build_actor(self):

        act_probs = Sequential()

        act_probs.add(Dense(64, input_dim=self.state_size, activation='tanh',
                        kernel_initializer='he_uniform')) ### glorot_uniform
        act_probs.add(Dense(64, activation='tanh',
                        kernel_initializer='he_uniform'))
        act_probs.add(Dense(64, activation='tanh',
                        kernel_initializer='he_uniform'))
        act_probs.add(Dense(4, activation='tanh',
                        kernel_initializer='he_uniform'))
        act_probs.add(Dense(self.action_size, activation='softmax',
                        kernel_initializer='he_uniform'))

        act_probs.summary()

        return act_probs

    def build_value(self):

        v_preds = Sequential()

        v_preds.add(Dense(64, input_dim=self.state_size, activation='tanh',
                         kernel_initializer='he_uniform'))
        v_preds.add(Dense(64,  activation='tanh',
                         kernel_initializer='he_uniform'))
        v_preds.add(Dense(30,  activation='tanh',
                         kernel_initializer='he_uniform'))
        v_preds.add(Dense(self.value_size, activation=None,
                         kernel_initializer='he_uniform'))

        v_preds.summary()

        return v_preds

    def get_action(self, state):
        policy = self.act_probs.predict(state)[0]
        action_index = np.random.choice(self.action_size, 1, p=policy)[0]

        v_predict = self.v_preds.predict(np.float32(state))[0]
        ##### return action_index, policy
        return action_index, v_predict

    def get_gaes(self, rewards, v_preds, v_preds_next):
        deltas = [r_t + self.gamma * v_next - v for r_t, v_next, v in zip(rewards, v_preds_next, v_preds)]
        # calculate generative advantage estimator(lambda = 1), see ppo paper eq(11)
        gaes = copy.deepcopy(deltas)
        for t in reversed(range(len(gaes) - 1)):  # is T-1, where T is time step which run policy
            gaes[t] = gaes[t] + self.gamma * gaes[t + 1]
        return gaes

    def act_optimizer(self):

        #####state = K.placeholder(shape=[None, self.state_size],dtype='float32', name='astate')
        actions = K.placeholder(shape=[None],dtype='int32', name='aactions')
        rewards = K.placeholder(shape=[None],dtype='float32', name='arewards')
        v_preds_next = K.placeholder(shape=[None],dtype='float32', name='av_preds_next')
        gaes = K.placeholder(shape=[None],dtype='float32', name='agaes')

        ##### self.assign_ops = []
        ##### for v_old, v in zip(self.old_pi_trainable, self.pi_trainable):
            ##### self.assign_ops.append(tf.assign(v_old, v))

        ##### act_probs = self.act_probs * tf.one_hot(indices=self.actions, depth=self.act_probs.shape[1])
        ##### act_probs_old = self.act_probs_old * tf.one_hot(indices=self.actions, depth=self.act_probs_old.shape[1])

        ##### act_probs = tf.reduce_sum(act_probs, axis=1)
        ##### act_probs_old = tf.reduce_sum(act_probs_old, axis=1)

        act_probs = self.act_probs.output * K.one_hot(actions, 4)
        act_probs_old = self.act_probs_old.output * K.one_hot(actions, 4)

        act_probs = K.sum(act_probs, axis=1)
        act_probs_old = K.sum(act_probs_old, axis=1)

        ##### with tf.variable_scope('loss/clip'):
        ##### act_ratios = tf.exp(tf.log(act_probs)-tf.log(act_probs_old))
        ##### clipped_spatial_ratios = tf.clip_by_value(act_ratios, clip_value_min=1 - 0.2,
        #####                                          clip_value_max=1 + 0.2)
        ##### loss_spatial_clip = tf.minimum(tf.multiply(self.gaes, act_ratios),
        #####                               tf.multiply(self.gaes, clipped_spatial_ratios))
        ##### loss_spatial_clip = tf.reduce_mean(loss_spatial_clip)
        act_ratios = K.exp(K.log(act_probs)-K.log(act_probs_old))
        clipped_spatial_ratios = K.clip(act_ratios, 1 - 0.2, 1 + 0.2)
        loss_spatial_clip = K.minimum(gaes * act_ratios,
                                       gaes * clipped_spatial_ratios)
        loss_spatial_clip = K.mean(loss_spatial_clip)

        ##### with tf.variable_scope('loss/vf'):
        ##### loss_vf = tf.squared_difference(self.rewards + self.gamma * self.v_preds_next, self.v_preds)
        ##### loss_vf = tf.reduce_mean(loss_vf)

        loss_vf_difference = rewards + self.gamma * v_preds_next - self.v_preds.output
        loss_vf_squared = K.square(loss_vf_difference)
        loss_vf = K.mean(loss_vf_squared)

        ##### with tf.variable_scope('loss'):
        ##### loss = loss_spatial_clip - loss_vf
        ##### loss = -loss

        loss = loss_spatial_clip - loss_vf
        loss = -loss

        optimizer = Adam(lr=self.learning_rate)
        updates = optimizer.get_updates(self.act_probs.trainable_weights, [], loss)
        train = K.function([self.act_probs.input, self.act_probs_old.input, self.v_preds.input, actions, rewards, v_preds_next, gaes], [loss], updates=updates)
        return train

    def value_optimizer(self):
        ##### state = K.placeholder(shape=[None, self.state_size], dtype='float32', name='vstate')
        rewards = K.placeholder(shape=[None],dtype='float32', name='vrewards')
        v_preds_next = K.placeholder(shape=[None],dtype='float32', name='vv_preds_next')

        ##### with tf.variable_scope('loss/vf'):
        ##### loss_vf = tf.squared_difference(self.rewards + self.gamma * self.v_preds_next, self.v_preds)
        ##### loss_vf = tf.reduce_mean(loss_vf)
        loss_vf_difference = rewards + self.gamma * v_preds_next - self.v_preds.output
        loss_vf_squared = K.square(loss_vf_difference)
        loss_vf = K.mean(loss_vf_squared)

        optimizer = Adam(lr=self.learning_rate)
        updates = optimizer.get_updates(self.v_preds.trainable_weights, [], loss_vf)
        train = K.function([self.v_preds.input, rewards, v_preds_next], [loss_vf], updates=updates)
        return train

    def assign_policy_parameters(self):
        self.act_probs_old.set_weights(self.act_probs.get_weights())
        self.v_preds_old.set_weights(self.v_preds.get_weights())

    def train(self, obs, actions, rewards, v_preds_next, gaes):

        self.v_preds.trainable = False
        self.act_probs_old.trainable = False
        self.act_updater([obs, obs, obs, actions, rewards, v_preds_next, gaes])
        self.v_preds.trainable = True
        self.act_probs_old.trainable = True

        self.value_updater([obs, rewards, v_preds_next])



