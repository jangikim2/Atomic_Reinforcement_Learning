import numpy as np
import copy
from keras.layers import Dense, Flatten, Input
from keras.layers.convolutional import Conv2D
from keras import backend as K
from keras.optimizers import Adam
from keras.models import Model

class PPOIQN:
    def __init__(self):
        self.state_size = (16,16,2)
        self.action_size = 3

        self.learning_rate = 0.001
        self.gamma = 0.95
        self.clip_value = 0.2
        self.c_1 = 1
        self.c_2 = 0.005

        self.act_probs, self.spatial_probs, self.act_spatial_probs, self.v_preds = self.build_act_spatial_value()
        self.act_probs_old, self.spatial_probs_old, self.act_spatial_probs_old, self.v_preds_old = self.build_act_spatial_value()

        self.act_spatial_updater = self.act_spatial_optimizer()
        self.value_updater = self.value_optimizer()

    def build_act_spatial_value(self):
        ##### non-spatial action policy, spatial action policy
        input = Input(shape=self.state_size, name='obs')
        conv = Conv2D(filters=16, kernel_size=[5, 5], strides=[1, 1], padding='same', activation='relu')(input)
        conv = Conv2D(filters=32, kernel_size=[3, 3], strides=[1, 1], padding='same', activation='relu')(conv)
        conv = Conv2D(filters=64, kernel_size=[3, 3], strides=[1, 1], padding='same', activation='relu')(conv)
        conv = Conv2D(filters=128, kernel_size=[3, 3], strides=[1, 1], padding='same', activation='relu')(conv)

        conv = Flatten()(conv)

        dense_1 = Dense(units=256, activation=K.relu)(conv)

        vision_model = Model(input, dense_1)

        # Then define the tell-digits-apart model
        digit_a = Input(shape=self.state_size)
        digit_b = Input(shape=self.state_size)

        # The vision model will be shared, weights and all
        out_a = vision_model(digit_a)
        out_b = vision_model(digit_b)

        o_act_probs = Dense(units=3, activation=K.softmax)(out_a)
        o_spatial_probs = Dense(units=16 * 16, activation=K.softmax)(out_b)

        act_probs = Model(inputs=digit_a, outputs=o_act_probs)
        spatial_probs = Model(inputs=digit_b, outputs=o_spatial_probs)

        act_spatial_probs = Model(inputs=[digit_a, digit_b], outputs=[o_act_probs, o_spatial_probs])

        act_probs.summary()
        spatial_probs.summary()

        ##### value-state
        dense_2 = Dense(units=64, activation=K.relu)(dense_1)
        o_v_preds = Dense(units=1, activation=None, trainable=True, kernel_initializer='glorot_uniform')(dense_2)

        v_preds = Model(inputs=input, outputs=o_v_preds)
        v_preds.summary()

        return act_probs, spatial_probs, act_spatial_probs, v_preds

    def get_action(self, obs):
        action_policy = []
        spatial_policy = []
        v_predict = []

        state = np.reshape(obs, (-1, 16, 16, 2))

        o_action_policy = self.act_probs.predict(state)[0]

        action_policy.append([])
        for i in range(len(o_action_policy)):
            action_policy[0].append(o_action_policy[i])

        o_spatial_policy = self.spatial_probs.predict(state)[0]
        spatial_policy.append([])
        for i in range(len(o_spatial_policy)):
            spatial_policy[0].append(o_spatial_policy[i])

        o_v_predict = self.v_preds.predict(state)[0]
        v_predict.append([])
        v_predict[0].append(o_v_predict[0])

        return np.array(action_policy), np.array(spatial_policy), np.array(v_predict)

    def get_gaes(self, rewards, v_preds, v_preds_next):
        deltas = [r_t + self.gamma * v_next - v for r_t, v_next, v in zip(rewards, v_preds_next, v_preds)]
        # calculate generative advantage estimator(lambda = 1), see ppo paper eq(11)
        gaes = copy.deepcopy(deltas)
        for t in reversed(range(len(gaes) - 1)):  # is T-1, where T is time step which run policy
            gaes[t] = gaes[t] + self.gamma * gaes[t + 1]
        return gaes

    def act_spatial_optimizer(self):
        '''
        self.actions = tf.placeholder(dtype=tf.int32, shape=[None], name='actions')
        self.space = tf.placeholder(dtype=tf.int32, shape=[None], name='space')
        self.rewards = tf.placeholder(dtype=tf.float32, shape=[None], name='rewards')
        self.v_preds_next = tf.placeholder(dtype=tf.float32, shape=[None], name='v_preds_next')
        self.gaes = tf.placeholder(dtype=tf.float32, shape=[None], name='gaes')
        '''

        actions = K.placeholder(dtype='int32', shape=[None], name='actions')
        space = K.placeholder(dtype='int32', shape=[None], name='space')
        rewards = K.placeholder(dtype='float32', shape=[None], name='rewards')
        v_preds_next = K.placeholder(dtype='float32', shape=[None], name='v_preds_next')
        gaes = K.placeholder(dtype='float32', shape=[None], name='gaes')

        '''
        act_probs = self.Policy.act_probs
        spatial_probs = self.Policy.spatial_probs

        act_probs_old = self.Old_Policy.act_probs
        spatial_probs_old = self.Old_Policy.spatial_probs
                
        act_probs = act_probs * tf.one_hot(indices=self.actions, depth=act_probs.shape[1])
        act_probs = tf.reduce_sum(act_probs, axis=1)

        spatial_probs = spatial_probs * tf.one_hot(indices=self.space, depth=spatial_probs.shape[1])
        spatial_probs = tf.reduce_sum(spatial_probs, axis=1)

        action_probs = tf.clip_by_value(act_probs * spatial_probs, 1e-10, 1.0)

        act_probs_old = act_probs_old * tf.one_hot(indices=self.actions, depth=act_probs_old.shape[1])
        act_probs_old = tf.reduce_sum(act_probs_old, axis=1)

        spatial_probs_old = spatial_probs_old * tf.one_hot(indices=self.space, depth=spatial_probs_old.shape[1])
        spatial_probs_old = tf.reduce_sum(spatial_probs_old, axis=1)

        action_probs_old = tf.clip_by_value(act_probs_old * spatial_probs_old, 1e-10, 1.0)
        '''

        act_probs = self.act_probs.output * K.one_hot(actions, self.act_probs.output_shape[1])
        act_probs = K.sum(act_probs, axis=1)

        spatial_probs = self.spatial_probs.output * K.one_hot(space, self.spatial_probs.output_shape[1])
        spatial_probs = K.sum(spatial_probs, axis=1)

        action_probs = K.clip(act_probs * spatial_probs, 1e-10, 1.0)

        act_probs_old = self.act_probs_old.output * K.one_hot(actions, self.act_probs_old.output_shape[1])
        act_probs_old = K.sum(act_probs_old, axis=1)

        spatial_probs_old = self.spatial_probs_old.output * K.one_hot(space, self.spatial_probs_old.output_shape[1])
        spatial_probs_old = K.sum(spatial_probs_old, axis=1)

        action_probs_old = K.clip(act_probs_old * spatial_probs_old, 1e-10, 1.0)

        '''
        with tf.variable_scope('loss/clip'):
            spatial_ratios = tf.exp(tf.log(action_probs)-tf.log(action_probs_old))
            clipped_spatial_ratios = tf.clip_by_value(spatial_ratios, clip_value_min=1-clip_value, clip_value_max=1+clip_value)
            loss_spatial_clip = tf.minimum(tf.multiply(self.gaes, spatial_ratios), tf.multiply(self.gaes, clipped_spatial_ratios))
            loss_spatial_clip = tf.reduce_mean(loss_spatial_clip)
            tf.summary.scalar('loss_spatial', loss_spatial_clip)
        '''
        spatial_ratios = K.exp(K.log(action_probs) - K.log(action_probs_old))
        clipped_spatial_ratios = K.clip(spatial_ratios, 1 - self.clip_value, 1 + self.clip_value)
        loss_spatial_clip = K.minimum(gaes * spatial_ratios,
                                       gaes * clipped_spatial_ratios)
        loss_spatial_clip = K.mean(loss_spatial_clip)

        '''
        with tf.variable_scope('loss/vf'):
            v_preds = self.Policy.v_preds
            loss_vf = tf.squared_difference(self.rewards + self.gamma * self.v_preds_next, v_preds)
            loss_vf = tf.reduce_mean(loss_vf)
            tf.summary.scalar('loss_vf', loss_vf)
        '''

        loss_vf_differnece = rewards + self.gamma * v_preds_next - self.v_preds.output
        loss_vf_squared = K.square(loss_vf_differnece)
        loss_vf = K.mean(loss_vf_squared)

        '''
        with tf.variable_scope('loss/entropy'):
            act_probs = self.Policy.act_probs
            spatial_probs = self.Policy.spatial_probs

            act_entropy = -tf.reduce_sum(self.Policy.act_probs * tf.log(tf.clip_by_value(self.Policy.act_probs, 1e-10, 1.0)), axis=1)
            spatial_entropy = -tf.reduce_sum(self.Policy.spatial_probs * tf.log(tf.clip_by_value(self.Policy.spatial_probs, 1e-10, 1.0)), axis=1)
            act_entropy = tf.reduce_mean(act_entropy, axis=0)
            spatial_entropy = tf.reduce_mean(spatial_entropy, axis=0)

            entropy = act_entropy + spatial_entropy
            tf.summary.scalar('entropy', entropy)
        '''

        act_entropy = -K.sum(self.act_probs.output * K.log(K.clip(self.act_probs.output, 1e-10, 1.0)), axis=1)
        spatial_entropy = -K.sum(self.spatial_probs.output * K.log(K.clip(self.spatial_probs.output, 1e-10, 1.0)), axis=1)
        act_entropy = K.mean(act_entropy, axis=0)
        spatial_entropy = K.mean(spatial_entropy, axis=0)

        entropy = act_entropy + spatial_entropy

        '''
        with tf.variable_scope('loss'):
            loss = loss_spatial_clip - c_1 * loss_vf + c_2 * entropy
            loss = -loss  # minimize -loss == maximize loss
            tf.summary.scalar('loss', loss)
        '''

        loss = loss_spatial_clip - self.c_1 * loss_vf + self.c_2 * entropy
        loss = -loss  # minimize -loss == maximize loss

        optimizer = Adam(lr=self.learning_rate)
        updates = optimizer.get_updates(self.act_spatial_probs.trainable_weights, [], loss)
        train = K.function([self.act_probs.input, self.spatial_probs.input, self.act_probs_old.input,
                            self.spatial_probs_old.input, self.v_preds.input, space, actions, rewards,
                            v_preds_next, gaes], [loss], updates=updates)
        return train


    def value_optimizer(self):
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
        self.spatial_probs_old.set_weights(self.spatial_probs.get_weights())
        self.v_preds_old.set_weights(self.v_preds.get_weights())

    def train(self, obs, spatial, actions, rewards, v_preds_next, gaes):
        state = K.reshape(obs, [-1, 16, 16, 2])

        self.act_probs_old.trainable = self.spatial_probs_old.trainable = self.v_preds.trainable = self.act_spatial_probs_old.trainable = False
        self.act_spatial_updater([state, state, state, state, state, spatial, actions, rewards, v_preds_next, gaes])
        self.act_probs_old.trainable = self.spatial_probs_old.trainable = self.v_preds.trainable = self.act_spatial_probs_old.trainable =True

        self.value_updater([state, rewards, v_preds_next])

