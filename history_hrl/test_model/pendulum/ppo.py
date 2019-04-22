import tensorflow as tf
import numpy as np

class PPO(object):

    def __init__(self, sess, act_range, METHOD, A_LR, C_LR, S_DIM, A_DIM, A_UPDATE_STEPS, C_UPDATE_STEPS):

        self.act_range = act_range
        self.METHOD = METHOD
        self.A_LR = A_LR
        self.C_LR = C_LR
        self.S_DIM = S_DIM
        self.A_DIM = A_DIM
        self.A_UPDATE_STEPS = A_UPDATE_STEPS
        self.C_UPDATE_STEPS = C_UPDATE_STEPS

        self.sess = sess
        self.tfs = tf.placeholder(tf.float32, [None, self.S_DIM], 'state')

        # critic
        with tf.variable_scope('critic'):
            cnet = tf.layers.dense(self.tfs, 100, tf.nn.relu)
            self.v = tf.layers.dense(cnet, 1)
            self.tfdc_r = tf.placeholder(tf.float32, [None, 1], 'discounted_r')
            self.advantage = self.tfdc_r - self.v
            self.closs = tf.reduce_mean(tf.square(self.advantage))
            self.ctrain_op = tf.train.AdamOptimizer(self.C_LR).minimize(self.closs)

        # actor
        pi, pi_params = self._build_anet('pi', trainable=True)
        oldpi, oldpi_params = self._build_anet('oldpi', trainable=False)
        with tf.variable_scope('sample_action'):
            self.sample_op = tf.squeeze(pi.sample(1), axis=0)       # choosing action
        with tf.variable_scope('update_oldpi'):
            self.update_oldpi_op = [oldp.assign(p) for p, oldp in zip(pi_params, oldpi_params)]

        self.tfa = tf.placeholder(tf.float32, [None, self.A_DIM], 'action')
        self.tfadv = tf.placeholder(tf.float32, [None, 1], 'advantage')
        with tf.variable_scope('loss'):
            with tf.variable_scope('surrogate'):
                # ratio = tf.exp(pi.log_prob(self.tfa) - oldpi.log_prob(self.tfa))
                ratio = pi.prob(self.tfa) / oldpi.prob(self.tfa)
                surr = ratio * self.tfadv
            if self.METHOD['name'] == 'kl_pen':
                self.tflam = tf.placeholder(tf.float32, None, 'lambda')
                kl = tf.distributions.kl_divergence(oldpi, pi)
                self.kl_mean = tf.reduce_mean(kl)
                self.aloss = -(tf.reduce_mean(surr - self.tflam * kl))
            else:   # clipping method, find this is better
                self.aloss = -tf.reduce_mean(tf.minimum(
                    surr,
                    tf.clip_by_value(ratio, 1.-self.METHOD['epsilon'], 1.+self.METHOD['epsilon'])*self.tfadv))

        with tf.variable_scope('atrain'):
            self.atrain_op = tf.train.AdamOptimizer(self.A_LR).minimize(self.aloss)

        tf.summary.FileWriter("log/", self.sess.graph)

        #self.sess.run(tf.global_variables_initializer())

    def update(self, s, a, r):
        self.sess.run(self.update_oldpi_op)
        adv = self.sess.run(self.advantage, {self.tfs: s, self.tfdc_r: r})
        # adv = (adv - adv.mean())/(adv.std()+1e-6)     # sometimes helpful

        # update actor
        if self.METHOD['name'] == 'kl_pen':
            for _ in range(self.A_UPDATE_STEPS):
                _, kl = self.sess.run(
                    [self.atrain_op, self.kl_mean],
                    {self.tfs: s, self.tfa: a, self.tfadv: adv, self.tflam: self.METHOD['lam']})
                if kl > 4*self.METHOD['kl_target']:  # this in in google's paper
                    break
            if kl < self.METHOD['kl_target'] / 1.5:  # adaptive lambda, this is in OpenAI's paper
                self.METHOD['lam'] /= 2
            elif kl > self.METHOD['kl_target'] * 1.5:
                self.METHOD['lam'] *= 2
                self.METHOD['lam'] = np.clip(self.METHOD['lam'], 1e-4, 10)    # sometimes explode, this clipping is my solution
        else:   # clipping method, find this is better (OpenAI's paper)
            [self.sess.run(self.atrain_op, {self.tfs: s, self.tfa: a, self.tfadv: adv}) for _ in range(self.A_UPDATE_STEPS)]

        # update critic
        [self.sess.run(self.ctrain_op, {self.tfs: s, self.tfdc_r: r}) for _ in range(self.C_UPDATE_STEPS)]

    def _build_anet(self, name, trainable):
        with tf.variable_scope(name):
            anet = tf.layers.dense(self.tfs, 100, tf.nn.relu, trainable=trainable)
            mu = 2 * tf.layers.dense(anet, self.A_DIM, tf.nn.tanh, trainable=trainable)
            sigma = tf.layers.dense(anet, self.A_DIM, tf.nn.softplus, trainable=trainable)
            norm_dist = tf.distributions.Normal(loc=mu, scale=sigma)
        params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name)
        return norm_dist, params

    def choose_action(self, s):
        s = s[np.newaxis, :]
        a = self.sess.run(self.sample_op, {self.tfs: s})[0]
        return np.clip(a, -2, 2)

    def get_v(self, s):
        if s.ndim < 2: s = s[np.newaxis, :]
        return self.sess.run(self.v, {self.tfs: s})[0, 0]
