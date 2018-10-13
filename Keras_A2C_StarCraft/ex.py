import sys
from absl import flags
from pysc2.env import sc2_env, environment
from pysc2.lib import actions, features
import matplotlib.pyplot as plt
import random
import tensorflow as tf
import numpy as np
# from a2c import a2c, disconut_rewards
from a2c import A2CAgent
import time

# from file_writer import open_file_and_save

if __name__ == "__main__":

    FLAGS = flags.FLAGS
    FLAGS(sys.argv)

    _SELECT_ARMY = actions.FUNCTIONS.select_army.id
    _MOVE_SCREEN = actions.FUNCTIONS.Move_screen.id
    _NO_OP = actions.FUNCTIONS.no_op.id
    _NOT_QUEUED = [0]
    _QUEUED = [1]
    _SELECT_ALL = [0]

    env = sc2_env.SC2Env(map_name='MoveToBeacon',
                         agent_interface_format=sc2_env.parse_agent_interface_format(
                             feature_screen=16,
                             feature_minimap=16,
                             rgb_screen=None,
                             rgb_minimap=None,
                             action_space=None,
                             use_feature_units=False),
                         step_mul=4,
                         game_steps_per_episode=None,
                         disable_fog=False,
                         visualize=False)
    #with tf.Session() as sess:
        #A2C = a2c(sess, 0.00001)

        #sess.run(tf.global_variables_initializer())
        #saver = tf.train.Saver()
        # saver.restore(sess, "4wayBeacon_a2c/tmp/model.ckpt")

    state_size = 2
    action_size = 4

    agent = A2CAgent(state_size, action_size)

    for episodes in range(62626):

        obs = env.reset() #####

        action = actions.FunctionCall(_SELECT_ARMY, [_SELECT_ALL])
        obs = env.step(actions=[action])

        done = False #####
        sub_done = False
        global_step = 0

        ##### states = np.empty(shape=[0, 2])
        ##### actions_list = np.empty(shape=[0, 4])
        ##### next_states = np.empty(shape=[0, 2])
        ##### rewards = np.empty(shape=[0, 1])

        marine_y, marine_x = (obs[0].observation.feature_screen.base[5] == 1).nonzero()
        beacon_y, beacon_x = (obs[0].observation.feature_screen.base[5] == 3).nonzero()
        marine_x, marine_y, beacon_x, beacon_y = np.mean(marine_x), np.mean(marine_y), np.mean(beacon_x), np.mean(
            beacon_y)
        state = [marine_x * 10 / 63 - beacon_x * 10 / 63, marine_y * 10 / 63 - beacon_y * 10 / 63]

        state = np.reshape(state, [1, state_size]) #####

        while not done:
            global_step += 1

            ##### act = A2C.choose_action(state)
            act = agent.get_action(state)

            if act == 0:
                dest_x, dest_y = marine_x + 3, marine_y
            if act == 1:
                dest_x, dest_y = marine_x, marine_y + 3
            if act == 2:
                dest_x, dest_y = marine_x - 3, marine_y
            if act == 3:
                dest_x, dest_y = marine_x, marine_y - 3
            actions_space = actions.FunctionCall(_MOVE_SCREEN, [_NOT_QUEUED, [np.clip(dest_x, 0, 15),
                                                                              np.clip(dest_y, 0, 15)]])  # move Up

            obs = env.step(actions=[actions_space])
            reward = -0.1
            marine_y, marine_x = (obs[0].observation.feature_screen.base[5] == 1).nonzero()
            beacon_y, beacon_x = (obs[0].observation.feature_screen.base[5] == 3).nonzero()
            marine_x, marine_y, beacon_x, beacon_y = np.mean(marine_x), np.mean(marine_y), np.mean(beacon_x), np.mean(
                beacon_y)
            distance = (marine_x * 10 / 63 - beacon_x * 10 / 63) ** 2 + (marine_y * 10 / 63 - beacon_y * 10 / 63) ** 2
            next_state = [marine_x * 10 / 63 - beacon_x * 10 / 63, marine_y * 10 / 63 - beacon_y * 10 / 63]
            next_state = np.reshape(next_state, [1, state_size]) #####

            if global_step == 200 or distance < 0.3: done = True
            ##### if global_step == 200: reward = -1
            ##### if distance < 0.3: reward = 0

            if global_step == 200:
                reward = -1
                sub_done = False
            if distance < 0.3:
                reward = 0
                sub_done = True

            ###### states = np.vstack([states, state])
            ###### next_states = np.vstack([next_states, next_state])
            ###### rewards = np.vstack([rewards, reward])
            ###### action = np.zeros(4)
            ###### action[act] = 1
            ###### actions_list = np.vstack([actions_list, action])

            # 샘플을 저장
            ##### self.append_sample(history, action, reward)
            agent.append_sample(state, act, reward)

            if done:
                ##### A2C.learn(states, next_states, rewards, actions_list)
                agent.train_model(sub_done, episodes)
                # saver.save(sess, "4wayBeacon_a2c/tmp/model.ckpt")
                ##### print(sum(rewards), episodes)
                # open_file_and_save('4wayBeacon_a2c/reward.csv', [sum(rewards)])

            state = next_state

    env.close()