import sys
from absl import flags
from pysc2.env import sc2_env, environment
from pysc2.lib import actions, features
import matplotlib.pyplot as plt
import random
import tensorflow as tf
import numpy as np
##### from policy_net import Policy_net
from ppo import PPOIQN
import time
##### from file_writer import open_file_and_save

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
                         visualize=True)
    #####with tf.Session() as sess:
        #####    Policy = Policy_net('policy')
        #####    Old_Policy = Policy_net('old_policy')
        #####    PPO = PPOTrain(Policy, Old_Policy, gamma=0.95)
        #####    sess.run(tf.global_variables_initializer())
        ##### saver = tf.train.Saver()
        ##### saver.restore(sess, "4wayBeacon_ppo/tmp/model.ckpt")

    PPO = PPOIQN()
    PPO.assign_policy_parameters()

    for episodes in range(1000000):

        observations = []
        actions_list = []
        v_preds = []
        rewards = []

        obs = env.reset()

        action = actions.FunctionCall(_SELECT_ARMY, [_SELECT_ALL])
        obs = env.step(actions=[action])

        done = False
        global_step = 0

        marine_y, marine_x = (obs[0].observation.feature_screen.base[5] == 1).nonzero()
        beacon_y, beacon_x = (obs[0].observation.feature_screen.base[5] == 3).nonzero()
        marine_x, marine_y, beacon_x, beacon_y = np.mean(marine_x), np.mean(marine_y), np.mean(beacon_x), np.mean(
            beacon_y)
        state = [marine_x * 10 / 63 - beacon_x * 10 / 63, marine_y * 10 / 63 - beacon_y * 10 / 63]

        while not done:
            global_step += 1

            state = np.stack([state]).astype(dtype=np.float32)
            ##### act, v_pred = Policy.act(obs=state, stochastic=True)
            act, v_pred = PPO.get_action(state)
            act, v_pred = np.asscalar(act), np.asscalar(v_pred)

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

            if global_step == 200 or distance < 0.3: done = True
            if global_step == 200: reward = -1
            if distance < 0.3: reward = 0


            observations.append(state)
            actions_list.append(act)
            v_preds.append(v_pred)
            rewards.append(reward)

            if done:
                v_preds_next = v_preds[1:] + [0]
                gaes = PPO.get_gaes(rewards, v_preds, v_preds_next)
                observations = np.reshape(observations, [-1, 2])
                actions_list = np.array(actions_list).astype(dtype=np.int32)
                rewards = np.array(rewards).astype(dtype=np.float32)
                v_preds_next = np.array(v_preds_next).astype(dtype=np.float32)
                gaes = np.array(gaes).astype(dtype=np.float32)

                PPO.assign_policy_parameters()

                inp = [observations, actions_list, rewards, v_preds_next, gaes]
                ##### print(gaes)
                for epoch in range(4):
                    sample_indices = np.random.randint(low=0, high=observations.shape[0],
                                                       size=64)  # indices are in [low, high)
                    sampled_inp = [np.take(a=a, indices=sample_indices, axis=0) for a in inp]  # sample training data
                    PPO.train(obs=sampled_inp[0],
                              actions=sampled_inp[1],
                              rewards=sampled_inp[2],
                              v_preds_next=sampled_inp[3],
                              gaes=sampled_inp[4])

                # saver.save(sess, "4wayBeacon_ppo/tmp/model.ckpt")
                print(sum(rewards), episodes)
                # open_file_and_save('4wayBeacon_ppo/reward.csv', [sum(rewards)])

            state = next_state

    env.close()