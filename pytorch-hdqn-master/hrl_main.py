import matplotlib
#%matplotlib inline
#matplotlib.style.use('ggplot')
import matplotlib.pyplot as plt

import torch.optim as optim

from envs.mdp import StochasticMDPEnv
#from agents.hdqn_mdp import hDQN, OptimizerSpec
from hrl_mdp import hrlTD3, OptimizerSpec
from hdqn import hdqn_learning
from htd3 import htd3_learning
from utils.plotting import plot_episode_stats, plot_visited_states
from utils.schedule import LinearSchedule

# for auto-reloading external modules
# see http://stackoverflow.com/questions/1907993/autoreload-of-modules-in-ipython
#%load_ext autoreload
#%autoreload 2

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy_name", default="TD3")  # Policy name
    parser.add_argument("--env_name", default="HalfCheetah-v1")  # OpenAI gym environment name
    parser.add_argument("--seed", default=0, type=int)  # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--start_timesteps", default=1e4,
                        type=int)  # How many time steps purely random policy is run for
    parser.add_argument("--eval_freq", default=5e3, type=float)  # How often (time steps) we evaluate
    parser.add_argument("--max_timesteps", default=1e6, type=float)  # Max time steps to run environment for
    parser.add_argument("--save_models", action="store_true")  # Whether or not models are saved
    parser.add_argument("--expl_noise", default=0.1, type=float)  # Std of Gaussian exploration noise
    parser.add_argument("--batch_size", default=100, type=int)  # Batch size for both actor and critic
    parser.add_argument("--discount", default=0.99, type=float)  # Discount factor
    parser.add_argument("--tau", default=0.005, type=float)  # Target network update rate
    parser.add_argument("--policy_noise", default=0.2, type=float)  # Noise added to target policy during critic update
    parser.add_argument("--noise_clip", default=0.5, type=float)  # Range to clip target policy noise
    parser.add_argument("--policy_freq", default=2, type=int)  # Frequency of delayed policy updates
    args = parser.parse_args()


    NUM_EPISODES = 12000
    BATCH_SIZE = 128
    GAMMA = 1.0
    REPLAY_MEMORY_SIZE = 1000000
    LEARNING_RATE = 0.00025
    ALPHA = 0.95
    EPS = 0.01

    plt.style.use('ggplot')


    optimizer_spec = OptimizerSpec(
        constructor=optim.RMSprop,
        kwargs=dict(lr=LEARNING_RATE, alpha=ALPHA, eps=EPS),
    )

    exploration_schedule = LinearSchedule(50000, 0.1, 1)

    '''
    agent = hDQN(
        optimizer_spec=optimizer_spec,
        replay_memory_size=REPLAY_MEMORY_SIZE,
        batch_size=BATCH_SIZE,
    )
    '''

    agent = hrlTD3(
        optimizer_spec=optimizer_spec,
        replay_memory_size=REPLAY_MEMORY_SIZE,
        batch_size=BATCH_SIZE,
    )

    env = StochasticMDPEnv()

    agent, stats, visits = htd3_learning(
        env=env,
        agent=agent,
        num_episodes=NUM_EPISODES,
        exploration_schedule=exploration_schedule,
        args=args,
        gamma=GAMMA,
    )

    plot_episode_stats(stats)