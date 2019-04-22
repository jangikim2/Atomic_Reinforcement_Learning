import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
#import utils
from baselines.hrl_td3.hrl_util import H_ReplayBuffer
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Implementation of Twin Delayed Deep Deterministic Policy Gradients (TD3)
# Paper: https://arxiv.org/abs/1802.09477


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()

        self.l1 = nn.Linear(state_dim, 400)
        self.l2 = nn.Linear(400, 300)
        #self.l1 = nn.Linear(state_dim, 300)
        #self.l2 = nn.Linear(300, 300)
        self.l3 = nn.Linear(300, action_dim)

        self.max_action = max_action

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = self.max_action * torch.tanh(self.l3(x))
        return x


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        # Q1 architecture
        self.l1 = nn.Linear(state_dim + action_dim, 400)
        self.l2 = nn.Linear(400, 300)
        #self.l1 = nn.Linear(state_dim + action_dim, 300)
        #self.l2 = nn.Linear(300, 300)
        self.l3 = nn.Linear(300, 1)

        # Q2 architecture
        #self.l4 = nn.Linear(state_dim + action_dim, 400)
        #self.l5 = nn.Linear(400, 300)
        self.l4 = nn.Linear(state_dim + action_dim, 300)
        self.l5 = nn.Linear(300, 300)
        self.l6 = nn.Linear(300, 1)

    def forward(self, x, u):
        xu = torch.cat([x, u], 1)

        x1 = F.relu(self.l1(xu))
        x1 = F.relu(self.l2(x1))
        x1 = self.l3(x1)

        x2 = F.relu(self.l4(xu))
        x2 = F.relu(self.l5(x2))
        x2 = self.l6(x2)
        return x1, x2

    def Q1(self, x, u):
        xu = torch.cat([x, u], 1)

        x1 = F.relu(self.l1(xu))
        x1 = F.relu(self.l2(x1))
        x1 = self.l3(x1)
        return x1


class TD3(object):
    def __init__(self, state_dim, action_dim, max_action):
        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters())

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters())

        self.max_action = max_action

        #jangikim
        self.default_clip_range = 5
        self.scaler = StandardScaler(with_mean=False, with_std =False)

    def select_action(self, state):
        '''
        self.scaler = self.scaler.fit(state1.reshape(-1, 1))
        state = self.scaler.transform(state1.reshape(-1, 1))

        state2 = state.reshape(1, -1)
        '''
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        return self.actor(state).cpu().data.numpy().flatten()

    def get_Q_value(self, o, u, policy_noise=0.2, noise_clip=0.5):
        # Compute the target Q value
        '''
        #target_Q1, target_Q2 = self.critic_target(next_state, next_action)
        #next_state = torch.FloatTensor(o).to(device)
        next_state = torch.FloatTensor(o.reshape(1, -1)).to(device)
        # Select action according to policy and add clipped noise
        noise = torch.FloatTensor(u).data.normal_(0, policy_noise).to(device)
        noise = noise.clamp(-noise_clip, noise_clip)
        next_action = (self.actor_target(next_state) + noise).clamp(-self.max_action, self.max_action)
        '''
        #target_Q1, target_Q2 = self.critic_target(next_state, next_action)
        #target_Q = torch.min(target_Q1, target_Q2)
        '''
        self.scaler = self.scaler.fit(o1.reshape(-1, 1))
        o = self.scaler.transform(o1.reshape(-1, 1))
        self.scaler = self.scaler.fit(u1.reshape(-1, 1))
        u = self.scaler.transform(u1.reshape(-1, 1))

        o2 = o.reshape(1, -1)
        '''
        state = torch.FloatTensor(o.reshape(1, -1)).to(device)
        action = torch.FloatTensor(u.reshape(1, -1)).to(device)
        current_Q1, current_Q2 = self.critic(state,action)
        current_Q = torch.min(current_Q1, current_Q2)
        #current_Q = torch.max(current_Q1, current_Q2)
        #return target_Q
        return current_Q.detach().numpy()

    #jangikim

    def reshape_for_broadcasting(self, source12, target):
        """Reshapes a tensor (source) to have the correct shape and dtype of the target
        before broadcasting it with MPI.
        """
        dim = len(target.shape)
        shape = ([1] * (dim - 1)) + [-1]
        return np.reshape(source12.astype(target.dtype), shape)

    def normalize(self, v, clip_range=None):
        if clip_range is None:
            clip_range = self.default_clip_range
        np_mean = np.mean(v, axis = 1)
        np_std = np.std(v, axis = 1)
        mean = self.reshape_for_broadcasting(np_mean, v)
        std = self.reshape_for_broadcasting(np_std, v)
        return np.clip((v - mean) / std, clip_range, clip_range)

    #jangikim
    def train(self, replay_buffer, iterations, batch_size=100, discount=0.99, tau=0.005, policy_noise=0.2,
              noise_clip=0.5, policy_freq=2):

        #print("iterations : ", iterations)
        for it in range(iterations):

            # Sample replay buffer
            x, y, u, r, d = replay_buffer.sample(batch_size)
            #x = self.normalize(x)
            '''
            self.scaler = self.scaler.fit(x1)
            x = self.scaler.transform(x1)
            self.scaler = self.scaler.fit(y1)
            y = self.scaler.transform(y1)
            '''
           # x = np.clip((x1 - np.mean(x1)) / np.std(x1), self.clip_range, self.clip_range)
           # y = np.clip((y1 - np.mean(y1)) / np.std(y1), self.clip_range, self.clip_range)


            state = torch.FloatTensor(x).to(device)
            action = torch.FloatTensor(u).to(device)
            next_state = torch.FloatTensor(y).to(device)
            done = torch.FloatTensor(1 - d).to(device)
            reward = torch.FloatTensor(r).to(device)

            # Select action according to policy and add clipped noise
            noise = torch.FloatTensor(u).data.normal_(0, policy_noise).to(device)
            noise = noise.clamp(-noise_clip, noise_clip)
            next_action = (self.actor_target(next_state) + noise).clamp(-self.max_action, self.max_action)

            # Compute the target Q value
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + (done * discount * target_Q).detach()

            # Get current Q estimates
            current_Q1, current_Q2 = self.critic(state, action)

            # Compute critic loss
            critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

            # Optimize the critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            # Delayed policy updates
            if it % policy_freq == 0:

                # Compute actor loss
                actor_loss = -self.critic.Q1(state, self.actor(state)).mean()

                # Optimize the actor
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

                # Update the frozen target models
                for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

                for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    def save(self, filename, directory):
        torch.save(self.actor.state_dict(), '%s/%s_actor.pth' % (directory, filename))
        torch.save(self.critic.state_dict(), '%s/%s_critic.pth' % (directory, filename))

    def load(self, filename, directory):
        self.actor.load_state_dict(torch.load('%s/%s_actor.pth' % (directory, filename)))
        self.critic.load_state_dict(torch.load('%s/%s_critic.pth' % (directory, filename)))