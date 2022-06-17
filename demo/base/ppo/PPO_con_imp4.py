import gym
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import rl_utils
from tensorboardX import SummaryWriter
from tqdm import tqdm

"""
    在使用截断式目标函数的基础上，增加了一个价值函数的惩罚项和一个熵的惩罚项
    使用值的梯度更新策略网络
"""
writer = SummaryWriter()

class PolicyNetContinuous(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(PolicyNetContinuous, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        # self.rnn = LSTM.MyLSTM(state_dim, hidden_dim)
        # self.lstm = nn.LSTM(state_dim, hidden_dim)
        self.fc_mu = torch.nn.Linear(hidden_dim, action_dim)
        self.fc_std = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        # x_out, _ = self.lstm(x)
        mu = 2.0 * torch.tanh(self.fc_mu(x))
        std = F.softplus(self.fc_std(x)) #激活函数
        return mu, std  #输出动作a 和 分布 std


class ValueNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super(ValueNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)

def train_on_policy_agent(env, agent, num_episodes):
    return_list = []
    for i in range(10):
        with tqdm(total=int(num_episodes/10), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes/10)):
                episode_return = 0
                transition_dict = {'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'dones': []}
                state = env.reset()
                done = False
                while not done:
                    action = agent.take_action(state)
                    next_state, reward, done, _ = env.step(action)
                    transition_dict['states'].append(state)
                    transition_dict['actions'].append(action)
                    transition_dict['next_states'].append(next_state)
                    transition_dict['rewards'].append(reward)
                    transition_dict['dones'].append(done)
                    state = next_state
                    episode_return += reward
                return_list.append(episode_return)
                # mv_return = rl_utils.moving_average(return_list, 21)
                agent.update(transition_dict)
                if (i_episode+1) % 10 == 0:
                    writer.add_scalar('ppo_im', np.mean(return_list[-10:]), (int)(num_episodes / 10 * i + i_episode + 1))
                    # writer.add_scalar('ac', np.mean(mv_return[-10:]), (int)(num_episodes / 10 * i + i_episode + 1))
                    pbar.set_postfix({'episode': '%d' % (num_episodes / 10 * i + i_episode + 1),
                                      'return': '%.3f' % np.mean(return_list[-10:])})
                pbar.update(1)
    writer.close()
    return return_list

class PPOContinuous:
    ''' 处理连续动作的PPO算法 '''
    def __init__(self, state_dim, hidden_dim, action_dim, actor_lr, critic_lr,
                 lmbda, epochs, eps, gamma, device,value_freq,num_episodes,value_epoch):
        self.actor = PolicyNetContinuous(state_dim, hidden_dim,
                                         action_dim).to(device)
        self.critic = ValueNet(state_dim, hidden_dim).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                 lr=critic_lr)
        self.gamma = gamma
        self.lmbda = lmbda
        self.epochs = epochs
        self.eps = eps
        self.device = device
        self.value_freq = value_freq
        self.num_episodes = num_episodes
        self.value_epoch = value_epoch

    def take_action(self, state):
        state = torch.tensor([state], dtype=torch.float).to(self.device)
        mu, sigma = self.actor(state)
        action_dist = torch.distributions.Normal(mu, sigma)
        action = action_dist.sample()  ## 就是直接在定义的正态分布（均值为mean，标准差std是１）上采样，生成数据
        return [action.item()]

    def update(self, transition_dict):
        states = torch.tensor(transition_dict['states'],
                              dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions'],
                               dtype=torch.float).view(-1, 1).to(self.device)
        rewards = torch.tensor(transition_dict['rewards'],
                               dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'],
                                   dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'],
                             dtype=torch.float).view(-1, 1).to(self.device)
        rewards = (rewards + 8.0) / 8.0  # 和TRPO一样,对奖励进行修改,方便训练

        td_target = rewards + self.gamma * self.critic(next_states) * (1 -
                                                                       dones)
        td_delta = td_target - self.critic(states)  #r+V(st+1)-V(st) TD误差
        advantage = rl_utils.compute_advantage(self.gamma, self.lmbda,
                                               td_delta.cpu()).to(self.device)  #GAE估计
        advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

        mu, std = self.actor(states)  #使用策略网络，输出动作a 和 分布
        action_dists = torch.distributions.Normal(mu.detach(), std.detach())
        # 动作是正态分布
        old_log_probs = action_dists.log_prob(actions) #旧策略动态分布函数

        for _ in range(self.epochs):
            mu, std = self.actor(states)
            action_dists = torch.distributions.Normal(mu, std)
            log_probs = action_dists.log_prob(actions) #当前策略动作分布
            ratio = torch.exp(log_probs - old_log_probs) ##实现以e为底的指数 log_probs/old_probs 的比率
            surr1 = ratio * advantage  #目标函数中的第一项PPO_con.py
            surr2 = torch.clamp(ratio, 1 - self.eps, 1 + self.eps) * advantage  ##目标函数中的第二项
            # actor_loss = torch.mean(-torch.min(surr1, surr2))  ##目标函数

            entropy = torch.sum(action_dists.entropy(), dim=-1)
            S = entropy.mean()  # 熵惩罚项
            actor_loss_CLIP = torch.mean(torch.min(surr1, surr2))  ##目标函数

            #增加优势损失函数
            adv = rewards + self.gamma * self.critic(next_states)
            adv_loss = 0.5 * (adv - advantage).pow(2).mean()
            actor_loss = -(actor_loss_CLIP - adv_loss + beta * S)  # 目标函数

            self.actor_optimizer.zero_grad()
            # self.critic_optimizer.zero_grad()
            actor_loss.backward()
            # critic_loss.backward()
            self.actor_optimizer.step()
            # self.critic_optimizer.step()

            if self.num_episodes % self.value_freq == 0:
                for _ in range(self.value_epoch):
                    mu, std = self.actor(states)
                    # action_dists = torch.distributions.Normal(mu, std)

                    critic_loss = torch.mean(
                        F.mse_loss(self.critic(states), td_target.detach()))

                    self.critic_optimizer.zero_grad()
                    critic_loss.backward()
                    self.critic_optimizer.step()

            # self.actor_optimizer.zero_grad()
            # self.critic_optimizer.zero_grad()
            # actor_loss.backward()
            # critic_loss.backward()
            # # torch.nn.utils.clip_grad_norm_(PolicyNetContinuous.parameters(), 10.0)
            # self.actor_optimizer.step()
            # self.critic_optimizer.step()

if __name__ == "__main__":
    value_epoch = 1
    value_freq = 1
    beta = .01
    actor_lr = 1e-4
    critic_lr = 5e-3
    num_episodes = 3000
    hidden_dim = 128
    gamma = 0.9
    lmbda = 0.9
    epochs = 10
    eps = 0.2
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device(
        "cpu")

    env_name = 'Pendulum-v1'
    env = gym.make(env_name)
    env.seed(0)
    torch.manual_seed(0)  # 设置 (CPU) 生成随机数的种子，并返回一个torch.Generator对象。设置种子的用意是一旦固定种子，后面依次生成的随机数其实都是固定的。

    state_dim = env.observation_space.shape[0]  # 3
    action_dim = env.action_space.shape[0]  # 连续动作空间1

    agent = PPOContinuous(state_dim, hidden_dim, action_dim, actor_lr, critic_lr,
                          lmbda, epochs, eps, gamma, device, value_freq, num_episodes, value_epoch)

    return_list = train_on_policy_agent(env, agent, num_episodes)



    # episodes_list = list(range(len(return_list)))
    # plt.plot(episodes_list, return_list)
    # plt.xlabel('Episodes')
    # plt.ylabel('Returns')
    # plt.title('PPO on {}'.format(env_name))
    # plt.show()
    #
    # mv_return = rl_utils.moving_average(return_list, 21)
    # plt.plot(episodes_list, mv_return)
    # plt.xlabel('Episodes')
    # plt.ylabel('Returns')
    # plt.title('PPO on {}'.format(env_name))
    # plt.show()