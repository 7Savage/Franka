import gym
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from tensorboardX import SummaryWriter

writer = SummaryWriter()

actor_lr = 1e-3
critic_lr = 1e-2
num_episodes = 2000
hidden_dim = 128
gamma = 0.98
device = torch.device("cuda") if torch.cuda.is_available() else torch.device(
    "cpu")
PATH = 'a2c.pt'


class PolicyNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(PolicyNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return F.softmax(self.fc2(x), dim=1)


class ValueNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super(ValueNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class ActorCritic:
    def __init__(self, state_dim, hidden_dim, action_dim, actor_lr, critic_lr,
                 gamma, device):
        # 策略网络
        self.actor = PolicyNet(state_dim, hidden_dim, action_dim).to(device)
        self.critic = ValueNet(state_dim, hidden_dim).to(device)  # 价值网络
        # 策略网络优化器
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                 lr=critic_lr)  # 价值网络优化器
        self.gamma = gamma
        self.device = device

    def take_action(self, state):
        state = torch.tensor([state], dtype=torch.float).to(self.device)
        probs = self.actor(state)
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()
        return action.item()


if __name__ == "__main__":
    env_name = 'CartPole-v1'
    env = gym.make(env_name)
    env.seed(0)
    torch.manual_seed(0)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent = ActorCritic(state_dim, hidden_dim, action_dim, actor_lr, critic_lr,
                        gamma, device)

    #train_on_policy_agent(env, agent, num_episodes)
    checkpoint = torch.load(PATH)
    agent.actor.load_state_dict(checkpoint['actor_net_state_dict'])
    agent.critic.load_state_dict(checkpoint['critic_net_state_dict'])

    return_list = []
    for i in range(10):
        with tqdm(total=int(num_episodes / 10), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes / 10)):
                episode_return = 0
                transition_dict = {'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'dones': []}
                state = env.reset()
                done = False
                while not done:
                    action = agent.take_action(state)
                    next_state, reward, done, _ = env.step(action)
                    env.render()
                    state = next_state
                    episode_return += reward
                return_list.append(episode_return)
                if (i_episode + 1) % 10 == 0:
                    pbar.set_postfix({'episode': '%d' % (num_episodes / 10 * i + i_episode + 1),
                                      'return': '%.3f' % np.mean(return_list[-10:])})
                    writer.add_scalar('A2C-ten episodes average rewards', np.mean(return_list[-10:]),
                                      (int)(num_episodes / 10 * i + i_episode + 1))
                pbar.update(1)

    # evaluate the model
    # for i_episode in range(num_episodes):
    #     env.reset()
    #     state = env.step()
    #     for t in count():
    #         # Select and perform an action
    #         action = agent.actor.(stacked_states_t).max(1)[1].view(1, 1)
    #         _, reward, done, _ = env.step(action.item())
    #         # Observe new state
    #         next_state = get_screen()
    #         stacked_states.append(next_state)
    #         if done:
    #             break
    #     print("Episode: {0:d}, reward: {1}".format(i_episode + 1, reward), end="\n")

    # episodes_list = list(range(len(return_list)))
    # plt.plot(episodes_list, return_list)
    # plt.xlabel('Episodes')
    # plt.ylabel('Returns')
    # plt.title('Actor-Critic on {}'.format(env_name))
    # plt.show()

    # mv_return = rl_utils.moving_average(return_list, 9)
    # plt.plot(episodes_list, mv_return)
    # plt.xlabel('Episodes')
    # plt.ylabel('Returns')
    # plt.title('Advantage-Actor-Critic on {}'.format(env_name))
    # plt.show()
