import gym
import torch
import torch.nn.functional as F
import numpy as np
import torch.nn as nn
import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image

import torch.optim as optim
import torchvision.transforms as T

from tensorboardX import SummaryWriter

writer = SummaryWriter()

actor_lr = 1e-3
critic_lr = 1e-2
num_episodes = 500
hidden_dim = 128
gamma = 0.98
device = torch.device("cuda") if torch.cuda.is_available() else torch.device(
    "cpu")
PATH = 'a2c_cnn.pt'


class CNN(torch.nn.Module):
    def __init__(self, h, w, outputs):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)  # 设置第1个卷积层
        self.bn1 = nn.BatchNorm2d(16)  # 设置第1个卷积层的偏置
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)

        # 最后一个卷积层的feature map的尺寸
        # 线性输入连接的数量取决于conv2d层的输出，因此取决于输入图像的大小，因此请对其进行计算。
        def conv2d_size_out(size, kernel_size=5, stride=2):
            # - kernel_size 等价为 - (kernel_size - 1) - 1
            return (size - (kernel_size - 1) - 1) // stride + 1

        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))  # 宽
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))  # 高
        linear_input_size = convw * convh * 32
        # 全连接层，Linear 512节点 2节点
        self.head = nn.Linear(linear_input_size, outputs)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return self.head(x.view(x.size(0), -1))


class PolicyNet(torch.nn.Module):
    def __init__(self, h, w, outputs):
        super(PolicyNet, self).__init__()
        self.cnn = CNN(h, w, outputs)
        self.fc1 = torch.nn.Linear(outputs, action_dim)

    def forward(self, x):
        x = self.cnn(x)
        return F.softmax(self.fc1(x), dim=1)


class ValueNet(torch.nn.Module):
    def __init__(self, h, w, outputs):
        super(ValueNet, self).__init__()
        self.cnn = CNN(h, w, outputs)

    def forward(self, x):
        return self.cnn(x)


class ActorCritic:
    def __init__(self, screen_height, screen_width, action_dim, actor_lr, critic_lr,
                 gamma, device):

        self.actor = PolicyNet(screen_height, screen_width,
                               action_dim).to(device)
        self.critic = ValueNet(screen_height, screen_width, action_dim).to(device)
        # 网络优化器
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                 lr=critic_lr)
        self.gamma = gamma
        self.device = device

    def take_action(self, state):
        # state = torch.tensor([state], dtype=torch.float).to(self.device)
        probs = self.actor(state)
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()
        return action.item()

    def update(self, transition_dict, i_episode, i):
        states = torch.cat(transition_dict['states'])
        actions = torch.tensor(transition_dict['actions'], dtype=torch.int64).view(-1, 1).to(self.device)
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.cat(transition_dict['next_states'])
        dones = torch.tensor(transition_dict['dones'], dtype=torch.float).view(-1, 1).to(self.device)

        # 时序差分目标
        td_target = rewards + self.gamma * self.critic(next_states) * (1 - dones)
        td_delta = td_target - self.critic(states)  # 时序差分误差
        log_probs = torch.log(self.actor(states).gather(1, actions))
        # detach()方法切断td_delta的反向传播
        # 返回一个新的tensor，从当前计算图中分离下来的，但是仍指向原变量的存放位置,
        # 不同之处只是requires_grad为false，得到的这个tensor永远不需要计算其梯度，不具有grad
        actor_loss = torch.mean(-log_probs * td_delta.detach())
        # 均方误差损失函数
        critic_loss = torch.mean(
            F.mse_loss(self.critic(states), td_target.detach()))
        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        actor_loss.backward()  # 计算策略网络的梯度
        critic_loss.backward()  # 计算价值网络的梯度
        self.actor_optimizer.step()  # 更新策略网络的参数
        self.critic_optimizer.step()  # 更新价值网络的参数
        writer.add_graph(self.actor, states)
        if (i_episode + 1) % 10 == 0:
            episode = (int)(num_episodes / 10 * i + i_episode + 1)
            writer.add_scalar('A2C-CNN-ActorLoss', actor_loss,
                              episode)
            writer.add_scalar("A2C-CNN-CriticLoss", critic_loss, episode)



# 输入提取
# Compose把多个步骤整合到一起
resize = T.Compose([
    # 将tensor 或者 ndarray的数据转换为 PIL Image 类型数据
    # PIL(Python Image Library)是python的第三方图像处理库
    T.ToPILImage(),
    # 重置图像分辨率，interpolation 表示插值方法，默认为PIL.Image.BILINEAR
    # 图像还是太大，需要把图片转换为高40的图片
    T.Resize(40, interpolation=Image.CUBIC),
    # 将PIL图像转换为tensor图像，并且归一化至[0-1]
    T.ToTensor()])


def get_cart_location(screen_width):
    # 世界转屏幕系数 : world_width * scale = screen_width
    world_width = env.x_threshold * 2
    # 世界转屏幕系数 : world_width * scale = screen_width
    scale = screen_width / world_width
    # 世界中点在屏幕中间，所以偏移屏幕一半
    return int(env.state[0] * scale + screen_width / 2.0)  # MIDDLE OF CART


# gym要求的返回屏幕是400x600x3，但有时更大，如800x1200x3。 将其转换为torch order（CHW）。
# 截取游戏的屏幕，对游戏截图进行预处理用于做训练数据的状态
def get_screen():
    # HWC，channal在最后的维度上，transpose将第三位的channal移到第一位，HWC-->CHW
    screen = env.render(mode='rgb_array').transpose((2, 0, 1))
    # cart位于下半部分，因此不包括屏幕的顶部和底部
    _, screen_height, screen_width = screen.shape
    # 小车大概在高40%（400X0.4=160）到80%（400X0.8=320）之间，所以整个画面可以剪切一下
    screen = screen[:, int(screen_height * 0.4):int(screen_height * 0.8)]
    # 宽度只截取60%，左右各截30%
    view_width = int(screen_width * 0.6)
    cart_location = get_cart_location(screen_width)

    if cart_location < view_width // 2:
        # 太靠左了，左边没有30%空间，则从最左侧截取  [:half_view_width)
        slice_range = slice(view_width)
    elif cart_location > (screen_width - view_width // 2):
        # 太靠右了，同理 [-half_view_width:)
        slice_range = slice(-view_width, None)
    else:
        # 左右两侧都有空间，则截小车在中间 [-half_view_width: +half_view_width)
        slice_range = slice(cart_location - view_width // 2,
                            cart_location + view_width // 2)
    # 去掉边缘，使得我们有一个以cart为中心的方形图像，最后将图像X轴截了
    screen = screen[:, :, slice_range]
    # 转换为float类型，重新缩放，转换为torch张量
    # transpose函数，调整了维度，改变了数组的连续性，需要通过np.ascontiguousarray函数按行排列的方式重构数组排列，不需要进行内存拷贝
    # screen的格式是numpy数组，值范围[0, 255]，int8，PIL接受的是float32的tensor，值范围[0.0, 1.0]
    screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
    # 转换成张量类型
    screen = torch.from_numpy(screen)
    # 调整大小并在第一位添加batch维度（BCHW）并放进GPU
    # unsqueeze()的作用是在n维之前增加一个维度，这里是在0维之前增加一个维度
    # 因为pytorch.nn.Conv2d() 的输入形式为(N, C, Y, X)
    # N表示batch数
    # C表示channel数
    # Y，X表示图片的高和宽。
    return resize(screen).unsqueeze(0).to(device)


def train_on_policy_agent(env, agent, num_episodes):
    return_list = []
    for i in range(10):
        with tqdm(total=int(num_episodes / 10), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes / 10)):
                episode_return = 0
                transition_dict = {'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'dones': []}
                env.reset()
                last_screen = get_screen()
                current_screen = get_screen()
                state = current_screen - last_screen
                done = False
                while not done:
                    action = agent.take_action(state)
                    _, reward, done, _ = env.step(action)

                    # 观察新的状态
                    last_screen = current_screen
                    current_screen = get_screen()
                    next_state = current_screen - last_screen

                    transition_dict['states'].append(state)
                    transition_dict['actions'].append(action)
                    transition_dict['next_states'].append(next_state)
                    transition_dict['rewards'].append(reward)
                    transition_dict['dones'].append(done)
                    state = next_state
                    episode_return += reward
                return_list.append(episode_return)
                agent.update(transition_dict, i_episode, i)
                if (i_episode + 1) % 10 == 0:
                    pbar.set_postfix({'episode': '%d' % (num_episodes / 10 * i + i_episode + 1),
                                      'return': '%.3f' % np.mean(return_list[-10:])})
                    writer.add_scalar('ten episodes average rewards', np.mean(return_list[-10:]),
                                      (int)(num_episodes / 10 * i + i_episode + 1))

                pbar.update(1)
    writer.close()
    return return_list


if __name__ == "__main__":
    env_name = 'CartPole-v1'
    env = gym.make(env_name).unwrapped
    env.seed(0)
    torch.manual_seed(0)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    env.reset()
    init_screen = get_screen()
    _, _, screen_height, screen_width = init_screen.shape  # 得到画面的尺寸：宽高

    agent = ActorCritic(screen_height, screen_width, action_dim, actor_lr, critic_lr,
                        gamma, device)

    train_on_policy_agent(env, agent, num_episodes)
