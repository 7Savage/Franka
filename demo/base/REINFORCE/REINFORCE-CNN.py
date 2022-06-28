import gym
import torch
import torch.nn.functional as F
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm
import rl_utils
import matplotlib.pyplot as plt
import matplotlib
from PIL import Image
import torchvision.transforms as T
from tensorboardX import SummaryWriter

learning_rate = 1e-3
num_episodes = 1000
hidden_dim = 128
gamma = 0.98
device = torch.device("cuda") if torch.cuda.is_available() else torch.device(
    "cpu")
writer = SummaryWriter()


def get_cart_location(screen_width):
    # 世界转屏幕系数 : world_width * scale = screen_width
    world_width = env.x_threshold * 2
    # 世界转屏幕系数 : world_width * scale = screen_width
    scale = screen_width / world_width
    # 世界中点在屏幕中间，所以偏移屏幕一半
    return int(env.state[0] * scale + screen_width / 2.0)  # MIDDLE OF CART


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


class PolicyNet(torch.nn.Module):
    def __init__(self, h, w,  outputs):
        super(PolicyNet, self).__init__()
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
        self.fc1 = torch.nn.Linear(outputs, action_dim)


    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.head(x.view(x.size(0), -1))

        return F.softmax(self.fc1(x), dim=1)


class REINFORCE:
    def __init__(self, screen_height, screen_width, action_dim, learning_rate, gamma,
                 device):
        self.action_dim = action_dim
        self.policy_net = PolicyNet(screen_height, screen_width, action_dim ).to(device)
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(),
                                          lr=learning_rate)  # 使用Adam优化器
        self.gamma = gamma  # 折扣因子
        self.device = device

    def take_action(self, state):  # 根据动作概率分布随机采样
        # state = torch.tensor([state], dtype=torch.float).to(self.device)
        probs = self.policy_net(state)
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()
        return action.item()

    def update(self, transition_dict):
        reward_list = transition_dict['rewards']
        state_list = transition_dict['states']
        action_list = transition_dict['actions']

        G = 0
        self.optimizer.zero_grad()
        for i in reversed(range(len(reward_list))):  # 从最后一步算起
            reward = reward_list[i]
            state = state_list[i]
            action = torch.tensor([action_list[i]]).view(-1, 1).to(self.device)
            log_prob = torch.log(self.policy_net(state).gather(1, action))
            G = self.gamma * G + reward
            loss = -log_prob * G  # 每一步的损失函数
            loss.backward()  # 反向传播计算梯度
        self.optimizer.step()  # 梯度下降


if __name__ == "__main__":
    env_name = "CartPole-v1"
    env = gym.make(env_name)
    env.seed(0)
    torch.manual_seed(0)
    state_dim = env.observation_space.shape[0]

    env.reset()
    init_screen = get_screen()
    _, _, screen_height, screen_width = init_screen.shape  # 得到画面的尺寸：宽高

    # 从gym行动空间中获取行动数量，动作空间是离散空间:0: 表示小车向左移动，1: 表示小车向右移动
    action_dim = env.action_space.n

    agent = REINFORCE(screen_height, screen_width, action_dim, learning_rate, gamma,
                      device)

    return_list = []
    for i in range(10):
        with tqdm(total=int(num_episodes / 10), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes / 10)):
                episode_return = 0
                transition_dict = {
                    'states': [],
                    'actions': [],
                    'next_states': [],
                    'rewards': [],
                    'dones': []
                }
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
                agent.update(transition_dict)
                if (i_episode + 1) % 10 == 0:
                    pbar.set_postfix({
                        'episode':
                            '%d' % (num_episodes / 10 * i + i_episode + 1),
                        'return':
                            '%.3f' % np.mean(return_list[-10:])
                    })

                    writer.add_scalar('REINFORCE-CNN ten episodes average rewards', np.mean(return_list[-10:]),
                                      (int)(num_episodes / 10 * i + i_episode + 1))
                pbar.update(1)
