import gym
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

BATCH_SIZE = 128  # 从transition提取样本的批次大小
GAMMA = 0.999  # 衰减系数
EPS_START = 0.9  # 贪婪参数初始值
EPS_END = 0.05  # 贪婪参数最小值
EPS_DECAY = 200  # 贪婪参数变化次数
TARGET_UPDATE = 10  # target net更新次数
steps_done = 0  # 记录全局步骤


class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    # *args 不知道传入参数的数量、名称
    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


# Q_网络
class DQN(nn.Module):

    # h 图片高度
    # w 图片宽度
    def __init__(self, h, w, outputs):
        super(DQN, self).__init__()
        # in_channels=3  3通道
        # out_channels=16 16通道
        # kernel_size=5 5*5的卷积核，相当于过滤器
        # stride=2 卷积核在图上滑动，每隔一个扫一次
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

    # 使用一个元素调用以确定下一个操作，或在优化期间调用batch。返回tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        # x.view(x.size(0), -1)这句话是说将第三次卷积的输出拉伸为一行
        # view函数相当于resize函数，改变张量维度，主义保持元素个数总数不变
        # 而-1指在不告诉函数有多少列的情况下，根据原tensor数据和batchsize自动分配列数。
        return self.head(x.view(x.size(0), -1))


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


def select_action(state):
    global steps_done
    sample = random.random()  # 产生 0 到 1 之间的随机浮点数
    # 选择随机操作的概率将从EPS_START开始，并将以指数方式向EPS_END衰减。EPS_DECAY控制衰减的速度
    # 阈值 eps_threshold 由0.9 不断下降
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
                    math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:  # 判断是随即动作还是最优动作
        with torch.no_grad():  # torch.no_grad()一般用于神经网络的推理阶段, 表示张量的计算过程中无需计算梯度
            # t.max(1)将返回每行的最大列值。
            # 最大结果的第二列是找到最大元素的索引，因此我们选择具有较大预期奖励的行动。
            # 等价于 return 0 if value[0] > value[1] else 1
            # max(1)[1] 只取了索引值，也可以用 max(1).indices。view(1,1)
            # 把数值做成[[1]] 的二维数组形式。
            # 为何返回一个二维 [[1]] ? 这是因为后面要把所有的state用torch.cat() 合成batch
            return policy_net(state).max(1)[1].view(1, 1)  # 使用最优动作
    else:
        # 到后期会越来越趋向于（探索），u而就是随机选择一个动作。
        return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)


def plot_durations():
    plt.figure(2)
    plt.clf()  # 清除当前图形及其所有轴，但保持窗口打开，以便可以将其重新用于其他绘图。有了这个再次运行就不要关掉所有figure了
    durations_t = torch.tensor(episode_durations, dtype=torch.float)  # 转换成张量
    plt.title('Training...')  # 图的名字
    plt.xlabel('Episode')  # x轴坐标名
    plt.ylabel('Duration')  # y轴坐标名
    plt.plot(durations_t.numpy())  # 画图
    # 取100个episode的平均值并绘制它们
    if len(durations_t) >= 100:
        # dimension=0->待切片的维度
        # size=100->大小
        # step=1->步长
        # x = tensor([1, 2, 3, 4, 5])
        # x.unfold(0, 2, 1) -> [[1 ,2], [2, 3], [3, 4], [4, 5]]
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        # 不到100步的话，就用0代替
        means = torch.cat((torch.zeros(99), means))
        # 绘制黄色平均曲线
        plt.plot(means.numpy())

    plt.pause(0.001)  # 每隔0.001秒更新图表
    if is_ipython:
        display.clear_output(wait=True)
        display.display(plt.gcf())


def optimize_model():
    # 只有经验池的数量超过BATCH_SIZE才会采样
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)  # 从记忆池中随即采集BATCH_SIZE个样本

    # zip表示交叉元素，*号代表拆分
    # *transitions 拆分成Transition对象
    # zip(*transitions) 将Transition对象相应的属性打包在一起
    # *zip(*transitions) 将组合后的4个属性列拆分成四个元组对象
    batch = Transition(*zip(*transitions))

    # 计算非最终状态的掩码并连接batch元素（最终状态将是模拟结束后的状态）
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), device=device, dtype=torch.uint8)
    # 首先分析map()函数，lambda是一个简单的函数。把transition中的next_state赋值给s。
    # tuple()将状态转换为元组，元组是无法修改的
    non_final_next_states = torch.cat([s for s in batch.next_state
                                       if s is not None])

    state_batch = torch.cat(batch.state)

    action_batch = torch.cat(batch.action)

    reward_batch = torch.cat(batch.reward)

    # 计算Q(s_t，a) - 模型计算Q(s_t)，然后我们选择所采取的动作列。
    # 这些是根据policy_net对每个batch状态采取的操作
    # policy_net(state_batch)输出所有动作，gather后选择当前action
    # 类似与Qtable中的查表。计算的是Q值
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # 计算所有下一个状态的V(s_{t+1})
    # non_final_next_states的操作的预期值是基于“较旧的”target_net计算的;
    # 用max(1)[0]选择最佳奖励。这是基于掩码合并的，这样我们就可以得到预期的状态值，或者在状态是最终的情况下为0。
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
    # 计算预期的Q值
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # 计算Huber损失,unsqueeze(1)增加维度（1表示，在第二个位置增加一个维度）
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

    # 优化模型，把梯度置零，也就是把loss关于weight的导数变成0
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        # 可能出现梯度爆炸，训练时，加上梯度截断,param.grad.data.clamp_(-grad_clip, grad_clip)
        param.grad.data.clamp_(-1, 1)
    optimizer.step()

def plot_screen(screen):
    # 想要plt get_screen()返回的东西，先要将其放回到CPU，然后去掉batch，调换方向把颜色放到后边，再转换为numpy
    plt.imshow(screen.cpu().squeeze(0).permute(1, 2, 0).numpy(),
               interpolation='none')
    plt.title('Example extracted screen')
    plt.show()


if __name__ == '__main__':
    # unwrapped可以得到最原始的类，该类不受step次数限制
    env = gym.make('CartPole-v1').unwrapped

    # set up matplotlib
    is_ipython = 'inline' in matplotlib.get_backend()
    if is_ipython:
        from IPython import display

    # 显示模式转换为交互（interactive）模式
    # matplotlib的显示模式默认为阻塞（block）模式（即：在plt.show()之后，程序会暂停到那儿，并不会继续执行下去
    plt.ion()

    plt.figure()

    # if gpu is to be used
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 定义了一个Transition的对象，包含四种属性
    # 状态是屏幕差异图像
    Transition = namedtuple('Transition',
                            ('state', 'action', 'next_state', 'reward'))

    env.reset()

    episode_durations = []

    # 获取屏幕大小，以便我们可以根据AI gym返回的形状正确初始化图层。
    # 此时的典型尺寸接近3x40x90
    # 这是get_screen（）中的限幅和缩小渲染缓冲区的结果
    init_screen = get_screen()
    _, _, screen_height, screen_width = init_screen.shape  # 得到画面的尺寸：宽高

    # 从gym行动空间中获取行动数量，动作空间是离散空间:0: 表示小车向左移动，1: 表示小车向右移动
    n_actions = env.action_space.n

    policy_net = DQN(screen_height, screen_width, n_actions).to(device)
    target_net = DQN(screen_height, screen_width, n_actions).to(device)
    target_net.load_state_dict(policy_net.state_dict())  # 初始阶段target net和main net是一样的参数
    target_net.eval()  # 表示步更新，只评估输出。

    # 使用RMSprop优化网络
    optimizer = optim.RMSprop(policy_net.parameters())
    # optimizer = optim.Adam(policy_net.parameters())
    # 定义经验池的容量capacity
    memory = ReplayMemory(10000)

    # 开始训练
    num_episodes = 10  # 迭代次数
    for i_episode in range(num_episodes):
        # 初始化环境和状态
        env.reset()

        last_screen = get_screen()
        plot_screen(last_screen)
        current_screen = get_screen()
        plot_screen(current_screen)
        state = current_screen - last_screen

        for t in count():
            # 选择动作并执行
            action = select_action(state)
            # 与环境交互获取奖励和是否终止，将选择的action输入给env，env 按照这个动作走一步进入下一个状态
            _, reward, done, _ = env.step(action.item())
            reward = torch.tensor([reward], device=device)

            # 观察新的状态
            last_screen = current_screen
            current_screen = get_screen()
            if not done:  # 如果没有结束
                # 两次屏幕截图的差别来训练网络
                next_state = current_screen - last_screen
            else:
                next_state = None  # 没有下一个状态，表示是死亡

            # 在记忆中存储过渡
            memory.push(state, action, next_state, reward)

            # 移动到下一个状态
            state = next_state

            # 执行优化的一个步骤（在目标网络上）
            optimize_model()
            if done:
                episode_durations.append(t + 1)  # 将过程数据添加到列表中
                plot_durations()  # 画图
                break
        # 更新目标网络，复制DQN中的所有权重和偏差
        if i_episode % TARGET_UPDATE == 0:  # 每10个episode更新一次
            target_net.load_state_dict(policy_net.state_dict())
            #torch.save(policy_net.state_dict(), 'weights/policy_net_weights_{0}.pth'.format(i_episode))  # 保存模型

    print('Complete')
    env.render()
    env.close()
    #torch.save(policy_net.state_dict(), 'weights/policy_net_weights.pth')
    # 显示前关掉交互模式，如果不加，界面会一闪而过，不会停留
    plt.ioff()
    plt.show()
