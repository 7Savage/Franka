import random
import torch
from collections import namedtuple, deque

state_que = deque([], maxlen=4)

memory = deque([], maxlen=100)
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))
st1 = torch.rand(2, 2)
st2 = torch.rand(2, 2)
st3 = torch.rand(2, 2)
st4 = torch.rand(2, 2)

a1 = torch.ones(1)
a2 = torch.ones(1)
a3 = torch.ones(1)
a4 = torch.ones(1)

# 模拟截屏代码get_screen，并将其处理成(1,1,84,84)的格式，在本文中，我是用图像格式为2*2
nst1 = torch.rand(2, 2)  # unsqueeze(0)
nst1 = nst1.unsqueeze(0)
nst1 = nst1.unsqueeze(0)
nst2 = torch.rand(2, 2)
nst2 = nst2.unsqueeze(0)
nst2 = nst2.unsqueeze(0)
nst3 = torch.rand(2, 2)
nst3 = nst3.unsqueeze(0)
nst3 = nst3.unsqueeze(0)
nst4 = torch.rand(2, 2)
nst4 = nst4.unsqueeze(0)
nst4 = nst4.unsqueeze(0)

# 将相应的变量添加到Transition中
s1 = Transition(st1, a1, nst1, 5)
s2 = Transition(st2, a2, nst2, 4)
s3 = Transition(st3, a3, nst3, 2)
s4 = Transition(st4, a4, nst4, 3)
# 添加到state_que中
state_que.append(nst1)
state_que.append(nst2)
state_que.append(nst3)
state_que.append(nst4)
print('state_que', state_que)
# 转换成元组
print('转换成元组和拼接')
state = torch.cat(tuple(state_que), dim=1)
print('state', state)
print('statesize', state.size())

memory.append(s1)
memory.append(s2)
memory.append(s3)
memory.append(s4)

# print(memory)


m2 = random.sample(memory, 2)
print('m2', m2)
print()
batch = Transition(*zip(*m2))
print('zip*-----------------------')
print('batch:000', batch.state)
non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.state)), dtype=torch.bool)
print(non_final_mask)
state_batch = torch.cat(batch.next_state)
print('next_state_batch', state_batch)
print('state_batch_size = ', state_batch.size())
action_batch = torch.cat(batch.action)
print('action_batch', action_batch)
