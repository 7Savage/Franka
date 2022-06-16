from numpy import array  # 从numpy中引入array，为创建矩阵做准备
import numpy as np
import torch

import torch as t

a = t.arange(0, 16).view(4, 4)
print(a)
'''
a = tensor([[ 0,  1,  2,  3],
            [ 4,  5,  6,  7],
            [ 8,  9, 10, 11],
            [12, 13, 14, 15]])
'''
index = t.LongTensor([[3, 2, 1, 0]])
# 0，表示行号变动。1  就是列号变动
print(a.gather(0, index))
'''
a = tensor([[12,  9,  6,  3]])
'''
# 为什么会是这四个数：大家看下面的四个数，说好的行号变动，也就是第1个[]中的数值是我们的index数值，
# 而第二个[]中的数值是从0,1,2,3逐个增加的。
# a[3][0] = 12 ；a[2][1] = 9； a[1][2] = 6； a[0][3] = 3

print(a.gather(1, index))

'''
tensor([[ 3],
        [ 6],
        [ 9],
        [12]])
'''
# a[0][3] = 3; a[1][2] = 6; a[2][1] = 9; a[3][0] = 12
# 再看这个，就是从第一个[]中的数值是从0,1,2,3逐个增加的。二变动的则是后面的
##################
# 上面举例子用的index是[3,2,1,0]，是为了告诉大家index的索引值的最大值是3，最小值是0，这个需要满足张量的size
# 其实真正的情况可能是下面这样index = t.LongTensor([[1,2,1,3]])
# 然后，我们[1,2,1,3]做一次索引试试
##################
index = t.LongTensor([[1, 2, 1, 3]])
b = a.gather(0, index)
print(b)
'''
tensor([[ 0,  1,  2,  3],
        [ 4,  5,  6,  7],
        [ 8,  9, 10, 11],
        [12, 13, 14, 15]])
tensor([[ 4,  9,  6, 15]])
我们按照上面的要求，即第一个索引值[]中的数值是变动的，而第二个[]中的数值是不变的，即[0,1,2,3]
那么这四个数分别是a[1][0] = 4 ；a[2][1] = 9； a[1][2] = 6； a[3][3] = 15
'''