import gym
import gym_panda
import os
import pybullet as p
import pybullet_data
import math




if __name__=='__main__':
    env = gym.make('panda-v0')
    env.reset()

    for _ in range(1000):
        env.render()
        env.step(env.action_space.sample()) # take a random action
    env.close()