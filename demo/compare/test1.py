import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

sns.set_style('darkgrid')
# plt.style.use('ggplot')
# plt.rcParams['font.sans-serif'] = ['SimHei']
# plt.rcParams['axes.unicode_minus'] = False

data = {0: pd.read_csv("DoubleDQN.csv", usecols=['Step', 'Value']),
        1: pd.read_csv("DuelingDQN.csv", usecols=['Step', 'Value']),
        2: pd.read_csv("TRPO.csv", usecols=['Step', 'Value']), 3: pd.read_csv("PPO.csv", usecols=['Step', 'Value']),
        4: pd.read_csv("DDPG.csv", usecols=['Step', 'Value']), 5: pd.read_csv("SAC.csv", usecols=['Step', 'Value'])}

label = ['DoubleDQN', 'DuelingDQN', 'TRPO', 'PPO', "DDPG", "SAC"]
df = []
for i in range(len(data)):
    df.append(pd.DataFrame(data[i]))
    df[i]['algo'] = label[i]
df = pd.concat(df)
df.index = range(len(df))

sns.lineplot(x='Step', y='Value', hue="algo", data=df, style="algo",err_style="band")
plt.xlabel('Step', fontsize=12)
plt.ylabel('Ten episodes average rewards', fontsize=12)
plt.show()
