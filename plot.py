import pandas as pd
import matplotlib.pyplot as plt
#%matplotlib inline

# monitor.csvの読み込み
df = pd.read_csv('./logs/monitor.csv', names=['r', 'l','t'])
df = df.drop(range(2))

# 報酬のプロット
x = range(len(df['r']))
y = df['r'].astype(float)
plt.plot(x, y)
plt.xlabel('episode')
plt.ylabel('reward')
plt.show()
