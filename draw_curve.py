import matplotlib.pyplot as plt
import json
import os

with open("./loss.json", "r") as f:
    data = json.load(f)

fig = plt.figure()

x_epoch = [i for i in range(200)]
# plt.plot(x_epoch, data['train'][-200:], label='train')
plt.plot(x_epoch, data['val'][-200:], label='val')
fig.legend()
fig.savefig(os.path.join('./check.jpg'))
plt.clf()
