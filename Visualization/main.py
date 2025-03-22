import numpy as np
from matplotlib import pyplot as plt, colors
from matplotlib import animation as animation
import os

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
accuracyFilePath = os.path.join(parent_dir, 'SelfOrganizingMap', 'accuracy.txt')
with open(accuracyFilePath, "r") as f:
    text = f.read()

lines = text.splitlines()

epochs = []
accuracies = []

for line in lines:
    parts = line.split(',')
    if len(parts) == 2:
        epochs.append(int(parts[0]))
        accuracies.append(float(parts[1]))

plt.plot(epochs, accuracies)
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Epoch vs Accuracy')
plt.show()

labelFilePath = os.path.join(parent_dir, 'SelfOrganizingMap', 'labels.txt')
with open(labelFilePath, "r") as f:
    text = f.read()
lines = text.splitlines()
epochs = []
maps = []
current_map = []
for line in lines:
    line = line.strip()
    if not line:
        continue
    if line.startswith('epoch:'):
        epochs.append(int(line.split(':')[1]))
    elif line == ';':
        maps.append(current_map)
        current_map = []
    else:
        row = [int(x) for x in line.split()]
        current_map.append(row)


cmap = colors.ListedColormap(['red', 'blue', 'green', 'yellow'])
bounds = [-1, 0, 1, 2, 3]
norm = colors.BoundaryNorm(bounds, cmap.N)

fig, ax = plt.subplots()
plt.xlabel('Column')
plt.ylabel('Row')

def animate_f(i):
    ax.set_title(f'Epoch: {i + 1}')
    current_map = np.array(maps[i])

    im.set_data(current_map)
    return [im]

current_map = np.array(maps[0])
im = ax.imshow(current_map, cmap=cmap, norm=norm)

cbar = fig.colorbar(im, ax=ax, orientation='vertical')
cbar.set_ticks([-1, 0, 1, 2])
cbar.set_ticklabels(['-1 (Red)', '0 (Blue)', '1 (Green)', '2 (Yellow)'])
cbar.set_label('Values')

ani = animation.FuncAnimation(fig, animate_f, frames=len(maps), interval=200, blit=False)

plt.show()