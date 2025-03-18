from matplotlib import pyplot as plt

filePath = r"D:\PROGRAMOWANIE\Studia\Sieci Neuronowe 2\SelfOrganizingMap\SelfOrganizingMap\accuracy.txt"

with open(filePath, "r") as f:
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
