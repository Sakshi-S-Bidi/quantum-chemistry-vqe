import matplotlib.pyplot as plt

processes = [1, 2, 4]
times = [0.30, 0.32, 0.23]  # replace with your exact values

plt.plot(processes, times, marker='o')
plt.xlabel("Number of Processes")
plt.ylabel("Execution Time (s)")
plt.title("Scaling Performance")
plt.show()