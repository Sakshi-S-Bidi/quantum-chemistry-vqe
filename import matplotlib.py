import matplotlib.pyplot as plt

# read H2 data
with open("energy_log.txt") as f:
    h2_data = [float(line.strip()) for line in f]

# read LiH data
with open("lih_energy_log.txt") as f:
    lih_data = [float(line.strip()) for line in f]

# plot both
plt.plot(h2_data, label="H2")
plt.plot(lih_data, label="LiH")

plt.xlabel("Iteration")
plt.ylabel("Energy")
plt.title("VQE Convergence Comparison")
plt.legend()
plt.show()




