import matplotlib.pyplot as plt

energies = [-0.52, -0.31, -0.28, -0.83, -0.38]

plt.bar(range(len(energies)), energies)
plt.xlabel("Candidate Index")
plt.ylabel("Energy")
plt.title("Drug Candidate Screening")
plt.show()
