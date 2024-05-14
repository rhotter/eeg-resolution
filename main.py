import numpy as np
import matplotlib.pyplot as plt


def compute_eeg_transfer_function(conductivities, radii, l):
    N = len(radii)

    def compute_gamma(i, zetta_ip1):
        sigma = conductivities[i] / conductivities[i + 1]
        return (sigma - zetta_ip1) / ((l + 1) * sigma + zetta_ip1)

    def compute_zetta(i, gamma_i):
        r_ratio = radii[i - 1] / radii[i]
        num = l * r_ratio**l - (l + 1) * gamma_i * 1 / r_ratio ** (l + 1)
        denom = r_ratio**l + gamma_i * (1 / r_ratio ** (l + 1))
        return num / denom

    def compute_A_B(i, A_im1, B_im1, gamma_i):
        r_ratio = radii[i - 1] / radii[i]
        num = A_im1 + B_im1
        denom = (r_ratio) ** l + gamma_i * (1 / r_ratio ** (l + 1))
        A = num / denom
        B = gamma_i * A
        return A, B

    gamma = np.zeros(N)
    zetta = np.zeros(N)
    A = np.zeros(N)
    B = np.zeros(N)

    # first compute gamma and zetta
    for i in reversed(range(N)):
        if i == N - 1:
            gamma[i] = l / (l + 1)
        else:
            gamma[i] = compute_gamma(i, zetta[i + 1])
        zetta[i] = compute_zetta(i, gamma[i])

    A[0] = 1
    B[0] = 1
    for i in range(1, N):
        A[i], B[i] = compute_A_B(i, A[i - 1], B[i - 1], gamma[i])

    H_l = (A[-1] + B[-1]) / (A[0] + B[0])
    return H_l


radii = [7.9, 8.0, 8.6, 9.1]
conductivities = [1, 5, 1 / 15, 1]
l_max = 100
H = np.zeros(l_max)
for l in range(l_max):
    H[l] = compute_eeg_transfer_function(conductivities, radii, l)

plt.plot(H)
plt.xlabel("Spherical harmonics degree ($l$)")
plt.ylabel("Transfer function ($H_l$)")
plt.yscale("log")
plt.title("EEG Transfer Function")
plt.show()
