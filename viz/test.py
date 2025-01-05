# %%
import importlib
import viz.spherical_utils as spherical_utils

importlib.reload(spherical_utils)

import numpy as np
import pyshtools
import matplotlib.pyplot as plt
from viz.spherical_utils import plot_function_on_sphere_plotly


# %%
def gaussian_on_sphere(theta, phi, theta0=np.pi / 2, phi0=0, sigma=0.4):
    """Create a Gaussian function on a sphere centered at (theta0, phi0)"""
    # Calculate the great circle distance
    cos_dist = np.sin(theta) * np.sin(theta0) * np.cos(phi - phi0) + np.cos(
        theta
    ) * np.cos(theta0)
    dist = np.arccos(np.clip(cos_dist, -1, 1))
    return np.exp(-(dist**2) / (2 * sigma**2))


# Create a grid
lmax = 100  # maximum spherical harmonic degree
grid = pyshtools.SHGrid.from_zeros(lmax=lmax)
nlat, nlon = grid.nlat, grid.nlon
theta = np.linspace(0, np.pi, nlat)
phi = np.linspace(0, 2 * np.pi, nlon)
phi_grid, theta_grid = np.meshgrid(phi, theta)

# Generate Gaussian data
gaussian_data = gaussian_on_sphere(theta_grid, phi_grid, sigma=0.03)
grid.data = gaussian_data

# Perform spherical harmonic analysis
coeffs = grid.expand()

# %%
# Create decayed coefficients
decay_factor = 0.5  # Controls how fast the coefficients decay
coeffs_decayed = pyshtools.SHCoeffs.from_zeros(lmax=lmax)

# Get the coefficient arrays
coeffs_array = coeffs.to_array()

# Apply exponential decay to coefficients
for l in range(lmax + 1):
    decay = np.exp(-l * decay_factor)  # Exponential decay with l
    for m in range(-l, l + 1):
        if m >= 0:
            # Set cosine coefficients (m >= 0)
            coeffs_decayed.set_coeffs(values=coeffs_array[0, l, m] * decay, ls=l, ms=m)
        else:
            # Set sine coefficients (m < 0)
            coeffs_decayed.set_coeffs(
                values=coeffs_array[1, l, abs(m)] * decay, ls=l, ms=m
            )

# Plot comparison
fig = plt.figure(figsize=(15, 5))

# Original
ax1 = fig.add_subplot(131)
img = grid.plot(show=False)
plt.title("Original Gaussian")

# Full reconstruction
ax2 = fig.add_subplot(132)
grid_recon_full = coeffs.expand()
img = grid_recon_full.plot(show=False)
plt.title(f"Full Reconstruction (l≤{lmax})")

# Decayed reconstruction
ax3 = fig.add_subplot(133)
grid_recon_decayed = coeffs_decayed.expand()
img = grid_recon_decayed.plot(show=False)
plt.title(f"Decayed Reconstruction (decay={decay_factor})")

plt.tight_layout()
plt.show()

# %%
# Create 3D Plotly visualizations
fig1 = plot_function_on_sphere_plotly(
    grid.data, theta_grid, phi_grid, title="Original Gaussian", colorbar_title="Voltage"
)
fig1.show()

fig2 = plot_function_on_sphere_plotly(
    grid_recon_full.data,
    theta_grid,
    phi_grid,
    title=f"Full Reconstruction (l≤{lmax})",
    colorbar_title="Voltage",
)
fig2.show()

fig3 = plot_function_on_sphere_plotly(
    grid_recon_decayed.data,
    theta_grid,
    phi_grid,
    title=f"Decayed Reconstruction (decay={decay_factor})",
)
fig3.show()
# %%
