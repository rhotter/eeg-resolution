import numpy as np
import plotly.graph_objects as go
import pyshtools


def gaussian_on_sphere(theta, phi, theta0=0, phi0=0, sigma=1):
    """
    Calculate Gaussian function on a sphere
    theta, phi: angular coordinates on sphere
    theta0, phi0: center of the Gaussian
    sigma: width of the Gaussian
    """
    # Calculate great-circle distance
    cos_distance = np.sin(theta) * np.sin(theta0) * np.cos(phi - phi0) + np.cos(
        theta
    ) * np.cos(theta0)
    # Clip to avoid numerical errors
    cos_distance = np.clip(cos_distance, -1, 1)
    distance = np.arccos(cos_distance)

    return np.exp(-(distance**2) / (2 * sigma**2))


def decompose_spherical(func, resolution=100, l_max=50, **func_kwargs):
    """
    Decompose an arbitrary function on a sphere into spherical harmonics using pyshtools.
    Returns the spherical harmonic coefficients up to l_max.

    Parameters
    ----------
    func : callable
        Function that takes theta, phi as arguments and returns values on sphere
    resolution : int
        Number of points to sample on sphere
    l_max : int
        Maximum degree of spherical harmonics to compute
    **func_kwargs : dict
        Additional keyword arguments to pass to func
    """
    # Create meshgrid
    theta = np.linspace(0, np.pi, resolution)
    phi = np.linspace(0, 2 * np.pi, resolution)
    theta, phi = np.meshgrid(theta, phi)

    # Evaluate the function on the sphere
    f_vals = func(theta, phi, **func_kwargs)

    # Convert to grid that pyshtools expects (latitude from 90 to -90)
    lats = np.linspace(90, -90, resolution)
    lons = np.linspace(0, 360, resolution, endpoint=False)

    # Rearrange data to match pyshtools grid format
    grid = f_vals.T  # pyshtools expects (lat, lon) ordering

    # Create a pyshtools grid object
    grid_obj = pyshtools.SHGrid.from_array(grid, grid="DH")

    # Expand into spherical harmonics
    coeffs = grid_obj.expand()

    # Truncate to desired l_max if needed
    if l_max is not None and l_max < coeffs.lmax:
        coeffs = coeffs.truncate(lmax=l_max)

    return coeffs


def recompose_spherical(coeffs, theta, phi):
    """
    Recompose function from spherical harmonic coefficients using pyshtools.

    Parameters
    ----------
    coeffs : pyshtools.SHCoeffs
        Spherical harmonic coefficients
    theta : array-like
        Colatitude values (0 to pi)
    phi : array-like
        Longitude values (0 to 2pi)
    """
    # Convert input coordinates to grid that pyshtools expects
    resolution = len(theta)

    # Synthesize the function on a regular grid
    grid = coeffs.expand()

    # Get the values in the original coordinate system
    f_recomposed = grid.to_array()

    # Interpolate to the requested theta-phi grid if necessary
    if theta.ndim == 2 and phi.ndim == 2:
        from scipy.interpolate import RectBivariateSpline

        lats = np.linspace(90, -90, resolution)
        lons = np.linspace(0, 360, resolution, endpoint=False)

        # Create interpolator
        interp = RectBivariateSpline(lats, lons, f_recomposed)

        # Convert theta, phi to lat, lon
        lat_grid = 90 - np.degrees(theta)
        lon_grid = np.degrees(phi) % 360

        # Interpolate
        f_recomposed = interp(lat_grid, lon_grid, grid=False)

    return f_recomposed


def plot_function_on_sphere(
    function_values,
    theta,
    phi,
    ax=None,
    title=None,
    colorbar=True,
    rstride=1,
    cstride=1,
):
    """
    Plot a function on a sphere using matplotlib.

    Parameters:
    -----------
    function_values : 2D array
        Values of the function to plot on the sphere
    theta : 2D array
        Theta coordinates (meshgrid)
    phi : 2D array
        Phi coordinates (meshgrid)
    ax : matplotlib axis, optional
        Axis to plot on. If None, current axis is used
    title : str, optional
        Title for the plot
    colorbar : bool, optional
        Whether to add a colorbar
    rstride : int, optional
        Row stride for surface plot (controls mesh resolution)
    cstride : int, optional
        Column stride for surface plot (controls mesh resolution)

    Returns:
    --------
    ax : matplotlib axis
        The axis with the plot
    surface : matplotlib surface
        The plotted surface
    """
    import matplotlib.pyplot as plt

    if ax is None:
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection="3d")

    # Convert to Cartesian coordinates
    r = 1  # radius of sphere
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)

    # Normalize values for colormap
    normalized_values = (function_values - function_values.min()) / (
        function_values.max() - function_values.min()
    )

    # Plot the surface with specified stride
    surface = ax.plot_surface(
        x,
        y,
        z,
        facecolors=plt.cm.viridis(normalized_values),
        alpha=0.9,
        rstride=rstride,
        cstride=cstride,
        antialiased=True,
    )

    # Add colorbar if requested
    if colorbar:
        m = plt.cm.ScalarMappable(cmap=plt.cm.viridis)
        m.set_array(function_values)
        plt.colorbar(m, ax=ax, label="Function Value")

    # Set equal aspect ratio and labels
    ax.set_box_aspect([1, 1, 1])
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    if title:
        ax.set_title(title)

    return ax, surface


def plot_function_on_sphere_plotly(
    function_values, theta, phi, title=None, colorbar_title="Function Value"
):
    """
    Plot a function on a sphere using Plotly.

    Parameters:
    -----------
    function_values : 2D array
        Values of the function to plot on the sphere
    theta : 2D array
        Theta coordinates (meshgrid)
    phi : 2D array
        Phi coordinates (meshgrid)
    title : str, optional
        Title for the plot
    colorbar_title : str, optional
        Title for the colorbar. Defaults to "Function Value"

    Returns:
    --------
    fig : plotly.graph_objects.Figure
        The plotly figure object
    """
    # Convert to Cartesian coordinates
    r = 1  # radius of sphere
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)

    # Create the surface plot
    fig = go.Figure(
        data=[
            go.Surface(
                x=x,
                y=y,
                z=z,
                surfacecolor=function_values,
                colorscale="viridis",
                showscale=True,
                colorbar=dict(title=colorbar_title),
            )
        ]
    )

    # Update the layout
    fig.update_layout(
        title=title,
        scene=dict(
            aspectmode="data", xaxis_title="X", yaxis_title="Y", zaxis_title="Z"
        ),
        width=800,
        height=800,
    )

    return fig
