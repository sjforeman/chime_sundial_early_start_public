import numpy as np
import matplotlib.pyplot as plt

# Set global plotting style
plt.rc("font", **{"size": "10"})
plt.rcParams.update({
    'figure.figsize': (10, 3),
    'axes.grid': True,
    'grid.linestyle': "--",
})

# Load saved visibility file
data = np.load("chime_vis_v2.npz")

# Parse arrays in file
unixtime_axis = data["unixtime_axis"]
PSTtime_axis = data["PSTtime_axis"]
fractional_csd_axis = data["fractional_csd_axis"]
ra_axis = data["ra_axis"]
bx_values = data["bx_values"]
by_values = data["by_values"]
dates = data["dates"]
freq = data["freq"]
vis = data["vis"]

# Function to plot visibilities for csd and baseline
def plot_vis(day=0, baseline=0, x_min=0, x_max=360, y_min=-2000, y_max=2000, figsize=(10, 3)):
    """Plot real part and absolute value of saved visibilities.

    Parameters
    ----------
    day : int, optional
        CSD index to plot.
    baseline : int, optional
        Baseline index to plot.
    x_min, x_max : float, optional
        RA axis limits. Values outside of [0, 360] will be clipped to this range.
    y_min, y_max : float, optional
        Y axis limits.
    figsize : tuple, optional
        Figure size.
    """
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    ax.plot(ra_axis[day], np.real(vis[day, baseline]), '.-', markersize=4, c='blue')
    ax.plot(ra_axis[day], np.abs(vis[day, baseline]), '.-', markersize=4, c='black')

    ax.set_xlim(max(0, x_min), min(360, x_max))
    ax.set_ylim(y_min, y_max)

    ax.set_xlabel("Right Ascension [deg]")
    ax.set_ylabel("Visibility [Jy]")

    ax.set_title(f"Date = {dates[day]}, Baseline = {bx_values[baseline]:.1f}m EW")

    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())

    tick_locs = ax.get_xticks()
    tick_locs = tick_locs[(tick_locs >= ax.get_xlim()[0]) & (tick_locs <= ax.get_xlim()[1])]

    nearest_time_idx = [np.argmin(np.abs(ra_axis[day] - loc)) for loc in tick_locs]
    tick_labels = [PSTtime_axis[day][i] for i in nearest_time_idx]

    ax2.set_xticks(tick_locs)
    ax2.set_xticklabels(tick_labels)
    # ax2.set_xlabel("Local Time at Telescope")

    plt.tight_layout()