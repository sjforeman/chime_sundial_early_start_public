"""Interactive two-element interferometer activities"""

import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import interact, FloatSlider


def fringe_phase(B, lam, theta_deg):
    """Geometric phase difference between the antennas [rad]."""
    theta = np.radians(theta_deg)
    return 2 * np.pi * B * np.sin(theta) / lam


theta_grid = np.linspace(-90, 90, 4000)


def interferometer(B=100.0, lam=0.21, theta=20.0):
    """Plot the interferometer geometry and its real visibility fringe."""
    th = np.radians(theta)
    fig, (ax_geo, ax_vis) = plt.subplots(2, 1, figsize=(8, 7), gridspec_kw={"height_ratios": [3, 1]})

    # ---------- top: geometry ----------
    x1 = -B / 2
    x2 = B / 2

    # Unit vector toward the source
    s_hat = np.array([np.sin(th), np.cos(th)])

    # Antennas
    ax_geo.plot([x1, x2], [0, 0], "ko", ms=12)
    ax_geo.text(x1, -0.10 * B, "ant 1", ha="center")
    ax_geo.text(x2, -0.10 * B, "ant 2", ha="center")

    # Baseline B
    y_B = -0.05 * B
    ax_geo.annotate(
        "",
        xy=(x2, y_B),
        xytext=(x1, y_B),
        arrowprops=dict(arrowstyle="<->", color="purple", lw=2),
    )
    ax_geo.text(
        0,
        y_B - 0.08 * B,
        rf"$B = {B:.0f}\ \mathrm{{m}}$",
        color="purple",
        fontsize=12,
        ha="center",
    )

    # Zenith direction
    ax_geo.plot([0, 0], [0, 0.7 * B], "k--", lw=1)
    ax_geo.text(0, 0.73 * B, "zenith", ha="center")

    # Theta: angle between zenith and source direction
    angle_values = np.linspace(0, th, 50)
    R = 0.25 * B
    arc_x = R * np.sin(angle_values)
    arc_y = R * np.cos(angle_values)
    ax_geo.plot(arc_x, arc_y, color="green", lw=2)

    middle_angle = th / 2
    ax_geo.text(
        0.30 * B * np.sin(middle_angle),
        0.30 * B * np.cos(middle_angle),
        r"$\theta$",
        color="green",
        fontsize=14,
    )

    # Incoming wave direction
    tip = 0.75 * B * s_hat
    ax_geo.annotate(
        "",
        xy=(0.15 * tip[0], 0.15 * tip[1]),
        xytext=(tip[0], tip[1]),
        arrowprops=dict(arrowstyle="-|>", lw=2, color="C0"),
    )
    ax_geo.text(
        tip[0],
        tip[1] + 0.05 * B,
        "incoming wave",
        color="C0",
        ha="center",
    )

    # Wavefront through the nearer antenna
    near, far = (x2, x1) if theta >= 0 else (x1, x2)
    p_hat = np.array([-s_hat[1], s_hat[0]])
    L = 1.2 * B
    ax_geo.plot(
        [near - L * p_hat[0], near + L * p_hat[0]],
        [-L * p_hat[1], L * p_hat[1]],
        "C0--",
        lw=1.5,
    )

    # Extra path length
    extra = B * abs(np.sin(th))
    p1 = np.array([far, 0]) + extra * s_hat
    ax_geo.plot([far, p1[0]], [0, p1[1]], "r-", lw=3)
    ax_geo.text(
        0.5 * (far + p1[0]),
        0.5 * p1[1],
        r"$B|\sin\theta|$",
        color="red",
        fontsize=12,
    )

    ax_geo.set_xlim(-1.1 * B, 1.1 * B)
    ax_geo.set_ylim(-0.2 * B, 1.0 * B)
    ax_geo.set_aspect("equal")
    ax_geo.set_xticks([])
    ax_geo.set_yticks([])
    ax_geo.set_title(
        rf"Path difference = $B\sin\theta$ = {B * np.sin(th):.2f} m"
    )

    # ---------- bottom: visibility fringe ----------
    visibility = np.cos(fringe_phase(B, lam, theta_grid))
    ax_vis.plot(theta_grid, visibility)

    current_visibility = np.cos(fringe_phase(B, lam, theta))
    ax_vis.axvline(theta, color="r")
    ax_vis.plot(theta, current_visibility, "ro")

    phi = fringe_phase(B, lam, theta)
    ax_vis.set_title(f"Current phase = {phi:.1f} rad")
    ax_vis.set_xlabel(r"Source angle from zenith, $\theta$ [deg]")
    ax_vis.set_ylabel(r"Re[V] = $\cos(2\pi B\sin\theta/\lambda)$")
    ax_vis.set_ylim(-1.15, 1.15)

    plt.tight_layout()
    plt.show()


def show_interferometer():
    """Show sliders for the geometry and fringe activity."""
    return interact(
        interferometer,
        B=FloatSlider(min=1, max=20, step=1, value=5, description="B [m]"),
        lam=FloatSlider(
            min=0.05, max=1.0, step=0.01, value=0.21, description="λ [m]"
        ),
        theta=FloatSlider(
            min=-90, max=90, step=0.5, value=20, description="θ [deg]"
        ),
    )


def correlate(phi_turns=0.1):
    """Show how multiplying and averaging two signals gives correlation."""
    phi = 2 * np.pi * phi_turns
    t = np.linspace(0, 4, 2000)  # time in wave periods

    s1 = np.sin(2 * np.pi * t)
    s2 = np.sin(2 * np.pi * t - phi)
    product = s1 * s2

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 5), sharex=True)

    # Antenna signals
    ax1.plot(t, s1, label="antenna 1")
    ax1.plot(t, s2, label=f"antenna 2 (delayed by {phi_turns:.2f} turns)")
    ax1.legend(loc="upper right")
    ax1.set_ylabel("voltage")

    # Product and its time average
    ax2.plot(t, product, color="grey", lw=1, label=r"instantaneous product $s_1s_2$")
    ax2.axhline(
        product.mean(),
        color="r",
        lw=2,
        label=f"time-averaged correlation = {product.mean():.2f}",
    )
    ax2.axhline(0, color="k", lw=0.5)
    ax2.legend(loc="upper right")
    ax2.set_xlabel("time [wave periods]")
    ax2.set_ylabel(r"$s_1 \times s_2$")
    ax2.set_ylim(-1.1, 1.1)

    plt.tight_layout()
    plt.show()


def show_correlation():
    """Show the phase-shift/correlation slider activity."""
    return interact(
        correlate,
        phi_turns=FloatSlider(
            min=0,
            max=1,
            step=0.01,
            value=0.1,
            description="φ [turns]",
        ),
    )
