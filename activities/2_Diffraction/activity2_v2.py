

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import j1

# ----------------------------------------------------------------------
# Constants and presets
# ----------------------------------------------------------------------
RAD_TO_ARCSEC = np.degrees(1.0) * 3600.0          
AIRY_FIRST_ZERO = 3.8317059702                     # first zero of J1(u)/u

#: wavelength presets for the dropdown  (label -> metres)
WAVELENGTH_PRESETS = {
    "Visible light (550 nm)":        550e-9,
    "Near-IR (2.2 \u03bcm)":         2.2e-6,
    "Millimetre / ALMA (1.3 mm)":    1.3e-3,
    "Radio: HI 21 cm line":          0.21,
    # "Radio: CHIME band (~50 cm)":    0.50,
    # "Radio: metre-wave (1 m)":       1.0,
}



# ----------------------------------------------------------------------
def diffraction_limit(wavelength, D, arcsec=True, rayleigh=True):
    """
    Diffraction-limited angular resolution of a filled circular aperture.

    Parameters
    ----------
    wavelength : float
        Observing wavelength in metres.
    D : float
        Aperture diameter in metres.
    arcsec : bool
        Return arcseconds (default) instead of radians.
    rayleigh : bool
        Include the 1.22 Airy factor (default).  Set False for plain lambda/D.

    Returns
    -------
    theta : float
    """
    factor = 1.22 if rayleigh else 1.0
    theta = factor * wavelength / D
    return theta * RAD_TO_ARCSEC if arcsec else theta


def airy_psf(theta, theta_R):
    """
    Normalised Airy intensity pattern.

    Parameters
    ----------
    theta : array
        Angular offset from the source centre (any angular unit).
    theta_R : float
        Rayleigh resolution 1.22*lambda/D in the *same* unit.
        (The first dark ring of the pattern sits exactly at theta_R.)
    """
    # u = pi D theta / lambda ; first zero at u = 3.8317 corresponds to theta_R
    u = AIRY_FIRST_ZERO * np.asarray(theta, dtype=float) / theta_R
    out = np.ones_like(u)
    nz = u != 0
    out[nz] = (2.0 * j1(u[nz]) / u[nz]) ** 2
    return out


# ----------------------------------------------------------------------
# Visualisation: two point sources through a finite aperture
# ----------------------------------------------------------------------
def show_two_sources(D=2.4, wavelength=550e-9, sep_over_theta=1.0,
                     ax=None, npix=400, field=3.5):
    """
    Render two equal point sources separated by `sep_over_theta` Rayleigh
    units, as seen through an aperture of diameter D at `wavelength`.

    Shows the simulated "sky image": the sum of the two Airy patterns.

    Parameters
    ----------
    D, wavelength : floats [m]
    sep_over_theta : float
        Source separation in units of theta_R = 1.22 lambda/D.
        ~1.0 = just resolved (Rayleigh), <1 unresolved, >1 clearly split.
    field : float
        Half-width of the plotted field in units of theta_R.
    """
    theta_R = diffraction_limit(wavelength, D)          # arcsec
    sep = sep_over_theta * theta_R                      # arcsec

    # --- 2D image -----------------------------------------------------
    x = np.linspace(-field * theta_R, field * theta_R, npix)
    X, Y = np.meshgrid(x, x)
    r1 = np.hypot(X - sep / 2, Y)
    r2 = np.hypot(X + sep / 2, Y)
    img = airy_psf(r1, theta_R) + airy_psf(r2, theta_R)

    # --- figure -------------------------------------------------------
    created = ax is None
    if created:
        fig, ax = plt.subplots(figsize=(5.5, 5.5))

    ext = [x[0], x[-1], x[0], x[-1]]
    ax.imshow(img ** 0.5, extent=ext, origin="lower", cmap="inferno")
    ax.set_xlabel("angle  [arcsec]")
    ax.set_ylabel("angle  [arcsec]")

    if sep_over_theta < 0.9:
        verdict = "UNRESOLVED \u2014 looks like one source!"
    elif sep_over_theta < 1.1:
        verdict = "Rayleigh criterion: just barely resolved"
    else:
        verdict = "clearly resolved"
    ax.set_title(verdict)

    txt = (f"D = {_fmt_length(D)},   \u03bb = {_fmt_length(wavelength)}\n"
           f"\u03b8 = 1.22\u03bb/D = {_fmt_angle(theta_R)}     "
           f"separation = {sep_over_theta:.2f} \u03b8")
    ax.text(0.02, 0.98, txt, transform=ax.transAxes, va="top",
            color="w", fontsize=8.5)

    if created:
        plt.tight_layout()
        plt.show()
    return theta_R


# ----------------------------------------------------------------------
# Interactive widget (Jupyter / Colab)
# ----------------------------------------------------------------------
# Turn off automatic figure display so we control redraws manually
plt.ioff()

def interactive_resolution():
    import ipywidgets as w
    from IPython.display import display

    D_slider = w.FloatLogSlider(
        value=2.4, base=10, min=-2.3, max=6, step=0.02,
        description="D [m]", readout=False,
        continuous_update=True   # <-- key change
    )
    D_box = w.FloatText(value=2.4, layout=w.Layout(width="100px"))
    w.jslink((D_slider, "value"), (D_box, "value"))

    wl_dd = w.Dropdown(options=WAVELENGTH_PRESETS, value=550e-9, description="λ")

    sep_slider = w.FloatSlider(
        value=1.5, min=0.2, max=3.0, step=0.05,
        description="Sep. / θ", readout=False,
        continuous_update=True   # <-- key change
    )
    sep_box = w.FloatText(value=1.5, layout=w.Layout(width="100px"))
    w.jslink((sep_slider, "value"), (sep_box, "value"))

    out_lbl = w.HTML()

    # --- build the figure ONCE, reuse it on every update ---
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    npix, field = 300, 3.5   # slightly reduced npix helps responsiveness too
    x0 = np.linspace(-1, 1, npix)  # placeholder grid, rescaled each call
    im = ax.imshow(np.zeros((npix, npix)), origin="lower", cmap="inferno")
    ax.set_xlabel("angle  [arcsec]")
    ax.set_ylabel("angle  [arcsec]")
    txt_artist = ax.text(0.02, 0.98, "", transform=ax.transAxes, va="top",
                          color="w", fontsize=8.5)

    out = w.Output()  # persistent output area

    def _update(D, wavelength, sep_over_theta):
        theta_R = diffraction_limit(wavelength, D)
        sep = sep_over_theta * theta_R

        x = np.linspace(-field * theta_R, field * theta_R, npix)
        X, Y = np.meshgrid(x, x)
        r1 = np.hypot(X - sep / 2, Y)
        r2 = np.hypot(X + sep / 2, Y)
        img = (airy_psf(r1, theta_R) + airy_psf(r2, theta_R)) ** 0.5

        im.set_data(img)
        im.set_extent([x[0], x[-1], x[0], x[-1]])
        im.set_clim(img.min(), img.max())

        if sep_over_theta < 0.9:
            verdict = "UNRESOLVED — looks like one source!"
        elif sep_over_theta < 1.1:
            verdict = "Rayleigh criterion: just barely resolved"
        else:
            verdict = "clearly resolved"
        ax.set_title(verdict)

        txt_artist.set_text(
            f"D = {_fmt_length(D)},   λ = {_fmt_length(wavelength)}\n"
            f"θ = 1.22λ/D = {_fmt_angle(theta_R)}     "
            f"separation = {sep_over_theta:.2f} θ"
        )

        out_lbl.value = (
            f'<div style="display:inline-block;font-size:22px;font-weight:bold;'
            f'border:2px solid #555;border-radius:6px;padding:8px 14px;margin-top:8px;">'
            f'θ = 1.22 λ/D = {_fmt_angle(theta_R)}</div>'
        )

        with out:
            out.clear_output(wait=True)
            display(fig)

    ui = w.VBox([wl_dd, w.HBox([D_slider, D_box]), w.HBox([sep_slider, sep_box]), out_lbl, out])

    inter = w.interactive_output(
        _update,
        dict(D=D_slider, wavelength=wl_dd, sep_over_theta=sep_slider)
    )

    display(ui)



def dish_to_match(target_wavelength=0.21,
                  ref_D=2.4, ref_wavelength=550e-9, ref_name="Hubble"):
    """
    Diameter of a single dish at `target_wavelength` that matches the
    diffraction limit of a reference telescope.

    Returns (needed_diameter_m, reference_theta_arcsec).
    """
    theta_ref = diffraction_limit(ref_wavelength, ref_D, arcsec=False)
    D_needed = 1.22 * target_wavelength / theta_ref
    print(f"{ref_name} ({_fmt_length(ref_D)} at {_fmt_length(ref_wavelength)}): "
          f"\u03b8 = {_fmt_angle(theta_ref * RAD_TO_ARCSEC)}")
    print(f"Single dish needed at \u03bb = {_fmt_length(target_wavelength)} "
          f"to match it:  D = {_fmt_length(D_needed)}"
          f"   ({D_needed / 1e3:,.0f} km!)")
    return D_needed, theta_ref * RAD_TO_ARCSEC




# ----------------------------------------------------------------------
# Small formatting helpers
# ----------------------------------------------------------------------
def _fmt_length(m):
    if m >= 1e3:
        return f"{m/1e3:,.0f} km"
    if m >= 1:
        return f"{m:.3g} m"
    if m >= 1e-2:
        return f"{m*100:.3g} cm"
    if m >= 1e-3:
        return f"{m*1e3:.3g} mm"
    if m >= 1e-6:
        return f"{m*1e6:.3g} \u03bcm"
    return f"{m*1e9:.3g} nm"


def _fmt_angle(arcsec):
    if arcsec >= 3600:
        return f"{arcsec/3600:.3g} deg"
    if arcsec >= 60:
        return f"{arcsec/60:.3g} arcmin"
    if arcsec >= 1e-2:
        return f"{arcsec:.3g} arcsec"
    return f"{arcsec*1e3:.3g} mas"
