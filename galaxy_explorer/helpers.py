"""
helpers.py — Utility functions for the Galaxy Explorer notebooks.

These functions handle all the "fiddly" technical details so you can focus on
the exciting science. You do not need to understand every line in here —
just know that when you call these functions in the notebooks, they do the
right thing behind the scenes.

If you are curious about any of it, ask Sophie!
"""

import os
import requests
import numpy as np
import h5py
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from astropy.cosmology import LambdaCDM

# Synthesizer imports
from synthesizer.grid import Grid
from synthesizer.particle.stars import Stars
from synthesizer.particle.galaxy import Galaxy
from synthesizer.emission_models import StellarEmissionModel
from synthesizer.instruments import FilterCollection, Instrument
from synthesizer.load_data.utils import age_lookup_table, lookup_age
from unyt import Msun, Mpc, yr, kpc

# ------------------------------------------------------------------ #
# TNG50 cosmological parameters — these match the simulation exactly   #
# ------------------------------------------------------------------ #
TNG_COSMO = LambdaCDM(Om0=0.3089, Ode0=0.6911, H0=67.74, Ob0=0.0486)
TNG_H = 0.6774   # the "little h" Hubble factor used in TNG units
TNG_SNAP_Z = 0.0  # snapshot 99 is redshift zero (today!)


# ================================================================== #
# STEP 1 — Download a single galaxy's particle data from the TNG API  #
# ================================================================== #

def download_tng_galaxy(subhalo_id, api_key, outdir="./galaxy_data"):
    os.makedirs(outdir, exist_ok=True)
    outfile = os.path.join(outdir, f"galaxy_{subhalo_id}.hdf5")

    if os.path.exists(outfile):
        print(f"Galaxy {subhalo_id} already downloaded.")
        return outfile

    url = f"https://www.tng-project.org/api/TNG50-1/snapshots/99/subhalos/{subhalo_id}/cutout.hdf5"

    params = {
        "stars": "Coordinates,GFM_InitialMass,GFM_StellarFormationTime,GFM_Metallicity"
    }

    print(f"Downloading galaxy {subhalo_id}...")
    r = requests.get(url, headers={"API-Key": api_key}, params=params)

    if r.status_code == 401:
        raise RuntimeError("Invalid API key.")
    if r.status_code != 200:
        raise RuntimeError(f"HTTP {r.status_code}: {r.text[:200]}")

    if "hdf5" not in r.headers.get("Content-Type", ""):
        raise RuntimeError("Did not receive HDF5 file.")

    with open(outfile, "wb") as f:
        f.write(r.content)

    print("Download complete.")
    return outfile


# ================================================================== #
# STEP 2 — Read the HDF5 file and convert units                       #
# ================================================================== #

def load_tng_particles(filepath):
    """
    Read star-particle data from the downloaded HDF5 file and
    convert from TNG's internal units to physical units.

    TNG stores coordinates in "comoving kpc/h" and masses in
    "10^10 Msun/h" — we convert everything to plain kpc, Msun, and years.

    Parameters
    ----------
    filepath : str
        Path to the galaxy HDF5 file.

    Returns
    -------
    dict with keys: coords_kpc, masses_msun, ages_yr, metallicity
    """
    with h5py.File(filepath, "r") as f:
        coords_raw   = f["PartType4/Coordinates"][:]                  # comoving kpc/h
        masses_raw   = f["PartType4/GFM_InitialMass"][:]              # 1e10 Msun/h
        form_a       = f["PartType4/GFM_StellarFormationTime"][:]     # scale factor
        metallicity  = f["PartType4/GFM_Metallicity"][:]              # dimensionless

    # --- Remove "wind particles" (unphysical, flagged with negative a) ---
    valid = form_a > 0
    coords_raw  = coords_raw[valid]
    masses_raw  = masses_raw[valid]
    form_a      = form_a[valid]
    metallicity = metallicity[valid]

    # --- Convert coordinates: comoving kpc/h  →  physical kpc ---
    # At z=0 the scale factor a=1, so comoving = physical
    coords_kpc = coords_raw / TNG_H   # kpc

    # --- Centre the galaxy on its own centre of mass ---
    com = np.average(coords_kpc, weights=masses_raw, axis=0)
    coords_kpc -= com

    # --- Convert masses: 1e10 Msun/h  →  Msun ---
    masses_msun = masses_raw * 1e10 / TNG_H

    # --- Convert formation scale factor  →  lookback age in years ---
    # We build a lookup table once, then interpolate for every particle
    a_table, age_table_gyr = age_lookup_table(TNG_COSMO, redshift=TNG_SNAP_Z)
    ages_gyr = lookup_age(form_a, a_table, age_table_gyr.value)
    ages_yr  = ages_gyr * 1e9   # Gyr → yr

    # Clamp to avoid zero or negative ages (numerical edge cases)
    ages_yr = np.clip(ages_yr, 1e6, 1.4e10)

    n = len(masses_msun)
    total_mass = masses_msun.sum()
    print(f"Loaded {n:,} star particles  |  total stellar mass = {total_mass:.2e} Msun")

    return {
        "coords_kpc":  coords_kpc,
        "masses_msun": masses_msun,
        "ages_yr":     ages_yr,
        "metallicity": metallicity,
    }


# ================================================================== #
# STEP 3 — Build a Synthesizer Galaxy object                          #
# ================================================================== #

def build_synthesizer_galaxy(particles, grid_dir="./grids",
                              grid_name="maraston24-Te00_kroupa-0.1,100"):
    """
    Create a Synthesizer Galaxy from the particle data.

    Synthesizer wants its own unit objects (unyt quantities), so we
    attach the right units here before handing the data over.

    Parameters
    ----------
    particles : dict
        Output of load_tng_particles().
    grid_dir : str
        Folder containing the SPS grid file.
    grid_name : str
        Name of the SPS grid (without the .hdf5 extension).

    Returns
    -------
    galaxy : synthesizer Galaxy object
    grid   : synthesizer Grid object
    """
    # Attach units that Synthesizer understands
    coords_mpc  = (particles["coords_kpc"] / 1000) * Mpc   # kpc → Mpc
    masses_u    = particles["masses_msun"] * Msun
    ages_u      = particles["ages_yr"] * yr

    stars = Stars(
        initial_masses = masses_u,
        ages           = ages_u,
        metallicities  = particles["metallicity"],
        coordinates    = coords_mpc,
    )

    grid = Grid(grid_name, grid_dir=grid_dir)
    galaxy = Galaxy(stars=stars)

    print("Synthesizer Galaxy created successfully!")
    print(f"  Stars: {stars.nparticles:,} particles")
    print(f"  Grid:  {grid_name}")

    return galaxy, grid


# ================================================================== #
# STEP 4 — Run Synthesizer: compute the spectrum (SED)                #
# ================================================================== #

def get_spectrum(galaxy, grid):
    """
    Use Synthesizer to assign a stellar spectrum to every star particle,
    then sum them all up to get the total galaxy spectrum (the SED).

    This is the core of what Synthesizer does: it looks up the spectrum
    for each star's age and metallicity in the SPS grid, weights it by
    the star's mass, and sums everything together.

    Parameters
    ----------
    galaxy : synthesizer Galaxy
    grid   : synthesizer Grid

    Returns
    -------
    sed : synthesizer Sed object — the galaxy's spectrum
    """
    model = StellarEmissionModel(grid)
    galaxy.get_spectra(model)

    # Print what we got so the student can see
    available_spectra = list(galaxy.stars.spectra.keys())
    print(f"Synthesizer generated these spectra: {available_spectra}")

    # Use the first (and usually only) one
    sed = galaxy.stars.spectra[available_spectra[0]]
    print(f"Wavelength range: {sed.lam.min():.0f} – {sed.lam.max():.0f} Å")
    return sed, available_spectra[0]


# ================================================================== #
# STEP 5 — Make a pixel image                                          #
# ================================================================== #

def make_image(galaxy, spectrum_label, fov_kpc=60.0, pixel_kpc=0.5,
               filter_codes=None):
    """
    Project the galaxy's star particles onto a 2D pixel grid to make an image.

    Each pixel's brightness is the total luminosity of all star particles
    that fall within that pixel.

    Parameters
    ----------
    galaxy         : synthesizer Galaxy
    spectrum_label : str — which spectrum to image (from get_spectrum)
    fov_kpc        : float — width of the image in kpc (try 40 – 100)
    pixel_kpc      : float — size of each pixel in kpc (smaller = sharper)
    filter_codes   : list of str — which telescope filters to use.
                     Examples: "LSST/LSST.r", "JWST/NIRCam.F200W"

    Returns
    -------
    images : synthesizer ImageCollection
    """
    if filter_codes is None:
        filter_codes = ["LSST/LSST.r", "LSST/LSST.g", "LSST/LSST.u"]

    fc         = FilterCollection(filter_codes=filter_codes)
    instrument = Instrument("telescope", filters=fc, resolution=pixel_kpc * kpc)

    images = galaxy.get_images_luminosity(
        spectrum_label,
        fov        = fov_kpc * kpc,
        instrument = instrument,
        img_type   = "hist",   # histogram — fast and needs no smoothing lengths
    )
    return images


# ================================================================== #
# PLOTTING HELPERS                                                     #
# ================================================================== #

def plot_spectrum(sed, title="Galaxy Spectrum"):
    """Plot the galaxy's spectral energy distribution."""
    lam_ang = sed.lam.to("Angstrom").value
    lum     = sed.luminosity

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.loglog(lam_ang, lum, color="steelblue", linewidth=1.5)

    # Shade wavelength regions
    ax.axvspan(912,  4000,  alpha=0.12, color="violet", label="Ultraviolet (UV)")
    ax.axvspan(4000, 7000,  alpha=0.12, color="gold",   label="Optical (visible)")
    ax.axvspan(7000, 30000, alpha=0.12, color="tomato", label="Near-infrared")

    ax.set_xlabel("Wavelength (Å)", fontsize=12)
    ax.set_ylabel(r"Luminosity  (erg s$^{-1}$ Hz$^{-1}$)", fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3, which="both")
    plt.tight_layout()
    plt.show()


def plot_one_image(image_data, title="", cmap="magma"):
    """Display a single galaxy image with a logarithmic colour scale."""
    arr = np.array(image_data)
    arr = np.where(arr > 0, np.log10(arr + 1e-40), np.nan)

    fig, ax = plt.subplots(figsize=(6, 6), facecolor="black")
    vmin = np.nanpercentile(arr, 5)
    vmax = np.nanpercentile(arr, 99.5)
    ax.imshow(arr, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title, color="white", fontsize=12, pad=8)
    ax.axis("off")
    ax.set_facecolor("black")
    fig.patch.set_facecolor("black")
    plt.tight_layout()
    plt.show()


def plot_rgb(images, r_filter, g_filter, b_filter, title="Colour image", stretch=0.01):
    """
    Combine three filter images into a single colour (RGB) picture.

    Uses the astronomers' standard asinh stretch so both bright cores
    and faint outer regions are visible at the same time.
    """
    def get_arr(images, filt):
        arr = np.array(images[filt].arr)
        arr = np.clip(arr, 0, None)
        return arr

    r = get_arr(images, r_filter)
    g = get_arr(images, g_filter)
    b = get_arr(images, b_filter)

    # Asinh stretch (Lupton et al. 2004 — the astronomer's colour trick)
    I = (r + g + b) / 3.0 + 1e-40
    fac = np.arcsinh(stretch * I) / (stretch * I)
    r_s, g_s, b_s = r * fac, g * fac, b * fac

    # Normalise to [0, 1]
    top = np.nanpercentile(np.stack([r_s, g_s, b_s]), 99.5)
    rgb = np.clip(np.stack([r_s, g_s, b_s], axis=-1) / top, 0, 1)

    fig, ax = plt.subplots(figsize=(7, 7), facecolor="black")
    ax.imshow(rgb, origin="lower")
    ax.set_title(title, color="white", fontsize=13, pad=8)
    ax.axis("off")
    fig.patch.set_facecolor("black")
    plt.tight_layout()
    plt.show()


def plot_particles(particles, title="Galaxy star particles", color_by="age"):
    """
    Quick scatter-plot of the star particles so you can see the galaxy
    before running the full Synthesizer pipeline.

    color_by : "age" or "metallicity" or "mass"
    """
    x = particles["coords_kpc"][:, 0]
    y = particles["coords_kpc"][:, 1]

    if color_by == "age":
        c = particles["ages_yr"] / 1e9   # Gyr
        cmap, label = "plasma_r", "Stellar age (Gyr)"
    elif color_by == "metallicity":
        c = np.log10(particles["metallicity"] + 1e-10)
        cmap, label = "viridis", "log Metallicity"
    else:
        c = np.log10(particles["masses_msun"])
        cmap, label = "inferno", r"log Mass (M$_\odot$)"

    # Only plot a random sample of 20 000 particles for speed
    rng  = np.random.default_rng(42)
    idx  = rng.choice(len(x), size=min(20_000, len(x)), replace=False)

    fig, ax = plt.subplots(figsize=(7, 7), facecolor="black")
    sc = ax.scatter(x[idx], y[idx], c=c[idx], s=0.3,
                    cmap=cmap, alpha=0.7, linewidths=0)
    cbar = fig.colorbar(sc, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label(label, color="white")
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")
    ax.set_facecolor("black")
    fig.patch.set_facecolor("black")
    ax.set_xlabel("x (kpc)", color="white")
    ax.set_ylabel("y (kpc)", color="white")
    ax.tick_params(colors="white")
    ax.set_title(title, color="white", fontsize=12)
    ax.set_aspect("equal")
    plt.tight_layout()
    plt.show()
