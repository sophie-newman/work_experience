"""
helpers.py — Utility functions for the Galaxy Explorer notebooks.

These functions handle all the "fiddly" technical details so you can focus on
the exciting science. You do not need to understand every line in here —
just know that when you call these functions in the notebooks, they do the
right thing behind the scenes.

If you are curious about any of it, ask Sophie!
"""

import os
import glob
import tarfile
import requests
import numpy as np
import h5py
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from mpl_toolkits.axes_grid1 import make_axes_locatable
from astropy.cosmology import LambdaCDM
from astropy.io import fits

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

    # HDF5 files start with the magic signature b'\x89HDF'
    if not r.content[:4] == b'\x89HDF':
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
        centre         = np.array([0.0, 0.0, 0.0]) * Mpc,
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
    model = StellarEmissionModel("incident", grid=grid, extract="incident",
                                  per_particle=True)
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

    # Photometry must be computed before images can be made
    galaxy.get_photo_lnu(fc)

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


def plot_one_image(image_data, title="", cmap="magma", fov_kpc=None,
                   xlim=None, ylim=None):
    """Display a single galaxy image with a logarithmic colour scale.

    Parameters
    ----------
    fov_kpc : float, optional
        Field of view in kpc (must match the fov_kpc used in make_image).
        When provided, axes are labelled in kpc so xlim/ylim can be given
        in kpc — e.g. xlim=(-20, 20) to zoom in on the central 40 kpc.
    xlim, ylim : tuple of (min, max), optional
        Axis limits. In kpc when fov_kpc is set, otherwise in pixels.
    """
    arr = np.array(image_data)
    arr_log = np.log10(arr + 1e-6)

    fig, ax = plt.subplots(figsize=(6, 6), facecolor="black")

    if fov_kpc is not None:
        half = fov_kpc / 2
        im = ax.imshow(arr_log, origin="lower", cmap=cmap,
                       extent=[-half, half, -half, half])
        ax.set_xlabel("x (kpc)", color="white", fontsize=11)
        ax.set_ylabel("y (kpc)", color="white", fontsize=11)
        ax.tick_params(colors="white")
        for spine in ax.spines.values():
            spine.set_edgecolor("white")
    else:
        im = ax.imshow(arr_log, origin="lower", cmap=cmap)
        ax.axis("off")

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label("log Luminosity", color="white", fontsize=10)
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")

    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)

    ax.set_title(title, color="white", fontsize=12, pad=8)
    ax.set_facecolor("black")
    fig.patch.set_facecolor("black")
    plt.tight_layout()
    plt.show()


def plot_rgb(images, r_filter, g_filter, b_filter, title="Colour image",
             stretch=0.01, fov_kpc=None, xlim=None, ylim=None):
    """
    Combine three filter images into a single colour (RGB) picture.

    Uses the astronomers' standard asinh stretch so both bright cores
    and faint outer regions are visible at the same time.

    Parameters
    ----------
    fov_kpc : float, optional
        Field of view in kpc (must match the fov_kpc used in make_image).
        When provided, axes are labelled in kpc so xlim/ylim can be given
        in kpc — e.g. xlim=(-20, 20) to zoom in on the central 40 kpc.
    xlim, ylim : tuple of (min, max), optional
        Axis limits. In kpc when fov_kpc is set, otherwise in pixels.
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

    if fov_kpc is not None:
        half = fov_kpc / 2
        ax.imshow(rgb, origin="lower", extent=[-half, half, -half, half])
        ax.set_xlabel("x (kpc)", color="white", fontsize=11)
        ax.set_ylabel("y (kpc)", color="white", fontsize=11)
        ax.tick_params(colors="white")
        for spine in ax.spines.values():
            spine.set_edgecolor("white")
    else:
        ax.imshow(rgb, origin="lower")
        ax.axis("off")

    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)

    ax.set_title(title, color="white", fontsize=13, pad=8)
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


# ================================================================== #
# SKIRT ATLAS FUNCTIONS                                                #
# ================================================================== #
#
# SKIRT is a radiative transfer code that post-processes TNG50 galaxies
# to produce realistic photometric images. Unlike Synthesizer (which sums
# particle SEDs analytically), SKIRT traces individual photon packets
# through the galaxy's gas and dust, computing absorption and scattering
# self-consistently. The result is much closer to what a real telescope
# would see.
#
# The SKIRT atlas provides:
#   - 1600×1600 pixel images in 18 photometric filters (UV → mid-IR)
#   - "nodust" versions of each image (intrinsic light, no attenuation)
#   - Physical property maps: stellar mass, age, metallicity, dust mass
#   - Five viewing orientations per galaxy
#
# File naming: TNG{ID:06d}_{orient}_{filter_name}[_nodust].fits
# e.g. TNG553837_O1_LSST_r.fits
#
# Image grid:  100 pc / pixel  →  160 kpc across
# Flux units:  MJy / sr  (surface brightness)

SKIRT_BASE_DIR = "/cosma7/data/dp276/dc-newm1/skirt_atlas_data"

# Ordered list of all photometric filters from UV → mid-IR
SKIRT_FILTERS = [
    ("GALEX_FUV",  "GALEX FUV\n(~1500 Å)",  "violet"),
    ("GALEX_NUV",  "GALEX NUV\n(~2300 Å)",  "blueviolet"),
    ("Johnson_U",  "Johnson U\n(~3650 Å)",   "royalblue"),
    ("Johnson_B",  "Johnson B\n(~4400 Å)",   "cornflowerblue"),
    ("Johnson_V",  "Johnson V\n(~5500 Å)",   "forestgreen"),
    ("LSST_r",     "LSST r\n(~6200 Å)",      "orange"),
    ("Johnson_I",  "Johnson I\n(~8000 Å)",   "tomato"),
    ("LSST_z",     "LSST z\n(~9000 Å)",      "firebrick"),
    ("2MASS_J",    "2MASS J\n(1.2 μm)",       "darkred"),
    ("2MASS_H",    "2MASS H\n(1.6 μm)",       "saddlebrown"),
    ("2MASS_Ks",   "2MASS Ks\n(2.2 μm)",      "peru"),
    ("WISE_W1",    "WISE W1\n(3.4 μm)",       "goldenrod"),
    ("WISE_W2",    "WISE W2\n(4.6 μm)",       "gold"),
]


def download_skirt_galaxy(subhalo_id, api_key, base_dir=SKIRT_BASE_DIR):
    """
    Download the SKIRT atlas tar.gz for a TNG50-1 galaxy.
    Skips if data is already present.

    Parameters
    ----------
    subhalo_id : str or int
    api_key    : str — your IllustrisTNG API key
    base_dir   : str — parent folder; a subdirectory named after the ID is created

    Returns
    -------
    tarfile_path : str
    outdir       : str
    """
    subhalo_id = str(subhalo_id)
    outdir = os.path.join(base_dir, subhalo_id)
    os.makedirs(outdir, exist_ok=True)
    tar_path = os.path.join(outdir, f"{subhalo_id}.tar.gz")

    existing = glob.glob(os.path.join(outdir, "*.fits"))
    if existing:
        print(f"Galaxy {subhalo_id}: already extracted ({len(existing)} FITS files).")
        return tar_path, outdir

    if os.path.exists(tar_path):
        print(f"Galaxy {subhalo_id}: tar.gz already downloaded.")
        return tar_path, outdir

    url = (f"https://www.tng-project.org/api/TNG50-1/snapshots/99"
           f"/subhalos/{subhalo_id}/skirt/skirt_atlas.tar.gz")
    print(f"Downloading galaxy {subhalo_id} (this may take a minute)...")
    r = requests.get(url, headers={"API-Key": api_key}, stream=True)
    if r.status_code != 200:
        raise RuntimeError(f"Download failed: HTTP {r.status_code}")
    with open(tar_path, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)
    print("Download complete.")
    return tar_path, outdir


def extract_skirt_tar(tar_path, outdir):
    """
    Extract a SKIRT atlas tar.gz. Does nothing if FITS files already exist.
    """
    existing = glob.glob(os.path.join(outdir, "*.fits"))
    if existing:
        print(f"Already extracted ({len(existing)} FITS files).")
        return
    print("Extracting files...")
    with tarfile.open(tar_path) as tar:
        tar.extractall(outdir)
    n = len(glob.glob(os.path.join(outdir, "*.fits")))
    print(f"Done — {n} FITS files.")


def skirt_file(subhalo_id, orientation, filter_name,
               nodust=False, base_dir=SKIRT_BASE_DIR):
    """
    Return the path to a specific SKIRT FITS file.

    Parameters
    ----------
    subhalo_id  : str or int
    orientation : str — 'O1' to 'O5'  (five viewing angles)
    filter_name : str — e.g. 'LSST_r', 'GALEX_FUV', 'stellarmass'
    nodust      : bool — use the dust-free version (photometric bands only)

    Examples
    --------
    skirt_file(553837, "O1", "LSST_r")              # with dust
    skirt_file(553837, "O1", "LSST_r", nodust=True) # intrinsic (no dust)
    skirt_file(553837, "O1", "stellarmass")          # stellar mass map
    """
    gid = str(subhalo_id).zfill(6)
    suffix = "_nodust" if nodust else ""
    fname = f"TNG{gid}_{orientation}_{filter_name}{suffix}.fits"
    return os.path.join(base_dir, str(subhalo_id), fname)


def load_fits_image(filepath):
    """Load a SKIRT FITS image and return a float32 numpy array."""
    with fits.open(filepath) as hdul:
        return hdul[0].data.astype(np.float32)


def plot_skirt_image(image, title="", cmap="magma", fov_kpc=160,
                     label="log Surface Brightness (MJy/sr)",
                     xlim=None, ylim=None):
    """
    Display a single SKIRT FITS image with a logarithmic colour scale.

    Parameters
    ----------
    image   : 2D numpy array
    fov_kpc : float — field of view in kpc (SKIRT default is 160 kpc)
    xlim, ylim : tuple — zoom region, e.g. xlim=(-40, 40)
    """
    arr = np.log10(np.clip(image, 1e-6, None))
    half = fov_kpc / 2

    fig, ax = plt.subplots(figsize=(6, 6), facecolor="black")
    im = ax.imshow(arr, origin="lower", cmap=cmap,
                   extent=[-half, half, -half, half])

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label(label, color="white", fontsize=9)
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")

    ax.set_xlabel("x (kpc)", color="white", fontsize=10)
    ax.set_ylabel("y (kpc)", color="white", fontsize=10)
    ax.tick_params(colors="white")
    ax.set_title(title, color="white", fontsize=12, pad=8)
    ax.set_facecolor("black")
    fig.patch.set_facecolor("black")
    for spine in ax.spines.values():
        spine.set_edgecolor("white")
    if xlim:
        ax.set_xlim(xlim)
    if ylim:
        ax.set_ylim(ylim)
    plt.tight_layout()
    plt.show()


def plot_skirt_multiwave(subhalo_id, orientation, filter_list,
                         base_dir=SKIRT_BASE_DIR, fov_kpc=160,
                         ncols=4, cmap="magma", xlim=None, ylim=None):
    """
    Show the same galaxy through multiple filters in a grid.

    Parameters
    ----------
    filter_list : list of (filter_name, display_label, title_colour) tuples
                  Use the SKIRT_FILTERS constant for the full set.
    """
    nrows = int(np.ceil(len(filter_list) / ncols))
    fig, axes = plt.subplots(nrows, ncols,
                              figsize=(4 * ncols, 4.2 * nrows),
                              facecolor="black")
    axes = np.array(axes).flatten()
    half = fov_kpc / 2

    for ax, (fname, label, colour) in zip(axes, filter_list):
        img = load_fits_image(skirt_file(subhalo_id, orientation, fname,
                                         base_dir=base_dir))
        arr = np.log10(np.clip(img, 1e-6, None))
        ax.imshow(arr, origin="lower", cmap=cmap,
                  extent=[-half, half, -half, half])
        ax.set_title(label, color=colour, fontsize=9, pad=4)
        ax.set_facecolor("black")
        ax.tick_params(colors="white", labelsize=7)
        ax.set_xlabel("x (kpc)", color="white", fontsize=7)
        ax.set_ylabel("y (kpc)", color="white", fontsize=7)
        for spine in ax.spines.values():
            spine.set_edgecolor("white")
        if xlim:
            ax.set_xlim(xlim)
        if ylim:
            ax.set_ylim(ylim)

    for ax in axes[len(filter_list):]:
        ax.set_visible(False)

    fig.patch.set_facecolor("black")
    plt.tight_layout()
    plt.show()


def plot_skirt_rgb(r_img, g_img, b_img, title="",
                   stretch=0.05, fov_kpc=160, xlim=None, ylim=None):
    """
    Combine three SKIRT FITS images into an RGB colour picture.
    Uses the astronomers' asinh stretch so faint outskirts stay visible.

    Parameters
    ----------
    r_img, g_img, b_img : 2D numpy arrays (e.g. LSST r, g, u bands)
    stretch             : float — controls brightness/contrast trade-off
    """
    r = np.clip(r_img, 0, None)
    g = np.clip(g_img, 0, None)
    b = np.clip(b_img, 0, None)
    I = (r + g + b) / 3.0 + 1e-40
    fac = np.arcsinh(stretch * I) / (stretch * I)
    r_s, g_s, b_s = r * fac, g * fac, b * fac
    top = np.nanpercentile(np.stack([r_s, g_s, b_s]), 99.5)
    rgb = np.clip(np.stack([r_s, g_s, b_s], axis=-1) / top, 0, 1)

    half = fov_kpc / 2
    fig, ax = plt.subplots(figsize=(7, 7), facecolor="black")
    ax.imshow(rgb, origin="lower", extent=[-half, half, -half, half])
    ax.set_xlabel("x (kpc)", color="white", fontsize=11)
    ax.set_ylabel("y (kpc)", color="white", fontsize=11)
    ax.tick_params(colors="white")
    ax.set_title(title, color="white", fontsize=13, pad=8)
    ax.set_facecolor("black")
    fig.patch.set_facecolor("black")
    for spine in ax.spines.values():
        spine.set_edgecolor("white")
    if xlim:
        ax.set_xlim(xlim)
    if ylim:
        ax.set_ylim(ylim)
    plt.tight_layout()
    plt.show()


def plot_skirt_property_maps(subhalo_id, orientation,
                              base_dir=SKIRT_BASE_DIR, fov_kpc=160,
                              xlim=None, ylim=None):
    """
    Display the four SKIRT physical property maps side by side:
    stellar mass surface density, mean stellar age, metallicity, dust mass.
    """
    maps = [
        ("stellarmass",        r"Stellar mass (M$_\odot$ pc$^{-2}$)", "inferno",  True),
        ("stellarage",         "Stellar age (Gyr)",                    "plasma",   False),
        ("stellarmetallicity", "Metallicity",                          "viridis",  True),
        ("dustmass",           r"Dust mass (M$_\odot$ pc$^{-2}$)",    "cividis",  True),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(20, 5), facecolor="black")
    half = fov_kpc / 2

    for ax, (name, label, cmap, use_log) in zip(axes, maps):
        img = load_fits_image(skirt_file(subhalo_id, orientation, name,
                                          base_dir=base_dir))
        img = np.clip(img, 1e-10, None)
        arr = np.log10(img) if use_log else img

        im = ax.imshow(arr, origin="lower", cmap=cmap,
                       extent=[-half, half, -half, half])
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = fig.colorbar(im, cax=cax)
        scale = "log " if use_log else ""
        cbar.set_label(f"{scale}{label}", color="white", fontsize=8)
        cbar.ax.yaxis.set_tick_params(color="white")
        plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")
        ax.set_title(label.split("(")[0].strip(), color="white", fontsize=11)
        ax.set_xlabel("x (kpc)", color="white", fontsize=8)
        ax.set_ylabel("y (kpc)", color="white", fontsize=8)
        ax.tick_params(colors="white", labelsize=7)
        ax.set_facecolor("black")
        for spine in ax.spines.values():
            spine.set_edgecolor("white")
        if xlim:
            ax.set_xlim(xlim)
        if ylim:
            ax.set_ylim(ylim)

    fig.patch.set_facecolor("black")
    fig.suptitle(f"Galaxy {subhalo_id} — Physical Property Maps",
                 color="white", fontsize=13)
    plt.tight_layout()
    plt.show()
