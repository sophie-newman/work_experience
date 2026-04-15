# Galaxy Explorer — A Day in the Life of an Astrophysicist

Welcome! By the end of today you will have:
- Written your first Python code
- Understood why astronomers need computers
- Downloaded real data from a cosmological simulation
- Run the same software that Sophie uses in her PhD research
- Made your own images of simulated galaxies

---

## What You Need

- A laptop (Windows, Mac, or Linux)
- Internet access
- About 1 GB of free disk space
- The small grid file that Sophie will give you (it is only 1.1 MB)

---

## Installation — Step by Step

### Step 1: Install Anaconda (Python + JupyterLab in one go)

Anaconda is a free, all-in-one installer. It gives you Python,
JupyterLab, numpy, matplotlib, and hundreds of other tools.

1. Go to **https://www.anaconda.com/download**
2. Download the installer for your operating system
3. Run the installer — the default options are all fine
   - On **Windows**: when asked about "Add to PATH", tick **Yes**
   - On **Mac/Linux**: the installer handles this automatically
4. The install takes about 5–10 minutes

### Step 2: Open a Terminal

- **Windows**: search for **Anaconda Prompt** in the Start menu and open it
- **Mac**: open **Terminal** (search with Cmd+Space → "Terminal")
- **Linux**: open your terminal

### Step 3: Install the Extra Packages

Type these commands one at a time, pressing **Enter** after each:

```bash
pip install cosmos-synthesizer
pip install astropy
pip install h5py
pip install requests
```

`cosmos-synthesizer` is the software Sophie uses every day to model galaxies.
The others are standard astronomy tools.

If `pip install cosmos-synthesizer` fails or takes more than a minute,
try this instead:

```bash
conda install -c conda-forge astropy h5py requests
pip install cosmos-synthesizer
```

### Step 4: Get a Free TNG API Key

The galaxy data comes from the IllustrisTNG project. You need Sophie's API key
to use it, which you will paste this into Notebook 4 when prompted.

### Step 5: Copy the Grid File

Sophie will give you a file called:

```
maraston24-Te00_kroupa-0.1,100.hdf5
```

This is the **Stellar Population Synthesis (SPS) grid** — a lookup table
that tells Synthesizer what spectrum a population of stars produces
depending on its age and chemical composition.

Copy this file into a folder called `grids` inside `galaxy_explorer`:

```
galaxy_explorer/
  grids/
    maraston24-Te00_kroupa-0.1,100.hdf5   ← put it here
  01_welcome_python.ipynb
  02_astronomy_code.ipynb
  ...
```

### Step 6: Launch JupyterLab

In your terminal (Anaconda Prompt on Windows), navigate to this folder:

```bash
cd path/to/galaxy_explorer
```

For example, if you put it on the Desktop:
- **Windows**: `cd Desktop\galaxy_explorer`
- **Mac/Linux**: `cd ~/Desktop/galaxy_explorer`

Then launch JupyterLab:

```bash
jupyter lab
```

A browser window will open. If it does not, look in the terminal for a line
like `http://localhost:8888/lab?token=...` and paste that URL into your browser.

---

## Notebook Order

Work through these **in order**. Each one builds on the previous.

| Notebook | Topic | Time |
|---|---|---|
| `01_welcome_python.ipynb` | Python basics — variables, loops, functions, plotting | ~40 min |
| `02_astronomy_code.ipynb` | The scale of the universe; why computers are essential | ~25 min |
| `03_tng_synthesizer.ipynb` | What TNG is; what Synthesizer is; forward modelling | ~20 min |
| `04_galaxy_images.ipynb` | Download TNG data and run Synthesizer yourself | ~45 min |
| `05_explore_discover.ipynb` | Change parameters and discover the physics | ~60 min |

---

## How to Run a Notebook Cell

- Click a cell to select it
- Press **Shift + Enter** to run it and jump to the next cell
- Press **Ctrl + Enter** to run it and stay on the same cell
- `[*]` next to a cell means it is running; a number `[3]` means it finished

---

## Troubleshooting

**"ModuleNotFoundError: No module named 'synthesizer'"**
Run `pip install synthesizer-astro` in your Anaconda Prompt.

**"401 Unauthorized" when downloading**
Your TNG API key is wrong or missing. Re-read Step 4 above.

**"FileNotFoundError: grids/maraston24..."**
You have not put the grid file in the `grids/` folder. Re-read Step 5.

**Kernel died / notebook crashed**
Go to **Kernel → Restart Kernel** in JupyterLab, then run cells again from the top.

**Something else**
Ask Sophie!

---

*Created for a student visit to the Institute for Computational Cosmology,
Durham University.*
