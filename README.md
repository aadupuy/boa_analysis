# 🌀 `boa_analysis` — BoA Segmentation Analysis & Visualization Toolkit

### Basin of Attraction (BoA) analysis of the large-scale structure  
*A modular Python toolkit for analyzing, visualizing, and comparing the cosmic web’s basins of attraction (BoA) from the CLUES and CF4 simulation pipelines.*

---

## 📦 Repository Structure

```bash
cosmic_basins/
│
├── analysis/
│   ├── number_basins.py          # Count of BoAs vs smoothing radius & redshift
│   ├── cumulative_mass_allz.py   # Cumulative mass distributions of basins
│   ├── percent_mass.py           # 5%, 50%, 95% basin mass percentiles vs R_s
│   ├── mass_function.py          # Cumulative mass function N(<M)
│   └── meandensity_rs_snapshots.py  # Weighted mean density vs R_s and z
│
├── visualization/
│   ├── visualizations_rs.py      # BoA slices across smoothing radii R_s
│   ├── visualizations_z.py       # BoA slices across redshifts z
│   └── visualizations_resolution.py # BoA comparison at different grid resolutions
│
├── utils/
│   └── functions.py              # Random colormap generator and misc helpers
│
├── data/                         # Cached .npy arrays (computed statistics)
├── notebooks/
│   └── visualization_demo.ipynb  # Example Jupyter notebook showing all plots
│
└── README.md
```

---

## 🌌 Overview

This repository provides a complete pipeline to analyze the **segmentation of the cosmic web** into **Basins of Attraction (BoA)** from velocity-field reconstructions and simulations.

The analysis scripts quantify:
- 🧮 Number and evolution of basins with scale and redshift  
- 📊 Basin mass functions and cumulative mass distributions  
- ⚖️ Mean density contrasts per basin (ρ/ρ̄)  
- 🔍 Dependence on grid resolution and smoothing scale  

The visualization suite produces consistent 2D mid-plane projections that reveal how large-scale cosmic flows evolve with redshift and smoothing scale.

---

## 🧠 Features

- **Modular analysis scripts** with reproducible data handling  
- **Unified plotting style** using Matplotlib (publication-ready)  
- **Weighted quantiles** and cumulative mass functions for BoA statistics  
- **Consistent colormap remapping** across redshifts and smoothing scales  
- **`scipy.io.readsav`-based I/O** for IDL `.sav` field data  
- **Caching system** to save computed arrays and avoid reprocessing  
- **Jupyter integration** for demo and interactive exploration  

---

## ⚙️ Requirements

| Package | Minimum version |
|----------|-----------------|
| `numpy` | 1.22 |
| `matplotlib` | 3.5 |
| `pandas` | 1.4 |
| `scipy` | 1.8 |
| `tqdm` *(optional)* | 4.65 |

To install all dependencies:

```bash
pip install -r requirements.txt
```

---

## 📘 Example Usage

Some examples can be found in `demo.ipynb`.

---

## 🧭 Citation 

These scripts were developed as part of the following publication:
Dupuy A., Courtois H. M., Libeskind N. I., Guinet D., (2020)
*Segmenting the Universe into dynamically coherent basins*,
MNRAS, 493(3), 3513–3520.