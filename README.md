# Hammer Throw Distance Estimation

Source code for hammer throw distance estimation using deep learning and physics-based analysis.

Project page: https://ahmedeh28.github.io/hammerthrow/

## Overview

- Detect hammer head position from side and back videos.
- Reconstruct 3D trajectory using camera calibration points.
- Estimate release parameters and throw distance.
- Run analysis through a web interface.

## Requirements

- Python 3.10+
- Docker with Compose plugin 
- Hammer head detection model files in `models/`

## Run With Docker Compose 

```bash
docker compose up --build -d
```

Open: http://localhost:5000/



## Run locally

```bash
pip install -r requirements.txt
python app.py
```

Open: http://localhost:<PORT>/ (default: 5000)

## Required Input Files

- Synchronized side-view and back-view videos
- Calibration file (`.csv` or `.txt`)
- Model weights (`.pt`) in `models/`

## Citation

If you use this repository in research, please cite:

```bibtex
@article{hasen2026hammer,
    title   = {Hammer Throw Distance Estimation Using Deep Learning and Physics-Based Modeling},
    author  = {Hasen, Ahmed Endris and Passalis, Nikolaos and V{"a}nttinen, Tomi and Raitoharju, Jenni},
    journal = {Expert Systems with Applications},
    year    = {2026},
    pages   = {132022},
    issn    = {0957-4174},
    doi     = {10.1016/j.eswa.2026.132022},
    url     = {https://www.sciencedirect.com/science/article/pii/S0957417426009358}
}
```











