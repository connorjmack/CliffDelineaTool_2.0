# CliffDelineaTool 2.0

A deep learning approach to coastal cliff delineation, building on the original CliffDelineaTool by Swirad & Young (2022).

## Overview

This repository contains two versions of the CliffDelineaTool:

| Version | Approach | Location | Description |
|---------|----------|----------|-------------|
| **v2.0** | Deep Learning (CNN-BiLSTM) | [`/v2`](./v2) | PyTorch-based model requiring no manual parameter tuning |
| **v1.0** | Rule-based thresholds | [`/v1`](./v1) | Original MATLAB/Python implementation by Swirad & Young |

**v2.0 is the recommended version** for new projects. It achieves better accuracy without requiring manual calibration per study site.

## Quick Start (v2.0)

```bash
cd v2
pip install -r requirements.txt

# Prepare data, train, and evaluate
python scripts/01_prepare_data.py --config config/default_config.yaml
python scripts/02_train_model.py --config config/default_config.yaml
python scripts/03_evaluate_model.py --checkpoint experiments/runs/checkpoints/best_model.pth
```

See [`v2/README.md`](./v2/README.md) for full documentation.

## v2.0 vs v1.0 Comparison

| Aspect | v1.0 | v2.0 |
|--------|------|------|
| **Manual tuning** | 8+ parameters per AOI | None |
| **Approach** | Hand-crafted slope thresholds | Learned CNN-BiLSTM features |
| **Expected MAE** | 3-7m (after tuning) | 2-4m |
| **Generalization** | Requires recalibration per region | Single model works across regions |
| **Confidence scores** | No | Yes |

## Repository Structure

```
CliffDelineaTool_2.0/
├── v1/                    # Original implementation (Swirad & Young 2022)
│   ├── CliffDelineaTool.m
│   ├── CliffDelineaToolPy.py
│   └── README.md
├── v2/                    # Deep learning implementation
│   ├── cliff_dl/          # Python package
│   ├── scripts/           # Training and inference scripts
│   ├── config/            # Hyperparameter configs
│   └── README.md
└── datasets/              # Calibration and validation data
    ├── calibration/       # AOI 1-4 (training)
    └── validation/        # AOI 5-8 (testing)
```

## Citation

If you use this tool, please cite:

**Original algorithm (v1.0):**
> Swirad, Z.M. and Young, A.P., 2022. CliffDelineaTool v1.2.0: an algorithm for identifying coastal cliff base and top positions. Geoscientific Model Development, 15(4), pp.1499-1512. https://doi.org/10.5194/gmd-15-1499-2022

**Deep learning implementation (v2.0):**
> [This repository] - uses training data from Swirad & Young (2022)

## Acknowledgments

- **Original CliffDelineaTool**: Zuzanna M. Swirad (zswirad@ucsd.edu), Scripps Institution of Oceanography, UC San Diego
- **Training/validation datasets**: From Swirad & Young (2022), DOI: [10.5281/zenodo.5724975](https://doi.org/10.5281/zenodo.5724975)
- **v2.0 deep learning implementation**: Built using the calibration data from the original work

## License

See [LICENSE](./LICENSE) file.
