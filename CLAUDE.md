# CLAUDE.md

CliffDelineaTool 2.0: a CNN-BiLSTM-attention model that finds the cliff toe (base) and top on elevation transects. It extends Swirad & Young (2022, v1.2). The goal is a JOSS submission.

## Layout
- `v2/cliff_dl/` is the package (data, models, training, inference); `v2/scripts/` is the numbered pipeline.
- `v1/` and `datasets/` (AOIs 1–8) are Swirad's and have no named licence. Don't edit, move or relicense them.

## Commands (run from `v2/scripts/`; add `--device cpu` on a Mac)
```bash
python 01_prepare_data.py
python 02_train_model.py
python 03_evaluate_model.py --checkpoint experiments/runs/checkpoints/best_model.pth
python 04_predict.py --checkpoint experiments/runs/checkpoints/best_model.pth --aoi_path ../../datasets/validation/aoi5
```
There is no test suite yet; `v2/test_v2_minimal.py` only prints.

## Invariants
- The published model is `experiments/runs/checkpoints/best_model.pth`: epoch 10, trained on AOIs 1–3, validated on 4, tested on 5–8, sha256 `e2ea5746…caf1`. It is gitignored. Never overwrite it or retrain into that folder. `runs_all_aois/` is a failed run (epoch 0).
- Until v2.0.0 is tagged and archived on Zenodo, no change may alter model outputs (preprocessing, model or inference).
- cliffCast imports `cliff_dl` from this folder (`../CliffDelineaTool_2.0/v2`), so changes here change cliffCast reruns.
- Each point has 13 features (`preprocessing.compute_features`); `n_vert` = 20 must match v1.

## Known issues
- Post-processing makes test error worse (toe MAE 2.0 m raw, 25.5 m after).
- `compare_v1_v2.py`, `create_aoi_gifs.py` and `run_v1_validation.py` hardcode `/Users/cjmack/...`.
