# Data directory

The app runs out-of-the-box on a **synthetic CMAPSS-format dataset** generated
by `pdm/data.py`, so no download is required to try it.

## Using the real NASA CMAPSS dataset

To train and evaluate on the real data, download the **FD001** subset and place
these three files here:

```
data/train_FD001.txt
data/test_FD001.txt
data/RUL_FD001.txt
```

Sources (public):
- Kaggle: https://www.kaggle.com/datasets/behrad3d/nasa-cmaps
- NASA Prognostics Data Repository (Turbofan Engine Degradation Simulation)

Then retrain:

```bash
pip install -r requirements-train.txt
python train.py
```

`pdm/data.py` auto-detects the real files and uses them instead of the synthetic
generator; everything downstream is unchanged. These `.txt` files are
git-ignored (not redistributed here).
