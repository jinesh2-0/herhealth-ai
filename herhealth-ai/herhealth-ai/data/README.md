# Data

## Real data (used for results)
The original Kaggle file is already included: `data/raw/PCOS_data_without_infertility.xlsx`.
It's licensed CC BY-NC-SA 4.0, which allows non-commercial sharing with attribution (see
`data/raw/ATTRIBUTION.md`). Source:
https://www.kaggle.com/datasets/prasoonkottarathil/polycystic-ovary-syndrome-pcos

Retrain: `python -m src.train --data data/raw/PCOS_data_without_infertility.xlsx`

## Synthetic data (testing only)
`python scripts/make_synthetic_data.py` writes `data/synthetic/pcos_synthetic.csv`, with invented
values and the same column names as the real file. Use it to test the pipeline. **Never report
results from it.**
