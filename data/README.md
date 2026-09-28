# Data

The repository intentionally does not commit the full CICIDS2017 raw dataset.

## Binary DDoS model
Place the DDoS CSV at `data/DDos.pcap_ISCX.csv` and run `python src/train.py`.

## Multi-class extension
Place the additional CICIDS2017 attack CSV files in `data/cicids2017_multiclass/` and run `python src/multiclass_train.py`.

Keep the dataset license/usage terms of the original CICIDS2017 distribution.
