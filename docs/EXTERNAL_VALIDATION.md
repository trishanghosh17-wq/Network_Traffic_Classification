# External validation protocol

1. Obtain a labeled CSV from a different capture, network, or dataset source.
2. Ensure the 30 flow features use the same definitions/units as the training schema.
3. Run `python src/evaluate_external.py path/to/external.csv`.
4. Record accuracy, precision, recall, F1, ROC-AUC, PR-AUC, FPR, FNR and the confusion matrix.
5. Compare the external result with `results/final_metrics.json` without merging the datasets.
6. Investigate class-specific failures before considering deployment.

For a stronger claim, repeat this with a temporal holdout and a production shadow test.
