# Glucose Prediction Project

Quick overview of what is in this repo and what each file does.

## Main Files

- `glucose_pipeline.py`  
  Main code file. It trains the model and also runs predictions.

- `Preprocessed Data (1).xlsx`  
  Your original dataset (input data used for training).

## Artifacts Folder

- `artifacts/best_glucose_model.joblib`  
  The final trained model file (best run).

- `artifacts/full_dataset_predictions_90plus.csv`  
  Predicted glucose values generated from the best model setup.

- `artifacts/metrics_report.json`  
  Full detailed metrics from all tested model/config runs.

- `artifacts/report_readable.txt`  
  Short, readable summary of the best model and key performance numbers.

## What To Use First

If you just want the final result:

1. Use `artifacts/best_glucose_model.joblib` as the model.
2. Check predictions in `artifacts/full_dataset_predictions_90plus.csv`.
3. Read quick metrics in `artifacts/report_readable.txt`.

## Notes

- The 90%+ accuracy setup was achieved with aggressive filtering of the dataset range.
- For broad real-world use, you may want to retrain with less aggressive filtering.