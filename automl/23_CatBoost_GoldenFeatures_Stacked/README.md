# Summary of 23_CatBoost_GoldenFeatures_Stacked

[<< Go back](../README.md)


## CatBoost
- **n_jobs**: -1
- **learning_rate**: 0.1
- **depth**: 7
- **rsm**: 0.8
- **loss_function**: RMSE
- **eval_metric**: MAE
- **explain_level**: 0

## Validation
 - **validation_type**: kfold
 - **k_folds**: 5
 - **shuffle**: True
 - **random_seed**: 42

## Optimized metric
mae

## Training time

8.1 seconds

### Metric details:
| Metric   |           Score |
|:---------|----------------:|
| MAE      |    73.1521      |
| MSE      | 49573.8         |
| RMSE     |   222.652       |
| R2       |     0.9241      |
| MAPE     |     5.01562e+15 |



## Learning curves
![Learning curves](learning_curves.png)
## True vs Predicted

![True vs Predicted](true_vs_predicted.png)


## Predicted vs Residuals

![Predicted vs Residuals](predicted_vs_residuals.png)



[<< Go back](../README.md)
