# Summary of Ensemble_Stacked

[<< Go back](../README.md)


## Ensemble structure
| Model                              |   Weight |
|:-----------------------------------|---------:|
| 23_CatBoost_GoldenFeatures_Stacked |        3 |
| 4_Default_LightGBM                 |        1 |
| 4_Default_LightGBM_Stacked         |        1 |
| 5_Xgboost_Stacked                  |        5 |
| Ensemble                           |        4 |

### Metric details:
| Metric   |           Score |
|:---------|----------------:|
| MAE      |    71.4022      |
| MSE      | 49488.6         |
| RMSE     |   222.46        |
| R2       |     0.924231    |
| MAPE     |     2.92355e+15 |



## Learning curves
![Learning curves](learning_curves.png)
## True vs Predicted

![True vs Predicted](true_vs_predicted.png)


## Predicted vs Residuals

![Predicted vs Residuals](predicted_vs_residuals.png)



[<< Go back](../README.md)
