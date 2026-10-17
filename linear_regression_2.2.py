import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

folder = Path(__file__).parent
file_path = folder / "linear_regression_dataset-2.csv"
data = pd.read_csv(file_path)

target = "goals_scored"

predictors = [
    "avg_goals_qlf",
    "avg_goals_agnst_qlf",
    "host_status",
    "trgt_tm_fifa_rank",
    "opp_tm_fifa_rank",
    "avg_goal_history",
    "total_team_caps",
    "team_avg_age"
]

model_data = data[[target] + predictors].apply(
    pd.to_numeric, errors="coerce"
).replace([np.inf, -np.inf], np.nan).dropna()
X = model_data[predictors]
y = model_data[target]
model = LinearRegression()

cv = KFold(n_splits=5, shuffle=True, random_state=42)
predictions = cross_val_predict(model, X, y, cv=cv)

mae = mean_absolute_error(y, predictions)
rmse = np.sqrt(mean_squared_error(y, predictions))
r2 = r2_score(y, predictions)

baseline = np.full(len(y), y.mean())
baseline_mae = mean_absolute_error(y, baseline)
baseline_rmse = np.sqrt(mean_squared_error(y, baseline))

model.fit(X, y)

formula = f"\npredicted goals = {model.intercept_:.4f}"

for name, coefficient in zip(predictors, model.coef_):
    sign = "+" if coefficient >= 0 else "-"
    formula += f" {sign} {abs(coefficient):.4f} * {name}"

print(formula)

coefficients = pd.DataFrame({
    "Variable": predictors,
    "Coefficient": model.coef_
})

print("\nCoefficients:")
print(coefficients.round(4).to_string(index=False))

print("\n5-fold cross-validated results:")
print(f"MAE: {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R-squared: {r2:.4f}")

print("\nBaseline:")
print(f"MAE: {baseline_mae:.4f}")
print(f"RMSE: {baseline_rmse:.4f}")
