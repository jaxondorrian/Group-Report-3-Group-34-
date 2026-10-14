import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

folder = Path(__file__).parent
file_path = folder / "dataset_21.csv"
data = pd.read_csv(file_path)

target = "Goal Difference"

predictors = [
    "fifa_rank_diff",
    "Elo_rank_diff",
    "Qualifying_gd_gap",
    "marketVal_diff",
    "Rest_days_diff",
    "goal_diff_history",
    "avg_age_diff",
    "squad_caps_diff"
]

model_data = data[[target] + predictors].apply(
    pd.to_numeric, errors="coerce"
).replace([np.inf, -np.inf], np.nan).dropna()
X = model_data[predictors]
y = model_data[target]
model = LinearRegression()

print(f"\nMatch rows: {len(model_data)}")
print(f"target: {target}")
print(f"explanatory variables: {len(predictors)}")

cv = KFold(n_splits=5, shuffle=True, random_state=42)
predictions = cross_val_predict(model, X, y, cv=cv)

mae = mean_absolute_error(y, predictions)
rmse = np.sqrt(mean_squared_error(y, predictions))
r2 = r2_score(y, predictions)

model.fit(X, y)

formula = f"predicted diffrence = {model.intercept_:.4f}"

for name, coefficient in zip(predictors, model.coef_):
    sign = "+" if coefficient >= 0 else "-"
    formula += f" {sign} {abs(coefficient):.4f} * {name}"

print(formula)

coefficients = pd.DataFrame({
    "Variable": predictors,
    "Coeficient": model.coef_
})

print(f"MAE: {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R-squared: {r2:.4f}")
