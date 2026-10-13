mport pandas as pd
import numpy as np
from pathlib import Path

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

folder = Path(__file__).parent
file_path = folder / "dataset_21.csv"

if not file_path.exists():
    raise FileNotFoundError(
        "dataset_21.csv must be in the same folder as this program."
    )
