import json
from pathlib import Path
import joblib
import numpy as np
from ucimlrepo import fetch_ucirepo

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# 1. Load dataset
wine = fetch_ucirepo(id=186)
X = wine.data.features
y = wine.data.targets.values.ravel()

# 2. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# 3. Preprocessing (change this per experiment)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 4. Model (change this per experiment)
model = LinearRegression()

# 5. Train
model.fit(X_train, y_train)

# 6. Evaluate
preds = model.predict(X_test)
mse = mean_squared_error(y_test, preds)
r2 = r2_score(y_test, preds)

# 7. Print metrics (MANDATORY for Actions)
print(f"MSE: {mse}")
print(f"R2: {r2}")

# 8. Save outputs
Path("output").mkdir(parents=True, exist_ok=True)
joblib.dump(model, "output/model.joblib")

results = {
    "mse": mse,
    "r2_score": r2
}

with open("output/results.json", "w") as f:
    json.dump(results, f, indent=4)
