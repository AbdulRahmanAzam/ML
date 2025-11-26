
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df1 = pd.read_csv(r"weather_forecast.csv")
df2 = pd.read_csv("solar_observed.csv")

print(df1.isnull().sum())
print('\n')
print(df2.isnull().sum())

print(df1.shape)

print(df2.shape)

if 'sky_cover' in df1.columns:
    # Keep sky_cover if present; it's often predictive of solar irradiance
    pass

# Simple mean imputation for weather features; avoid imputing timestamp
for col in ['precip_prob', 'temperature_C', 'dew_point_C', 'relative_humidity', 'wind_speed_m_s', 'sky_cover']:
    if col in df1.columns:
        df1[col] = df1[col].fillna(df1[col].mean())

print(df1.isnull().sum())


# Target imputation can bias results if missingness is systematic (e.g., nights);
# We'll drop rows where Solar is missing instead of imputing a global mean.
df2 = df2.dropna(subset=['Solar'])
df2.drop('timestamp', axis=1, inplace=True)
print(df2.isna().sum())

# renamiong the column name of df2 from local time to timestamp
df2 = df2.rename(columns={'Local_Time':'timestamp'})
df2.head()

df1['timestamp'] = pd.to_datetime(df1['timestamp'], errors='coerce', utc=True)
df2['timestamp'] = pd.to_datetime(df2['timestamp'], errors='coerce', utc=True)

print(df1.isnull().sum())
print(df2.isnull().sum())

df1 = df1.dropna(subset=['timestamp'])
df2 = df2.dropna(subset=['timestamp'])

print(df1.isnull().sum())
print(df2.isnull().sum())

df = pd.merge(df1, df2, on='timestamp', how='inner')
print('Rows df1:', len(df1), 'Rows df2:', len(df2), 'Merged rows:', len(df))

# --- Feature Engineering for Time-Based Cycles ---
df['hour'] = df['timestamp'].dt.hour
df['dayofyear'] = df['timestamp'].dt.dayofyear

# Encode cyclical features
df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24.0)
df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24.0)
df['dayofyear_sin'] = np.sin(2 * np.pi * df['dayofyear'] / 365.25)
df['dayofyear_cos'] = np.cos(2 * np.pi * df['dayofyear'] / 365.25)

print(df.head())

X = df.drop(columns=['timestamp', 'Solar', 'hour', 'dayofyear']) # Drop original time columns
y = df['Solar']

from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.svm import SVR

# Identify numeric columns for scaling
numeric_features = X.select_dtypes(include=np.number).columns.tolist()

# Create a preprocessor with ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features)
    ],
    remainder='passthrough' # Keep other columns (if any)
)


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=False
)


# --- Model Pipelines ---
lr_pipe = Pipeline([
    ('preprocessor', preprocessor),
    ('model', LinearRegression())
])

svr_pipe = Pipeline([
    ('preprocessor', preprocessor),
    ('model', SVR(kernel='rbf', C=10.0, gamma='scale'))
])

# RandomForest and HistGradientBoostingRegressor don't require scaling, but it doesn't hurt
rf_pipe = Pipeline([
    ('preprocessor', preprocessor),
    ('model', RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1))
])

hgbr_pipe = Pipeline([
    ('preprocessor', preprocessor),
    ('model', HistGradientBoostingRegressor(random_state=42))
])


print("Training models...")
lr_pipe.fit(X_train, y_train)
lr_yPred = lr_pipe.predict(X_test)

rf_pipe.fit(X_train, y_train)
rf_yPred = rf_pipe.predict(X_test)

svr_pipe.fit(X_train, y_train)
svr_yPred = svr_pipe.predict(X_test)

hgbr_pipe.fit(X_train, y_train)
hgbr_yPred = hgbr_pipe.predict(X_test)


from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def eval_model(model, y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    print(f"Model: {model:<25} - MAE: {mae:<7.2f}, RMSE: {rmse:<7.2f}, R2 Score: {r2:.2f}")

# Basic correlation check (train-only to avoid leakage in choices)
corr = pd.concat([X_train.reset_index(drop=True), y_train.reset_index(drop=True)], axis=1).corr(numeric_only=True)
print("\nTop correlations with Solar (train):")
print(corr['Solar'].sort_values(ascending=False).head(10))

print("\n--- Model Evaluation ---")
eval_model("Linear Regression", y_test, lr_yPred)
eval_model("SVR", y_test, svr_yPred)
eval_model("Random Forest", y_test, rf_yPred)
eval_model("Gradient Boosting", y_test, hgbr_yPred)
