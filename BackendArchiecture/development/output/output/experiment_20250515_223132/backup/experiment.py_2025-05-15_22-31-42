import numpy as np
import pandas as pd
try:
    from sklearn.model_selection import train_test_split, cross_val_score, KFold
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_squared_error
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.impute import SimpleImputer
    import xgboost as xgb
except ImportError as e:
    print(f"Error importing libraries: {e}. Please make sure you have scikit-learn and xgboost installed.")
    exit()

# Generate a synthetic dataset (replace with your actual data loading)
# For demonstration purposes, creating a dummy dataset that includes 'Address'
np.random.seed(42)
X = np.random.rand(1000, 10)  # 1000 samples, 10 features
y = np.random.rand(1000)  # 1000 target values

# Introduce some missing values
missing_indices = np.random.choice(1000, size=50, replace=False)
X[missing_indices, 0] = np.nan

# Create a dummy 'Address' column
address = ['Address_' + str(i) for i in range(1000)]

# Convert to Pandas DataFrame for easier handling
X = pd.DataFrame(X)
X['Address'] = address
y = pd.Series(y)

# Load the dataset and drop the 'Address' column
try:
    # Replace with your actual data loading, e.g., pd.read_csv('your_data.csv')
    # For this example, using the dummy data
    data = X.copy()
    data['target'] = y
    df = data.copy()

    if 'Address' in df.columns:
        df = df.drop('Address', axis=1)
    else:
        print("'Address' column not found in the dataset.")
        
    X = df.drop('target', axis=1)
    y = df['target']

except FileNotFoundError:
    print("Dataset file not found. Please provide the correct path.")
    exit()
except KeyError as e:
    print(f"KeyError: {e}. Make sure the target column is named 'target'.")
    exit()
except Exception as e:
    print(f"An error occurred during data loading and preprocessing: {e}")
    exit()


# Function to remove outliers using the IQR method
def remove_outliers_iqr(X, y, threshold=1.5):
    """Removes outliers from the target variable using the IQR method.

    Args:
        X (pd.DataFrame): The feature matrix.
        y (pd.Series): The target variable.
        threshold (float): The IQR threshold for outlier detection.

    Returns:
        pd.DataFrame: The feature matrix without outliers.
        pd.Series: The target variable without outliers.
    """
    Q1 = y.quantile(0.25)
    Q3 = y.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - threshold * IQR
    upper_bound = Q3 + threshold * IQR
    
    # Identify outliers
    outlier_indices = y[(y < lower_bound) | (y > upper_bound)].index
    
    # Remove outliers from both X and y
    X_filtered = X.drop(outlier_indices)
    y_filtered = y.drop(outlier_indices)
    
    return X_filtered, y_filtered


# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Linear Regression model on the original data
model_original = LinearRegression()
imputer_original = SimpleImputer(strategy='mean')
X_train_imputed_original = imputer_original.fit_transform(X_train)
X_test_imputed_original = imputer_original.transform(X_test)
model_original.fit(X_train_imputed_original, y_train)
y_pred_original = model_original.predict(X_test_imputed_original)
rmse_original = np.sqrt(mean_squared_error(y_test, y_pred_original))
print("RMSE on original data:", rmse_original)

# Apply outlier removal to the training data
X_train_filtered, y_train_filtered = remove_outliers_iqr(X_train, y_train)

# Data scaling for outlier removed data
scaler = StandardScaler()

# Impute missing values using the mean
imputer = SimpleImputer(strategy='mean')
X_train_imputed = imputer.fit_transform(X_train_filtered)
X_test_imputed = imputer.transform(X_test)


X_train_scaled = scaler.fit_transform(X_train_imputed)
X_test_scaled = scaler.transform(X_test_imputed)  # Use the same scaler fitted on the training data
X_train_scaled = pd.DataFrame(X_train_scaled, index=X_train_filtered.index, columns=X_train_filtered.columns)
X_test_scaled = pd.DataFrame(X_test_scaled, index=X_test.index, columns=X_test.columns)


# Train a Linear Regression model on the outlier-removed data
model_filtered = LinearRegression()
model_filtered.fit(X_train_scaled, y_train_filtered)

# Make predictions on the test set (scaled)
y_pred_filtered = model_filtered.predict(X_test_scaled)

# Evaluate the model on the test set
rmse_filtered = np.sqrt(mean_squared_error(y_test, y_pred_filtered))
print("RMSE on outlier-removed data:", rmse_filtered)


# Random Forest Regressor with Cross-Validation
rf_model = RandomForestRegressor(random_state=42)

# Impute missing values for the entire dataset before cross-validation
imputer_rf = SimpleImputer(strategy='mean')
X_imputed = imputer_rf.fit_transform(X)
X_imputed = pd.DataFrame(X_imputed, columns=X.columns)  # Convert back to DataFrame

# 5-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(rf_model, X_imputed, y, cv=kf, scoring='neg_mean_squared_error')

# Convert scores to RMSE
rmse_scores = np.sqrt(-cv_scores)

# Report the average RMSE
print("Average RMSE from 5-fold cross-validation (Random Forest):", rmse_scores.mean())

# Experiment 1: XGBoost Regressor with default hyperparameters
# Load the dataset and drop the 'Address' column (repeated for clarity)
try:
    data = X.copy()
    data['target'] = y
    df = data.copy()

    if 'Address' in df.columns:
        df = df.drop('Address', axis=1)
    else:
        print("'Address' column not found in the dataset.")
        
    X = df.drop('target', axis=1)
    y = df['target']

except FileNotFoundError:
    print("Dataset file not found. Please provide the correct path.")
    exit()
except KeyError as e:
    print(f"KeyError: {e}. Make sure the target column is named 'target'.")
    exit()
except Exception as e:
    print(f"An error occurred during data loading and preprocessing: {e}")
    exit()


# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Impute missing values
imputer_xgb = SimpleImputer(strategy='mean')
X_train_imputed_xgb = imputer_xgb.fit_transform(X_train)
X_test_imputed_xgb = imputer_xgb.transform(X_test)


# Instantiate XGBoost Regressor with default hyperparameters
xgb_model = xgb.XGBRegressor(random_state=42)

# Train the model on the training data
xgb_model.fit(X_train_imputed_xgb, y_train)

# Make predictions on the test set
y_pred_xgb = xgb_model.predict(X_test_imputed_xgb)

# Evaluate the model's performance on the test set using RMSE
rmse_xgb = np.sqrt(mean_squared_error(y_test, y_pred_xgb))
print("RMSE on test set (XGBoost with default hyperparameters):", rmse_xgb)