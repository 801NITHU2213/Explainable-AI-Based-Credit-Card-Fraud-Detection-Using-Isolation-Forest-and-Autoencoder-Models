import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap  # For interpretability
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.callbacks import EarlyStopping

# Set a random seed for reproducibility
np.random.seed(42)

# --- 1. Data Preparation (Credit Card Fraud) ---
print("--- 1. Data Preparation (Credit Card Fraud) ---")

# 1.1 Load the dataset
try:
    # Use the path from your screenshot
    data = pd.read_csv(r"C:\Users\Welcome\Downloads\creditcard.csv")
    print("Dataset loaded successfully.")
except FileNotFoundError:
    print("Error: 'creditcard.csv' not found. Please check the file path.")
    exit()

# 1.2 Scale the 'Amount' and 'Time' features
# The V1-V28 features are already scaled (from PCA).
scaler = StandardScaler()
data['scaled_Amount'] = scaler.fit_transform(data['Amount'].values.reshape(-1, 1))
data['scaled_Time'] = scaler.fit_transform(data['Time'].values.reshape(-1, 1))

# Drop the original Time and Amount columns
data = data.drop(['Time', 'Amount'], axis=1)

# 1.3 Define features (X) and labels (y)
# 'Class' is the label: 0 = Normal, 1 = Fraud
y = data['Class']
X = data.drop(['Class'], axis=1)

# Get feature names for SHAP plots
feature_names = X.columns.tolist()

# Get the number of features for the Autoencoder input
n_features = X.shape[1]

# 1.4 Calculate the fraud rate for our models
# This is crucial for setting 'contamination' and 'threshold'
fraud_rate = y.value_counts(normalize=True)[1]
print(f"Fraud Rate in dataset: {fraud_rate:.4f}")

# 1.5 Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# 1.6 Create a "normal-only" training set
# We train our models on what 'normal' (non-fraudulent) transactions look like.
X_train_normal = X_train[y_train == 0]

# Convert to numpy arrays for TensorFlow/Sklearn
X_train_normal_scaled = X_train_normal.values
X_test_scaled = X_test.values

print(f"Total features: {n_features}")
print(f"Normal-only training samples: {len(X_train_normal_scaled)}")
print(f"Test samples: {len(X_test_scaled)} (contains normal and fraud)")


# --- 2. Model 1: Isolation Forest ---
print("\n--- 2. Model 1: Isolation Forest ---")

# 2.1 Train Isolation Forest
# We set 'contamination' to our known fraud rate
iforest = IsolationForest(
    n_estimators=100,
    contamination=fraud_rate,
    random_state=42
)
iforest.fit(X_train_normal_scaled)

# 2.2 Evaluate Isolation Forest Accuracy
# .predict() gives 1 for inliers (normal) and -1 for outliers (anomalies)
y_pred_if = iforest.predict(X_test_scaled)

# Map predictions to 0 (normal) and 1 (anomaly) to match y_test
y_pred_if_mapped = np.where(y_pred_if == 1, 0, 1)

print("Accuracy Report (Isolation Forest):")
# Note: High recall for anomalies is the primary goal in fraud detection
print(classification_report(y_test, y_pred_if_mapped, target_names=['Normal (0)', 'Anomaly (1)']))

# Plot Confusion Matrix
ConfusionMatrixDisplay.from_predictions(y_test, y_pred_if_mapped, display_labels=['Normal', 'Anomaly'], cmap='Blues')
plt.title('Isolation Forest Confusion Matrix')
plt.show()


# --- 3. Model 2: Autoencoder ---
print("\n--- 3. Model 2: Autoencoder ---")

# 3.1 Define and Train Autoencoder
# Architecture can be tuned. A good start is to compress gradually.
latent_dim = 14  # Bottleneck size

# Encoder
input_layer = Input(shape=(n_features,))
encoder = Dense(20, activation='relu')(input_layer) # 29 -> 20
encoder = Dense(latent_dim, activation='relu')(encoder) # 20 -> 14

# Decoder
decoder = Dense(20, activation='relu')(encoder) # 14 -> 20
decoder = Dense(n_features, activation='linear')(decoder) # 20 -> 29

# Autoencoder model
autoencoder = Model(inputs=input_layer, outputs=decoder)
autoencoder.compile(optimizer='adam', loss='mean_squared_error')
autoencoder.summary()

# Train the model to reconstruct normal data
history = autoencoder.fit(
    X_train_normal_scaled,
    X_train_normal_scaled,
    epochs=50,
    batch_size=32,
    validation_split=0.1,
    callbacks=[EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)],
    verbose=1
)

print("Autoencoder training complete.")

# 3.2 Evaluate Autoencoder Accuracy
# First, get the reconstruction error for normal training data to find a threshold
X_train_normal_pred = autoencoder.predict(X_train_normal_scaled, verbose=0)
train_mse = np.mean(np.power(X_train_normal_scaled - X_train_normal_pred, 2), axis=1)

# Set the anomaly threshold based on the fraud rate percentile
# (e.g., if fraud_rate is 0.17%, we set the threshold at the 99.83rd percentile)
threshold = np.percentile(train_mse, (1 - fraud_rate) * 100)
print(f"Anomaly Threshold ({((1 - fraud_rate) * 100):.2f}th percentile): {threshold:.4f}")

# Now, get reconstruction error for the test set
X_test_pred = autoencoder.predict(X_test_scaled, verbose=0)
test_mse = np.mean(np.power(X_test_scaled - X_test_pred, 2), axis=1)

# Classify as anomaly (1) if error > threshold, else normal (0)
y_pred_ae = (test_mse > threshold).astype(int)

print("\nAccuracy Report (Autoencoder):")
print(classification_report(y_test, y_pred_ae, target_names=['Normal (0)', 'Anomaly (1)']))

# Plot Confusion Matrix
ConfusionMatrixDisplay.from_predictions(y_test, y_pred_ae, display_labels=['Normal', 'Anomaly'], cmap='Oranges')
plt.title('Autoencoder Confusion Matrix')
plt.show()


# --- 4. Interpretability Comparison (SHAP) ---
print("\n--- 4. Interpretability Comparison (SHAP) ---")

# We need to find one *true anomaly* that BOTH models correctly identified
true_positives_if = np.where((y_test == 1) & (y_pred_if_mapped == 1))[0]
true_positives_ae = np.where((y_test == 1) & (y_pred_ae == 1))[0]

# Find the intersection
common_true_positives = np.intersect1d(true_positives_if, true_positives_ae)

if len(common_true_positives) > 0:
    # Get the first common anomaly from the TEST set
    anomaly_index_in_test_set = common_true_positives[0]
    
    print(f"Explaining a true anomaly (Test Set index {anomaly_index_in_test_set}) identified by both models.")
    
    # Get the specific sample
    sample_to_explain = X_test_scaled[anomaly_index_in_test_set:anomaly_index_in_test_set+1] # Keep 2D shape
    
    # 4.1 Interpret Isolation Forest
    print("\nInterpreting Isolation Forest...")
    explainer_if = shap.TreeExplainer(iforest)
    
    # Get SHAP values. For iForest, these values represent the contribution
    # of each feature to the 'decision_function' score.
    shap_values_if = explainer_if.shap_values(sample_to_explain)

    # Plot the explanation
    print("Plotting iForest SHAP force plot...")
    shap.force_plot(
        explainer_if.expected_value,
        shap_values_if[0],
        sample_to_explain[0],
        feature_names=feature_names,
        matplotlib=True,
        show=False # Set to False to manage plot display
    )
    plt.title("Isolation Forest Explanation")
    plt.show()
    
    # 4.2 Interpret Autoencoder
    # We use a KernelExplainer, which is model-agnostic.
    # Our "prediction function" `f(x)` will be the reconstruction error (MSE).
    print("\nInterpreting Autoencoder (this may take a moment)...")
    
    # 1. Define the prediction function (model's output to be explained)
    def autoencoder_mse(x):
        # Ensure input is 2D and float32 for Keras
        x = x.reshape(-1, n_features).astype(np.float32)
        pred = autoencoder.predict(x, verbose=0)
        mse = np.mean(np.power(x - pred, 2), axis=1)
        return mse

    # 2. Create a "background" dataset for the explainer (a summary of normal data)
    # We use a small, representative sample of the normal data for speed
    background_data = shap.kmeans(X_train_normal_scaled, 50) # 50 cluster centers
    
    # 3. Create the explainer and get SHAP values
    explainer_ae = shap.KernelExplainer(autoencoder_mse, background_data)
    shap_values_ae = explainer_ae.shap_values(sample_to_explain)
    
    # Plot the explanation
    print("Plotting Autoencoder SHAP force plot...")
    shap.force_plot(
        explainer_ae.expected_value,
        shap_values_ae[0],
        sample_to_explain[0],
        feature_names=feature_names,
        matplotlib=True,
        show=False
    )
    plt.title("Autoencoder Explanation (Reconstruction Error)")
    plt.show()

else:
    print("No common true positive anomalies found by both models to explain.")
    print("Try re-running, or adjust model parameters (threshold/contamination).")


# --- 5. Global Feature Importance ---
print("\n--- 5. Global Feature Importance ---")

# We'll use a smaller sample of the test set for global SHAP analysis
# as it can be very slow, especially for KernelExplainer.
X_test_sample = shap.sample(X_test_scaled, 100) # 100 random samples

# 5.1 iForest: Global Importance
print("Calculating global importance for iForest...")
shap_values_if_all = shap.TreeExplainer(iforest).shap_values(X_test_sample)
    
plt.title('Global Feature Importance (Isolation Forest)')
shap.summary_plot(shap_values_if_all, X_test_sample, feature_names=feature_names)

# 5.2 Autoencoder: Global Importance
print("Calculating global importance for Autoencoder (this will be slow)...")
# Re-use the background data from before
explainer_ae_global = shap.KernelExplainer(autoencoder_mse, background_data)
shap_values_ae_all = explainer_ae_global.shap_values(X_test_sample)
    
plt.title('Global Feature Importance (Autoencoder)')
shap.summary_plot(shap_values_ae_all, X_test_sample, feature_names=feature_names)

print("\n--- Evaluation Complete ---")
