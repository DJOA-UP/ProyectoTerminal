import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import random
from mpl_toolkits.mplot3d import Axes3D

# ---------- STEP 1: Load your CSV ----------
df = pd.read_csv("synthetic_data.csv")  # Must contain 'SpO2', 'HR', 'Temp'
data = df[['SpO2', 'HR', 'Temp']].values

# ---------- STEP 2: Normalize ----------
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)
print("Means:", scaler.mean_)
print("Stds:", scaler.scale_)

# ---------- STEP 3: Clustering ----------
kmeans = KMeans(n_clusters=4, random_state=42)
cluster_labels = kmeans.fit_predict(data_scaled)

# === Print cluster centers in original scale ===
centers_original = scaler.inverse_transform(kmeans.cluster_centers_)
for i, center in enumerate(centers_original):
    print(f"Cluster {i}: SpO2={center[0]:.1f}, HR={center[1]:.1f}, Temp={center[2]:.1f}")

# ---------- STEP 4: Assign pseudo-labels ----------
cluster_to_label = {
    0: [1.0, 0.0, 0.0, 0.0], #Cluster 0 → Calm (SpO2=97.7, HR=81.6, Temp=37.4)
    1: [0.0, 1.0, 0.0, 0.0], #Cluster 1 → Activity Need (SpO2=97.0, HR=70.7, Temp=36.7)
    2: [0.0, 0.0, 1.0, 0.0], #Cluster 2 → Stress (SpO2=92.5, HR=106.9, Temp=37.6)
    3: [0.0, 0.0, 0.0, 1.0], #Cluster 3 → Fatigue (SpO2=93.6, HR=52.8, Temp=37.3)
}

labels = np.array([cluster_to_label[c] for c in cluster_labels])

# ---------- STEP 5: Train-test split ----------
X_train, X_test, y_train, y_test = train_test_split(
    data_scaled, labels, test_size=0.3, random_state=42
)

# ---------- STEP 6: Define neural network ----------
class LinearNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3, 4)

    def forward(self, x):
        return self.linear(x)

# ---------- STEP 7: Prepare tensors ----------
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

# ---------- STEP 8: Train neural network ----------
model = LinearNN()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

for epoch in range(200):
    model.train()
    output = model(X_train_tensor)
    loss = criterion(output, y_train_tensor)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 20 == 0:
        print(f"Epoch {epoch}, Train Loss: {loss.item():.4f}")

# ---------- STEP 9: Evaluate on test data ----------
model.eval()
with torch.no_grad():
    test_output = model(X_test_tensor)
    test_loss = criterion(test_output, y_test_tensor)
    print(f"\nTest Loss: {test_loss.item():.4f}")

# ---------- STEP 10: Prediction function ----------
def predict_state(spo2, hr, temp):
    x_input = np.array([[spo2, hr, temp]])
    x_scaled = scaler.transform(x_input)
    x_tensor = torch.tensor(x_scaled, dtype=torch.float32)
    model.eval()
    with torch.no_grad():
        prediction = model(x_tensor).numpy()[0]
    return prediction

# ---------- STEP 11: Predict on a random real test sample ----------
random_idx = random.randint(0, len(X_test) - 1)
input_scaled = X_test[random_idx]
input_unscaled = scaler.inverse_transform([input_scaled])[0]
spo2_val, hr_val, temp_val = input_unscaled

# Prediction
pred = predict_state(spo2_val, hr_val, temp_val)
true_label = y_test[random_idx]

print(f"\nRandom Test Sample #{random_idx}")
print(f"Input → SpO2: {spo2_val:.1f}, HR: {hr_val:.1f}, Temp: {temp_val:.1f}")
print(f"Prediction → Calm: {pred[0]:.2f}, Activity Need: {pred[1]:.2f}, Stress: {pred[2]:.2f}, Fatigue: {pred[3]:.2f}")
print(f"True Label → Calm: {true_label[0]:.2f}, Activity Need: {true_label[1]:.2f}, Stress: {true_label[2]:.2f}, Fatigue: {true_label[3]:.2f}")

# ---------- STEP 12: Confusion Matrix for Dominant State ----------
y_true_classes = np.argmax(y_test, axis=1)
y_pred_classes = np.argmax(test_output.numpy(), axis=1)

cm = confusion_matrix(y_true_classes, y_pred_classes)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Calm", "Activity Need", "Stress", "Fatigue"])
disp.plot(cmap='Blues')
plt.title("Confusion Matrix (Test Set)")
plt.show()

# ---------- STEP 13: 3D Visualization of the clusters ----------


fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(
    data_scaled[:, 0],  # SpO2
    data_scaled[:, 1],  # HR
    data_scaled[:, 2],  # Temp
    c=cluster_labels,
    cmap='viridis',
    alpha=0.5
)
# Plot cluster centers
ax.scatter(
    kmeans.cluster_centers_[:, 0],
    kmeans.cluster_centers_[:, 1],
    kmeans.cluster_centers_[:, 2],
    s=300, c='red', marker='X', label='Centers'
)
ax.set_title("KMeans Clustering (3D)")
ax.set_xlabel("SpO2 (scaled)")
ax.set_ylabel("HR (scaled)")
ax.set_zlabel("Temp (scaled)")
fig.colorbar(scatter, label='Cluster Label')
plt.legend()
plt.show()
torch.save(model.state_dict(), "Cluster_Model.pth")
print("Model saved as Cluster_Model.pth")