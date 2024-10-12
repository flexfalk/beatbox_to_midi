import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Define the output folder
output_folder = r"C:\Users\sofu0\code\beatbox_to_midi\output_images"
os.makedirs(output_folder, exist_ok=True)

# Load the audio file (replace with your actual file path)
audio_file = r"C:\Users\sofu0\code\beatbox_to_midi\data\beatbox1.wav"
y, sr = librosa.load(audio_file, sr=None)

# 1. Extracting MFCCs (Mel-frequency cepstral coefficients)
mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
mfccs = mfccs.T  # Transpose to shape (time_steps, features)
print("MFCC shape:", mfccs.shape)

# 2. Standardizing the MFCC features
scaler = StandardScaler()
mfccs_scaled = scaler.fit_transform(mfccs)

# 3. Applying K-Means clustering
kmeans = KMeans(n_clusters=3, random_state=42)  # Assuming 8 different drum sounds
clusters = kmeans.fit_predict(mfccs_scaled)

# 4. Reducing dimensions with PCA for visualization
pca = PCA(n_components=2)
mfccs_reduced = pca.fit_transform(mfccs_scaled)

# Plotting the clustered MFCC features in 2D space
plt.figure(figsize=(10, 6))
scatter = plt.scatter(mfccs_reduced[:, 0], mfccs_reduced[:, 1], c=clusters, cmap='viridis', s=5)
plt.colorbar(scatter)
plt.title('K-Means Clustering of MFCCs')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.tight_layout()

# Save the clustering result plot
clustering_output_path = os.path.join(output_folder, 'mfcc_clustering.png')
plt.savefig(clustering_output_path)
plt.close()

print(f"Clustering result saved to: {clustering_output_path}")
