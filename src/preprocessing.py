import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import os

# Define the output folder
output_folder = r"C:\Users\sofu0\code\beatbox_to_midi\output_images"
os.makedirs(output_folder, exist_ok=True)

# Load the audio file (replace with your actual file path)
audio_file = r"C:\Users\sofu0\code\beatbox_to_midi\data\beatbox1.wav"
y, sr = librosa.load(audio_file, sr=None)

# 1. Extracting MFCCs (Mel-frequency cepstral coefficients)
mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
print("MFCC shape:", mfccs.shape)

# Plotting MFCCs and saving the image
plt.figure(figsize=(10, 4))
librosa.display.specshow(mfccs, x_axis='time', sr=sr)
plt.colorbar()
plt.title('MFCC')
plt.tight_layout()
mfcc_output_path = os.path.join(output_folder, 'mfcc.png')
plt.savefig(mfcc_output_path)
plt.close()  # Close the plot to free up memory

# 2. Extracting Spectral Features (Zero-crossing rate, spectral centroid, etc.)
zero_crossings = librosa.feature.zero_crossing_rate(y)
spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)
spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)

# Plotting spectral features and saving the image
frames = range(len(spectral_centroids[0]))
t = librosa.frames_to_time(frames, sr=sr)

plt.figure(figsize=(10, 6))
plt.subplot(3, 1, 1)
plt.plot(t, zero_crossings[0], label='Zero-crossing rate')
plt.legend()

plt.subplot(3, 1, 2)
plt.semilogy(t, spectral_centroids[0], label='Spectral Centroid')
plt.legend()

plt.subplot(3, 1, 3)
plt.semilogy(t, spectral_bandwidth[0], label='Spectral Bandwidth')
plt.legend()

plt.tight_layout()
spectral_output_path = os.path.join(output_folder, 'spectral_features.png')
plt.savefig(spectral_output_path)
plt.close()

# 3. Creating a Mel Spectrogram
S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)

# Converting power spectrogram (amplitude squared) to decibel (log scale)
S_dB = librosa.power_to_db(S, ref=np.max)

# Plotting Mel Spectrogram and saving the image
plt.figure(figsize=(10, 4))
librosa.display.specshow(S_dB, x_axis='time', y_axis='mel', sr=sr, fmax=8000)
plt.colorbar(format='%+2.0f dB')
plt.title('Mel Spectrogram')
plt.tight_layout()
mel_spectrogram_output_path = os.path.join(output_folder, 'mel_spectrogram.png')
plt.savefig(mel_spectrogram_output_path)
plt.close()

print(f"Images saved to: {output_folder}")
