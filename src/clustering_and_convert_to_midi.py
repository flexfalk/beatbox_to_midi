import numpy as np
import librosa
import os
from sklearn.cluster import KMeans
import mido
import matplotlib.pyplot as plt

# Load the audio file
audio_file = r"C:\Users\sofu0\code\beatbox_to_midi\data\beatbox1.wav"
y, sr = librosa.load(audio_file, sr=None)

# Parameters for frame length and hop length
frame_length = 2048
hop_length = 512

# Extract MFCC features
mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
mfccs = mfccs.T  # Transpose to have time steps in rows

# Clustering using K-Means
num_clusters = 3
kmeans = KMeans(n_clusters=num_clusters)
labels = kmeans.fit_predict(mfccs)

# Define MIDI note mapping
midi_mapping = {
    0: 36,  # Kick
    1: 38,  # Snare
    2: 42   # Hi-hat
}

# Initialize data structures to store note timing
times = []
sounds = []

# Calculate the time for each segment
for i in range(len(labels)):
    # Calculate time in ticks based on hop length and sample rate
    time_in_ticks = (i * hop_length) / (sr / 1000)  # Convert to milliseconds
    times.append(time_in_ticks)
    sounds.append(midi_mapping[labels[i]])  # Map cluster label to MIDI note

# Create a MIDI file
mid = mido.MidiFile()
track = mido.MidiTrack()
mid.tracks.append(track)

# Parameters for MIDI creation
duration = 480  # Duration of each note in ticks
time = 0  # Start time

# Save only the first occurrence of each consecutive sound
previous_sound = None
for i in range(len(labels)):
    current_sound = sounds[i]
    if current_sound != previous_sound:
        # Create a note on message
        track.append(mido.Message('note_on', note=current_sound, velocity=64, time=time))
        # Create a note off message after the duration
        track.append(mido.Message('note_off', note=current_sound, velocity=64, time=duration))
        
        time += duration  # Increment time for the next note
        previous_sound = current_sound  # Update previous sound

# Save the MIDI file
midi_file_path = r"C:\Users\sofu0\code\beatbox_to_midi\output_audio\beatbox_midi.mid"
mid.save(midi_file_path)

print(f"MIDI file saved to: {midi_file_path}")

# Visualization
plt.figure(figsize=(12, 6))

# Reset previous_sound for plotting
previous_sound = None

# Loop through the sounds and plot them, but only the first occurrence of consecutive sounds
for i in range(len(sounds)):
    current_sound = sounds[i]
    if current_sound != previous_sound:
        plt.plot([times[i], times[i]], [0, current_sound], marker='o', markersize=10, label=f'Sound: {current_sound}')
        previous_sound = current_sound  # Update previous sound

# Set plot labels and title
plt.yticks(list(midi_mapping.values()), list(midi_mapping.keys()))
plt.xlabel('Time (ms)')
plt.title('Detected Sounds Timeline Visualization (Reduced)')
plt.grid()

# Save the visualization to a file
output_image_path = r"C:\Users\sofu0\code\beatbox_to_midi\output_audio\midi_visualization.png"
plt.savefig(output_image_path, bbox_inches='tight')
print(f"MIDI visualization saved to: {output_image_path}")

# Show the plot (optional)
plt.show()
