import librosa
import numpy as np
import os
import soundfile as sf  # Import soundfile for saving audio

# Load the audio file
audio_file = r"C:\Users\sofu0\code\beatbox_to_midi\data\beatbox1.wav"
y, sr = librosa.load(audio_file, sr=None)

# Parameters
frame_length = 1024  # Length of each frame
hop_length = 512     # Hop length

# Number of segments to output
num_segments = 5  # Number of segments to save

# Calculate the number of frames
num_frames = int(np.ceil(len(y) / hop_length))

# Define output folder for audio segments
output_audio_folder = r"C:\Users\sofu0\code\beatbox_to_midi\output_audio"
os.makedirs(output_audio_folder, exist_ok=True)

# Loop to extract and save segments
for i in range(num_segments):
    # Calculate the start sample for each segment
    start_sample = i * hop_length
    end_sample = start_sample + frame_length
    
    # Ensure we don't go out of bounds
    if start_sample >= len(y):
        break

    # Extract the segment
    segment = y[start_sample:end_sample]

    # Save the segment as a new audio file using soundfile
    segment_file_path = os.path.join(output_audio_folder, f'segment_{i + 1}.wav')
    sf.write(segment_file_path, segment, sr)

    print(f"Segment {i + 1} saved to: {segment_file_path}")
