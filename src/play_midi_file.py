import mido
import pygame
import time

# Initialize Pygame mixer
pygame.mixer.init()

# Load the MIDI file
midi_file_path = r"C:\Users\sofu0\code\beatbox_to_midi\output_audio\beatbox_midi.mid"

# Function to play MIDI file
def play_midi(midi_file_path):
    # Load the MIDI file
    midi_file = mido.MidiFile(midi_file_path)

    # Create a Pygame clock to manage the playback timing
    clock = pygame.time.Clock()

    # Start playback
    for message in midi_file.play():
        # Handle note_on messages
        if message.type == 'note_on':
            print(f"Playing note: {message.note}")
            # Optionally: Play a sound for the MIDI note here

        # Sleep for the duration of the message
        clock.tick(60)  # Adjust the tick rate as necessary
        time.sleep(message.time / 1000)  # Convert ticks to seconds

# Play the MIDI file
play_midi(midi_file_path)
