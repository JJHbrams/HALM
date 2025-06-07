import os
import simpleaudio as sa
from bark import SAMPLE_RATE, generate_audio, preload_models

os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"

# download and load all models
preload_models()

# generate audio from text
text_prompt = """
     Hello, my name is Suno. And, uh — and I like pizza. [laughs] 
     But I also have other interests such as playing tic tac toe.
"""
audio_array = generate_audio(text_prompt)


num_channels = 1
bytes_per_sample = 4  # float32
sample_rate = SAMPLE_RATE
play_obj = sa.play_buffer(audio_array, num_channels, bytes_per_sample, SAMPLE_RATE)

# Wait for playback to finish before exiting
play_obj.wait_done()
