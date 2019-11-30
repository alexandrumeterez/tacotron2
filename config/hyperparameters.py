# Audio
SAMPLE_RATE = 22050  # hz
FFT_FRAME_SIZE = 50.0  # ms
FFT_HOP_SIZE = 12.5  # ms
NUM_MELS = 80  # filters
MIN_FREQ = 125  # hz
MAX_FREQ = 7600  # hz
FLOOR_FREQ = 0.01  # reference freq for power to db conversion
SPECTROGRAM_PAD = 0.0  # change to -80.0? (-80 is the mel value of a window of zeros in the time dim (wave))