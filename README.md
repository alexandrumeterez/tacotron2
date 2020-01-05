# Natural TTS Synthesis by Conditioning WaveNet on Mel Spectrogram Predictions
## Description
An implementation of the Tacotron 2 model from the Natural TTS Synthesis by Conditioning WaveNet on Mel Spectrogram Predictions paper.
It contains some modifications to the original paper such as:
- Removed stop token prediction
- Replaced the WaveNet vocoder with the Griffin Lim algorithm used in the 
original Tacotron model

## Other
The repo contains a PDF written in Romanian. This PDF is the description of the project 
since this was used in a class homework. Also, because I lacked the necessary resources
to train the model, the demo uses the model provided by NVIDIA and my own implementation
of the Griffin-Lim algorithm.

### Requierments
- PyTorch
- Librosa
- Pandas 

#### Credits
- https://github.com/A-Jacobson/tacotron2
- https://github.com/NVIDIA/tacotron2
