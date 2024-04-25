import torch
import torchaudio
from speechbrain.inference.enhancement import SpectralMaskEnhancement

enhance_model = SpectralMaskEnhancement.from_hparams(
    source="speechbrain/metricgan-plus-voicebank",
    savedir="pretrained_models/metricgan-plus-voicebank",
)

# Load and add fake batch dimension
noisy = enhance_model.load_audio(
    "speechbrain/metricgan-plus-voicebank/example.wav"
).unsqueeze(0)

# Add relative length tensor
enhanced = enhance_model.enhance_batch(noisy, lengths=torch.tensor([1.]))

# Saving enhanced signal on disk
torchaudio.save('enhanced.wav', enhanced.cpu(), 16000)
