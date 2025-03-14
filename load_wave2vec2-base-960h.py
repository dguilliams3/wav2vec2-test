from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import torch

# Load the processor and the pre-trained model
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")

# Freeze the feature extractor layers and move the model to GPU if available
model.freeze_feature_extractor()
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
