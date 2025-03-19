import os
import torch
from datasets import load_dataset, Audio
from transformers import (
    Wav2Vec2Processor,
    Wav2Vec2ForCTC,
    TrainingArguments,
    Trainer,
    DataCollatorCTCWithPadding,
)

###############################################################################
# 1. SETUP & CONFIG
###############################################################################

# Root directory of this script
root_dir = os.path.dirname(os.path.abspath(__file__))

# We'll store final model files here
model_out_dir = os.path.join(root_dir, "wav2vec2-finetuned")

# Make sure you installed the needed libs in WSL:
#   sudo apt-get update && sudo apt-get install libsndfile1
#   pip install soundfile

# GPU or CPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

###############################################################################
# 2. LOAD A TINY SUBSET OF COMMON VOICE WITH STREAMING
###############################################################################

# We create a *train* and a *validation* subset to properly test the Trainer
# Only 200 train and 50 validation examples, to keep it minimal

print("Loading train dataset with streaming...")
train_stream = load_dataset(
    "mozilla-foundation/common_voice_11_0",
    "en",
    split="train",         # 'train' split
    streaming=True
).take(200)                # Only take 200 examples

print("Loading validation dataset with streaming...")
valid_stream = load_dataset(
    "mozilla-foundation/common_voice_11_0",
    "en",
    split="validation",    # 'validation' split
    streaming=True
).take(50)                 # Only take 50 examples

# We keep the "audio" and "sentence" columns but remove columns like "accent", "gender", etc.
# We won't store to disk, because streaming won't let us do that.
# We'll transform from stream to Python lists next.

###############################################################################
# 3. PROCESSOR FOR AUDIO → INPUT VALUES, TEXT → LABELS
###############################################################################

# Load pretrained Wav2Vec2 processor (tokenizer + feature extractor)
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")

# We'll define a function that maps raw streaming samples to:
#   - input_values (model input)
#   - labels (tokenized text)
import torchaudio

def prepare_example(example):
    # Example has fields: {"audio": {...}, "sentence": "...", ...}
    # 1) Extract waveform & sampling rate
    audio_array = example["audio"]["array"]
    original_sampling_rate = example["audio"]["sampling_rate"]

    # 2) If needed, resample to 16kHz
    target_sampling_rate = 16000
    if original_sampling_rate != target_sampling_rate:
        resampler = torchaudio.transforms.Resample(orig_freq=original_sampling_rate, new_freq=target_sampling_rate)
        audio_array = resampler(torch.tensor(audio_array, dtype=torch.float32)).numpy()

    # 3) Convert waveform to model input
    input_values = processor(
        audio_array, 
        sampling_rate=target_sampling_rate  # Ensure it's 16kHz now
    ).input_values[0]

    # 4) Convert text to tokenized labels
    input_ids = processor.tokenizer(
        example["sentence"], 
        truncation=True, 
        max_length=128
    ).input_ids

    return {
        "input_values": input_values,
        "labels": input_ids
    }


###############################################################################
# 4. MAP STREAMING DATA → LIST (so the Trainer can read it)
###############################################################################

def stream_to_list_and_map(stream):
    """
    1) Convert streaming dataset to Python list
    2) Apply 'prepare_example' to each sample
    3) Return a list of dicts with 'input_values' and 'labels'
    """
    # Convert the streaming dataset to an iterable
    data_list = []
    for example in stream:
        # If "audio" is just a path and not loaded, you might do something else,
        # but typically streaming includes the audio arrays automatically.
        # Map the example to input_values & labels
        mapped = prepare_example(example)
        data_list.append(mapped)
    return data_list

print("Converting train stream to list...")
train_dataset = stream_to_list_and_map(train_stream)

print("Converting validation stream to list...")
valid_dataset = stream_to_list_and_map(valid_stream)

# Each item in train_dataset / valid_dataset is now a dict like:
# {"input_values": [...], "labels": [...]}

###############################################################################
# 5. LOAD THE MODEL & COLLATOR
###############################################################################

model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
model.freeze_feature_extractor()  # Freeze the feature extractor layers
model.to(device)

# The DataCollator will pad input_values and labels to the max length in a batch
data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)

###############################################################################
# 6. TRAINING ARGUMENTS & TRAINER
###############################################################################

training_args = TrainingArguments(
    output_dir=model_out_dir,
    per_device_train_batch_size=2,  # 2 for small GPU usage (8 might be too big for some GPUs)
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=3e-5,
    warmup_steps=500,
    logging_dir=os.path.join(root_dir, "logs"),
    logging_steps=10,
    fp16=(device=="cuda"),  # Only do fp16 if we have a GPU
    num_train_epochs=2,     # Just 2 epochs for demonstration
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
    data_collator=data_collator,
)

###############################################################################
# 7. RUN TRAINING
###############################################################################

print("Starting training...")
trainer.train()
print("Training completed!")

###############################################################################
# 8. (Optional) Evaluate or Save
###############################################################################

# Evaluate
metrics = trainer.evaluate()
print("Validation metrics:", metrics)

# Save final model
trainer.save_model(os.path.join(model_out_dir, "final_model"))
print("Model saved!")
