#!/usr/bin/env python3
import os
import logging
import yaml
import argparse
import torch
import torchaudio

from dataclasses import dataclass
from typing import Any, Dict, List, Union
from datasets import load_dataset
from transformers import (
    Wav2Vec2Processor,
    Wav2Vec2ForCTC,
    TrainingArguments,
    Trainer
)

# ------------------------------------------------------------------------------
# Data collator that pads input_values and labels separately
# ------------------------------------------------------------------------------
@dataclass
class DataCollatorCTCWithPadding:
    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_values": feature["input_values"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            return_tensors="pt",
        )

        with self.processor.as_target_processor():
            labels_batch = self.processor.pad(
                label_features,
                padding=self.padding,
                return_tensors="pt",
            )

        # Replace padding with -100 so that loss is not computed on pad tokens
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        batch["labels"] = labels
        return batch

# ------------------------------------------------------------------------------
# Function to prepare a single example from the streaming dataset
# ------------------------------------------------------------------------------
def prepare_example(example, processor, target_sampling_rate, max_token_length):
    # Extract waveform and sampling rate from the example
    audio_array = example["audio"]["array"]
    original_sampling_rate = example["audio"]["sampling_rate"]

    # Resample if necessary
    if original_sampling_rate != target_sampling_rate:
        resampler = torchaudio.transforms.Resample(
            orig_freq=original_sampling_rate, new_freq=target_sampling_rate
        )
        audio_array = resampler(torch.tensor(audio_array, dtype=torch.float32)).numpy()

    # Process the audio to obtain input values
    input_values = processor(
        audio_array,
        sampling_rate=target_sampling_rate
    ).input_values[0]

    # Tokenize the sentence to obtain labels
    input_ids = processor.tokenizer(
        example["sentence"],
        truncation=True,
        max_length=max_token_length
    ).input_ids

    return {"input_values": input_values, "labels": input_ids}

# ------------------------------------------------------------------------------
# Function to convert streaming dataset to a list of examples
# ------------------------------------------------------------------------------
def stream_to_list_and_map(stream, processor, target_sampling_rate, max_token_length):
    data_list = []
    for example in stream:
        mapped = prepare_example(example, processor, target_sampling_rate, max_token_length)
        data_list.append(mapped)
    return data_list

# ------------------------------------------------------------------------------
# Main training pipeline function
# ------------------------------------------------------------------------------
def main(args):

    # Load configuration from YAML file
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Determine device (GPU if available)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    # Create output and logging directories if they don't exist
    os.makedirs(config["output_dir"], exist_ok=True)
    os.makedirs(config["logging_dir"], exist_ok=True)

    # ------------------------------------------------------------------------------
    # Load dataset using streaming
    # ------------------------------------------------------------------------------
    logger.info("Loading train dataset with streaming...")
    train_stream = load_dataset(
        config["dataset_name"],
        config["dataset_language"],
        split=config["train_split"],
        streaming=True,
        trust_remote_code=True
    ).take(config["train_examples"])

    logger.info("Loading validation dataset with streaming...")
    valid_stream = load_dataset(
        config["dataset_name"],
        config["dataset_language"],
        split=config["validation_split"],
        streaming=True,
        trust_remote_code=True
    ).take(config["validation_examples"])

    # ------------------------------------------------------------------------------
    # Load processor (tokenizer + feature extractor)
    # ------------------------------------------------------------------------------
    logger.info(f"Loading processor from {config['model_name']}...")
    processor = Wav2Vec2Processor.from_pretrained(config["model_name"])

    # ------------------------------------------------------------------------------
    # Convert streaming datasets to lists for Trainer compatibility
    # ------------------------------------------------------------------------------
    logger.info("Converting train stream to list...")
    train_dataset = stream_to_list_and_map(
        train_stream, processor,
        config["target_sampling_rate"],
        config["max_token_length"]
    )

    logger.info("Converting validation stream to list...")
    valid_dataset = stream_to_list_and_map(
        valid_stream, processor,
        config["target_sampling_rate"],
        config["max_token_length"]
    )

    # ------------------------------------------------------------------------------
    # Load model and freeze feature extractor
    # ------------------------------------------------------------------------------
    resume_checkpoint = config.get("resume_checkpoint", "").strip()
    if resume_checkpoint:
        logger.info(f"Resuming model from checkpoint: {resume_checkpoint}")
        model = Wav2Vec2ForCTC.from_pretrained(resume_checkpoint)
    else:
        logger.info(f"Loading model from {config['model_name']}...")
        model = Wav2Vec2ForCTC.from_pretrained(config["model_name"])
    model.freeze_feature_extractor()
    model.to(device)

    # If resuming and running on CPU, remove the fp16 scaler file to prevent errors.
    if resume_checkpoint:
        scaler_file = os.path.join(resume_checkpoint, "scaler.pt")
        if device == "cpu":
            if os.path.exists(scaler_file):
                logger.info(f"Removing fp16 scaler file from checkpoint: {scaler_file}")
                os.remove(scaler_file)
        elif device == "cuda":
            if not os.path.exists(scaler_file):
                logger.warning("No fp16 scaler file found in the checkpoint; training will resume with a new scaler state.")

    # ------------------------------------------------------------------------------
    # Initialize data collator
    # ------------------------------------------------------------------------------
    data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)

    # ------------------------------------------------------------------------------
    # Define training arguments and Trainer
    # ------------------------------------------------------------------------------
    training_args = TrainingArguments(
        output_dir=config["output_dir"],
        per_device_train_batch_size=config["per_device_train_batch_size"],
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=float(config["learning_rate"]),
        warmup_steps=config["warmup_steps"],
        logging_dir=config["logging_dir"],
        logging_steps=config["logging_steps"],
        fp16=(device == "cuda"),
        num_train_epochs=config["num_train_epochs"],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        data_collator=data_collator,
    )

    # ------------------------------------------------------------------------------
    # Run training
    # ------------------------------------------------------------------------------
    if resume_checkpoint:
        logger.info("Resuming training from checkpoint...")
        trainer.train(resume_from_checkpoint=resume_checkpoint)
    else:
        logger.info("Starting training from scratch...")
        trainer.train()
    logger.info("Training completed!")

    # ------------------------------------------------------------------------------
    # Evaluate and save the model
    # ------------------------------------------------------------------------------
    logger.info("Evaluating the model...")
    metrics = trainer.evaluate()
    logger.info(f"Validation metrics: {metrics}")

    final_model_dir = os.path.join(config["output_dir"], "final_model")
    trainer.save_model(final_model_dir)
    logger.info(f"Model saved in: {final_model_dir}")

    # ------------------------------------------------------------------------------
    # Note: To view TensorBoard logs, run: tensorboard --logdir {config['logging_dir']}
    # ------------------------------------------------------------------------------

if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    logger = logging.getLogger(__name__)

    parser_description = "Fine-tune Wav2Vec2 model with streaming Common Voice data"
    logger.info(f"""
             Loading Argument Parser...
             (Description {parser_description})...
             """)
    
    parser = argparse.ArgumentParser(
        description=parser_description
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to the YAML configuration file (default: config.yaml)"
    )
    args = parser.parse_args()
    main(args)