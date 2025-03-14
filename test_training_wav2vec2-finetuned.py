from transformers import TrainingArguments, Trainer

training_args = TrainingArguments(
    output_dir="./wav2vec2-finetuned",
    per_device_train_batch_size=8,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=3e-5,
    warmup_steps=500,
    logging_dir="./logs",
    logging_steps=10,
    fp16=True  # Use mixed precision for faster training on GPU
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
)

# Start fine-tuning
trainer.train()
