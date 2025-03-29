#!/bin/bash
set -e

echo "🔁 Syncing checkpoint from S3..."
aws s3 sync s3://ai-audio-checkpoints-dg/checkpoints/ models/wav2vec2-finetuned/ --region $AWS_REGION

echo "🚀 Starting training..."
python pipeline_finetune_wav2vec2.py --config config.yaml

echo "📤 Syncing updated checkpoint back to S3..."
aws s3 sync models/wav2vec2-finetuned/ s3://ai-audio-checkpoints-dg/checkpoints/ --region $AWS_REGION