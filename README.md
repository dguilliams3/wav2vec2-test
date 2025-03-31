# 🧠 AI Audio Processing Pipeline

**Fine-tuning Wav2Vec2 on Labeled Audio Data + CI/CD + Dockerized Inference + S3 Container for Checkpoints**

This repository implements a modular, containerized pipeline for fine-tuning Facebook’s [Wav2Vec2](https://huggingface.co/facebook/wav2vec2-base-960h) model on labeled audio data. It leverages streaming data from Hugging Face datasets, integrates AWS S3 for checkpoint management, and uses GitHub Actions for automated CI/CD and nightly training runs. The pipeline is robust, scalable, and portable across environments (GPU/CPU) and sets the stage for future enhancements like ONNX export for optimized inference.

---

## 🚀 Key Features

- **🎙 Fine-tuning Wav2Vec2:**  
  - Uses Hugging Face’s `Wav2Vec2Processor` and `Wav2Vec2ForCTC` to process audio and transcriptions.
  - Streams data from the Mozilla Common Voice dataset for scalable training.

- **🔁 Checkpoint Management with AWS S3:**  
  - Automatically syncs model checkpoints (stored under `models/wav2vec2-finetuned`) with an AWS S3 bucket.
  - Ensures seamless resume capability and persistent backup of training progress.

- **🧱 Modular Entrypoint:**  
  - The `entrypoint.sh` script orchestrates the entire training cycle:
    - Syncs checkpoints from S3.
    - Launches the training pipeline.
    - Uploads updated checkpoints back to S3 post-training.

- **🐳 Containerized Execution:**  
  - A robust `Dockerfile` (and optional `docker-compose.yml`) creates a reproducible runtime environment.
  - Supports both local development and production deployment.

- **⏰ Automated CI/CD & Nightly Runs:**  
  - GitHub Actions workflows automatically build, cache, and push Docker images on every push.
  - A scheduled (and manually triggerable) nightly workflow runs the training container at 2 AM UTC.
  - Remote caching via Docker Hub significantly speeds up builds.

- **🔮 Future Enhancements:**  
  - **ONNX Conversion:** Plans to export the fine-tuned model to ONNX for faster, hardware-agnostic inference.
  - Enhanced logging/monitoring, multi-language support, and a REST API for real-time inference.

---

## 🧬 Project Structure

```
ai-audio-project/
├── pipeline_finetune_wav2vec2.py        # Core training pipeline script
├── entrypoint.sh                        # Entrypoint for syncing checkpoints & launching training
├── config.yaml                          # Configuration file for model, dataset, and training parameters
├── Dockerfile                           # Defines the containerized runtime environment
├── docker-compose.yaml                  # Optional Docker Compose for local development
├── requirements.txt                     # Python dependencies
├── .dockerignore                        # Excludes files from the Docker build context
├── .gitignore                           # Git exclusions for repository
├── .env                                 # (Optional) Local environment variables (not committed)
└── .github/
    └── workflows/
        ├── docker-build-push.yml        # CI workflow: Build & push Docker image on pushes
        └── nightly-run.yml              # CI workflow: Scheduled nightly training run
        
# Additional folders (created locally, not committed)
├── models/
│   └── wav2vec2-finetuned/              # Checkpoints and final models (gitignored)
└── logs/                               # Training logs (gitignored)
```

> **Note:** Although the `models/` and `logs/` folders are not committed (see `.dockerignore`), you can include empty placeholder files (e.g., `.keep`) to maintain the structure if desired.

---

## ⚙️ Installation & Setup

### Prerequisites

- **Docker & Docker Compose:** Install these for building and running containers.
- **AWS Credentials:** Set up an AWS S3 bucket for checkpoint syncing. Configure environment variables (`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_REGION`).
- **GitHub Secrets & Environment:**  
  - Repository secrets: `DOCKER_USERNAME` (set to `dguilliams3`), `DOCKER_PASSWORD`.
  - Create a GitHub Environment (e.g., **AWS-S3-AUDIO-2025-03**) and add your AWS credentials there.

### Local Setup

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/dguilliams3/ai-audio-project.git
   cd ai-audio-project
   ```

2. **Configure the Pipeline:**
   - Edit `config.yaml` to specify model, dataset, hyperparameters, and output directories.
   - Create a `.env` file (for local testing) with:
     ```dotenv
     AWS_ACCESS_KEY_ID=your_access_key_id
     AWS_SECRET_ACCESS_KEY=your_secret_access_key
     AWS_REGION=your_region
     ```

3. **Run the Pipeline:**
   - **Directly via Shell:**
     ```bash
     bash entrypoint.sh
     ```
   - **Using Docker Compose:**
     ```bash
     docker-compose up --build
     ```

4. **Direct Python Execution (Optional):**
   ```bash
   pip install -r requirements.txt
   python pipeline_finetune_wav2vec2.py --config config.yaml
   ```

---

## 🛠 Docker & CI/CD

### Dockerfile

The [Dockerfile](./Dockerfile) installs system dependencies (ffmpeg, curl, unzip), AWS CLI v2, Python dependencies, and sets `entrypoint.sh` as the container’s default command.

### .dockerignore

The [.dockerignore](./.dockerignore) file ensures unnecessary files (e.g., virtual environments, logs, local models) are not sent to Docker during build.

### GitHub Actions Workflows

#### Docker Build & Push Workflow (`.github/workflows/docker-build-push.yml`)
- **Trigger:** On pushes to the `main` branch.
- **Function:** Builds the Docker image (using remote caching) and pushes it to Docker Hub.
- **Key Caching Flags:**
  - `cache-from`: Pulls cached layers from the remote cache image.
  - `cache-to`: Pushes new/updated layers to the remote cache image.

#### Nightly Training Run Workflow (`.github/workflows/nightly-run.yml`)
- **Trigger:** Scheduled at 2 AM UTC (and manually triggerable via `workflow_dispatch`).
- **Function:** Builds the image (with caching), pushes it, and runs the training container with AWS credentials.
- **Environment:** Uses the GitHub Environment **AWS-S3-AUDIO-2025-03** to inject AWS secrets.

*Example snippet from the nightly workflow:*
```yaml
      - name: Build Docker image with caching
        run: |
          docker build \
            --cache-from type=registry,ref=${{ secrets.DOCKER_USERNAME }}/ai-audio-project:cache \
            --cache-to type=registry,ref=${{ secrets.DOCKER_USERNAME }}/ai-audio-project:cache,mode=max \
            -t ${{ secrets.DOCKER_USERNAME }}/ai-audio-project:latest .
```

---

## 🧪 Testing

- **Manual Trigger in GitHub Actions:**  
  Add `workflow_dispatch:` to your YAML files to allow manual runs from the GitHub Actions tab.
  
- **Local Testing:**  
  Run your container with:
  ```bash
  docker run --rm --env-file .env dguilliams3/ai-audio-project:latest
  ```
  or use Docker Compose:
  ```bash
  docker-compose up --build
  ```

---

## 🔮 Future Enhancements

- **ONNX Conversion:**  
  Export the fine-tuned model to [ONNX](https://onnx.ai/) for faster, hardware-agnostic inference using ONNX Runtime.
  
- **Enhanced Monitoring:**  
  Integrate TensorBoard or a service like Weights & Biases for real-time training metrics.
  
- **Multi-Service Deployment:**  
  Expand the architecture to include a REST API for real-time inference, logging services, and a model registry.
  
- **Improved Checkpoint Management:**  
  Automate S3 lifecycle policies to manage and clean up older checkpoints.

---

## 📌 Contact & Showcase

This project demonstrates robust DevOps practices and AI pipeline engineering—from Docker containerization and CI/CD automation to AWS-integrated checkpoint management. It is designed to be production-ready and is part of my professional portfolio.

- **GitHub:** [github.com/dguilliams3](https://github.com/dguilliams3)
- **Docker Hub:** [hub.docker.com/r/dguilliams3/ai-audio-project](https://hub.docker.com/r/dguilliams3/ai-audio-project)
- **Email:** [daniellguilliams3@outlook.com](mailto:daniellguilliams3@outlook.com)

---

## Conclusion

The AI Audio Processing Pipeline is a state-of-the-art solution for fine-tuning speech recognition models. It leverages best practices in containerization, automated CI/CD with remote caching, and cloud-integrated checkpoint management. Future enhancements like ONNX conversion and enhanced monitoring will further optimize the model for real-world deployment.
