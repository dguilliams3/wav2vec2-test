FROM python:3.9-slim

# Install system dependencies (ffmpeg + AWS CLI deps)
RUN apt-get update && apt-get install -y \
    ffmpeg \
    curl \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# Install AWS CLI v2
RUN curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip" && \
    unzip awscliv2.zip && \
    ./aws/install && \
    rm -rf awscliv2.zip aws/

# Set workdir
WORKDIR /app

# Copy and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all source code
COPY . .

# Debugging: Validate that the entrypoint script is present and executable
RUN chmod +x entrypoint.sh
RUN ls -la /app

# Set default entrypoint
CMD ["./entrypoint.sh"]
