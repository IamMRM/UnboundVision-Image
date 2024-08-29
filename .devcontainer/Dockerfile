# .devcontainer/Dockerfile
FROM determinedai/environments:cuda-11.8-pytorch-2.0-gpu-mpi-0.31.1

# Install sudo
USER root
RUN apt-get update && \
    apt-get install -y sudo git

# Optional: Create a non-root user with sudo privileges
RUN useradd -ms /bin/bash devuser && \
    echo "devuser:devpassword" | chpasswd && \
    adduser devuser sudo && \
    adduser devuser video

# Switch to the non-root user
USER devuser

WORKDIR /workspace

# Copy the requirements.txt and install Python packages
COPY requirements.txt /workspace/
RUN pip install --upgrade pip && pip install -r requirements.txt

# Install NVIDIA Apex
RUN git clone https://github.com/NVIDIA/apex && \
    cd apex && \
    pip install -v --disable-pip-version-check --no-build-isolation --no-cache-dir ./

# Optional: Clean up to reduce image size
RUN rm -rf /workspace/apex
