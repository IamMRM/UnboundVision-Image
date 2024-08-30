# .devcontainer/Dockerfile
#FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-devel
#FROM pytorchignite/apex:2.4.0-0.5.1
FROM pytorchignite/apex

# Install sudo
USER root
RUN apt-get update && \
    apt-get install -y sudo git

WORKDIR /workspace

# Optional: Create a non-root user with sudo privileges
RUN useradd -ms /bin/bash devuser && \
    echo "devuser:devpassword" | chpasswd && \
    adduser devuser sudo && \
    adduser devuser video

# Install NVIDIA Apex
#RUN git clone https://github.com/NVIDIA/apex && \
#    cd apex && \
#    pip install -v --disable-pip-version-check --no-build-isolation --no-cache-dir ./

# Switch to the non-root user
USER devuser


# Copy the requirements.txt and install Python packages
COPY requirements.txt /workspace/
RUN pip install -r requirements.txt
