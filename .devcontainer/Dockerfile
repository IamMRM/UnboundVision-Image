FROM pytorchignite/apex

# Install sudo
USER root
RUN apt-get update && \
    apt-get install -y sudo git

WORKDIR /workspace
RUN chmod -R 777 /tmp

COPY . /workspace/

# Copy the requirements.txt and install Python packages
COPY requirements.txt /workspace/
RUN pip install -r requirements.txt
