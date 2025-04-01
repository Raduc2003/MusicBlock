# Use Ubuntu 20.04 as a base image
FROM ubuntu:20.04

# Disable interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install required dependencies, including python-is-python3 so that the 'python' command is available,
# and python3-numpy.
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    cmake \
    qt5-default \
    libqt5x11extras5-dev \
    libyaml-dev \
    swig \
    python3-dev \
    python-is-python3 \
    python3-numpy \
    pkg-config \
    libeigen3-dev \
    wget \
    && rm -rf /var/lib/apt/lists/*

# ----------------------------
# Build and install Gaia using WAF
# ----------------------------
RUN git clone https://github.com/MTG/gaia.git /opt/gaia && \
    cd /opt/gaia && \
    chmod +x waf && \
    ./waf configure && \
    ./waf -j$(nproc) && \
    ./waf install && \
    ldconfig

# ----------------------------
# Build and install Essentia with Gaia support using WAF
# ----------------------------
RUN git clone https://github.com/MTG/essentia.git /opt/essentia && \
    cd /opt/essentia && \
    ./waf configure --with-python --with-gaia && \
    ./waf -j$(nproc) && \
    ./waf install && \
    ldconfig

# Ensure that /usr/local/lib (where Gaia and Essentia install their libraries) is in the library path
ENV LD_LIBRARY_PATH="/usr/local/lib:${LD_LIBRARY_PATH}"
# Ensure that /usr/local/bin is in the PATH (where the binaries are installed)
ENV PATH="/usr/local/bin:${PATH}"

# Set the working directory where you'll mount your files (audio, profiles, SVM models, etc.)
WORKDIR /workspace

# Start a bash shell by default
CMD ["bash"]
