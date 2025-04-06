FROM ubuntu:18.04

# Set environment variables
ENV ESSENTIA_SVM_MODELS_VERSION 2.1_beta5
ENV ESSENTIA_VERSION 4237da97237397caf837dc79020f34af8dfc35b2
ENV GAIA_VERSION 2.4.6

ENV PYTHONPATH /usr/local/lib/python3/dist-packages
ENV LANG C.UTF-8
ENV TERM=xterm

# # Copy required directories
# COPY ./essentia /essentia

# Install common dependencies and utils
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    curl

# Install Gaia dependencies
RUN apt-get install -y --no-install-recommends \
    libeigen3-dev \
    libqt4-dev \
    libyaml-dev \
    pkg-config \
    python-dev \
    swig

# Compile and install Gaia
RUN curl -L# https://github.com/MTG/gaia/archive/v${GAIA_VERSION}.tar.gz | tar xz -C /tmp && \
    cd /tmp/gaia-${GAIA_VERSION} && \
    ./waf configure --with-python-bindings && \
    ./waf && \
    ./waf install

# Install Essentia dependencies
RUN apt-get install -y --no-install-recommends \
    libavcodec-dev \
    libavcodec57 \
    libavformat-dev \
    libavformat57 \
    libavresample-dev \
    libavresample3 \
    libavutil55 \
    libchromaprint-dev \
    libfftw3-3 \
    libfftw3-dev \
    libsamplerate0 \
    libsamplerate0-dev \
    libtag1-dev \
    libtag1v5 \
    libyaml-0-2 \
    libyaml-dev \
    python3-dev \
    python3-numpy \
    python3-six \
    python3-yaml

RUN cd /usr/include && \
    ln -sf eigen3/Eigen Eigen && \
    ln -sf eigen3/unsupported unsupported

# Compile and install Essentia
RUN curl -L# https://github.com/MTG/essentia/archive/${ESSENTIA_VERSION}.tar.gz | tar xz -C /tmp && \
    cd /tmp/essentia-${ESSENTIA_VERSION} && \
    ./waf configure --mode=release --with-gaia --with-example=streaming_extractor_music && \
    ./waf && \
    cp ./build/src/examples/essentia_streaming_extractor_music /usr/local/bin && \
    cp ./build/src/libessentia.so /usr/local/lib

RUN ldconfig /usr/local/laib

# Download SVM models
RUN curl -L# https://essentia.upf.edu/svm_models/essentia-extractor-svm_models-v${ESSENTIA_SVM_MODELS_VERSION}.tar.gz | tar xz -C /tmp && \
    mv /tmp/essentia-extractor-svm_models-v${ESSENTIA_SVM_MODELS_VERSION}/* /essentia/svm_models/

# Clean up
RUN apt-get autoremove -y && \
    apt-get clean && \
    rm -rf /tmp/*

# Install additional Python packages using pip
RUN apt-get update && \
    apt-get install -y python3-pip ffmpeg && \
    pip3 install qdrant-client colorama librosa soundfile pyloudnorm numpy

# Set working directory
WORKDIR /essentia
