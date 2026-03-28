FROM nvcr.io/nvidia/cuda:12.6.0-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8
ENV TZ="Asia/Tokyo"

# Install essential
RUN apt update -y && \
    apt install -y --no-install-recommends \
    git \
    sudo \
    libopencv-dev \
    curl \
    g++ \
    make \
    git \
    snapd \
    zip \
    unzip \
    wget \
    libssl-dev \
    freeglut3-dev \
    build-essential \
    software-properties-common \
    lsb-core \
    ffmpeg \
    apt-utils \
    joystick \
    jstest-gtk \
    libcanberra-gtk* \
    usbutils

# GPU environment variable
ENV NVIDIA_VISIBLE_DEVICES=${NVIDIA_VISIBLE_DEVICES:-all}
ENV NVIDIA_DRIVER_CAPABILITIES=${NVIDIA_DRIVER_CAPABILITIES:+$NVIDIA_DRIVER_CAPABILITIES,}graphics
ENV NVIDIA_DRIVER_CAPABILITIES=${NVIDIA_DRIVER_CAPABILITIES:+$NVIDIA_DRIVER_CAPABILITIES,}display,compute

# Install python3 and venv
RUN apt update -y && apt install -y --no-install-recommends \
    python3-pip \
    python3-wheel \
    python3-setuptools \
    python3-tk \
    python3-venv \
    python3-dev \
    libffi-dev

# Install other dependencies
RUN apt update -y && apt install -y libglfw3 libgl1 libosmesa6

# Add sudo user
ARG USER
ARG USER_ID
RUN groupadd -g 1000 developer && \
    useradd  -g      developer -G sudo -m -u $USER_ID -s /bin/bash ${USER} && \
    echo "${USER}:${USER}" | chpasswd

RUN echo "Defaults visiblepw"             >> /etc/sudoers
RUN echo "${USER} ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

# Install venv
ENV VENV_PATH="/home/${USER}/venv/venv_mujaco_gym_lite"
WORKDIR ${VENV_PATH}
RUN python3 -m venv ${VENV_PATH}

# Upgrade basics
RUN ${VENV_PATH}/bin/pip install --upgrade pip setuptools wheel

# Install required python packages
RUN mkdir -p /home/${USER}/tmp/deps
COPY ./requirements.txt /home/${USER}/tmp/deps
RUN ${VENV_PATH}/bin/pip install -U -r /home/${USER}/tmp/deps/requirements.txt

# Install torch cuda126
RUN ${VENV_PATH}/bin/pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

RUN echo "alias load_mujaco_gym_lite_venv='source ${VENV_PATH}/bin/activate'" >> /home/${USER}/.bashrc

# Remove apt caches
RUN rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*

WORKDIR /home/$USER

# For streamdeck
RUN apt update -y && apt install -y libxcb-xinerama0 libxcb-cursor0
RUN apt install -y libudev-dev libusb-1.0-0-dev libjpeg-dev zlib1g-dev libhidapi-libusb0 pipx
RUN ${VENV_PATH}/bin/pip install hid pillow
RUN echo 'SUBSYSTEMS=="usb", ATTRS{idVendor}=="0fd9", GROUP="users"' | tee /etc/udev/rules.d/10-streamdeck.rules
RUN ${VENV_PATH}/bin/pip install streamdeck

# Install Mujoco
RUN mkdir -p .mujoco
WORKDIR /home/$USER/.mujoco
ENV MUJOCO_VERSION="3.3.0"
ENV MUJOCO_ARCHIVE="mujoco-${MUJOCO_VERSION}-linux-x86_64.tar.gz"
RUN wget -q "https://github.com/deepmind/mujoco/releases/download/${MUJOCO_VERSION}/${MUJOCO_ARCHIVE}"
RUN wget -q "https://github.com/deepmind/mujoco/releases/download/${MUJOCO_VERSION}/${MUJOCO_ARCHIVE}.sha256"
RUN shasum -c "${MUJOCO_ARCHIVE}.sha256"
RUN tar -xvzf ${MUJOCO_ARCHIVE}

# For using IKFast lib
RUN apt update && apt install -y libopenblas-dev 

# Install Fonts
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    fontconfig \
    fonts-liberation \
    ttf-mscorefonts-installer && \
    fc-cache -f -v && \
    rm -rf /var/lib/apt/lists/*

# chown
RUN chown -R $USER /home/$USER

# Switch user
USER $USER

CMD [ "bash" ]
