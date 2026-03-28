#!/usr/bin/env bash
LOCAL_WORKDIR=$(dirname $(dirname $(dirname $(pwd))))
USER_NAME=${USER:-$(whoami)}
IMAGENAME="${USER_NAME}_mujaco_gym_lite_cuda126-image"
CONTAINER_PATH="/home/${USER}/mujaco_gym_lite_dev"
docker run --gpus all -d \
    --name ${USER_NAME}_mujaco_gym_lite_cuda126-container \
    -e DISPLAY=$DISPLAY \
    -e QT_X11_NO_MITSHM="1" \
    -e QT_LOGGING_RULES='*.debug=false;qt.qpa.*=false' \
    -v "/tmp/.X11-unix:/tmp/.X11-unix:rw" \
    -v "/dev:/dev" \
    -v "/mnt:/mnt" \
    --privileged \
    --workdir=${CONTAINER_PATH} \
    --rm \
    -v ${LOCAL_WORKDIR}:${CONTAINER_PATH} \
    --device=/dev/bus/usb \
    ${IMAGENAME} \
    tail -f /dev/null
xhost +local:`docker inspect --format='{{ .Config.Hostname }}' ${USER_NAME}_mujaco_gym_lite_cuda126-container`
docker exec -it ${USER_NAME}_mujaco_gym_lite_cuda126-container /bin/bash -c 'sudo chown -R $(whoami) /dev/bus'
docker exec -it ${USER_NAME}_mujaco_gym_lite_cuda126-container /bin/bash -c 'cd ~/mujaco_gym_lite_dev/mujaco-gym-lite && ${VENV_PATH}/bin/pip install -e .'
docker exec -it ${USER_NAME}_mujaco_gym_lite_cuda126-container bash
