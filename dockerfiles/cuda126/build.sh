#!/bin/bash
SCRIPT_DIR=$(cd $(dirname $0); pwd)
USER_NAME=${USER:-$(whoami)}
IMAGENAME="${USER_NAME}_mujaco_gym_lite_cuda126-image"
TAG="latest"
docker build -t ${IMAGENAME}:${TAG} --build-arg USER=$USER --build-arg USER_ID=$UID ${SCRIPT_DIR}
