#!/usr/bin/env bash

docker run --gpus all -it --shm-size=8g \
  --volume="/home/konradl/doktorat/mmdet3d_shared:/home/appuser/shared" \
  --name=mmdet3d-training-instance mmdet3d:training
