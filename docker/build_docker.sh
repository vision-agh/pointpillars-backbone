#!/usr/bin/env bash

docker build --build-arg USER_ID=$UID -t mmdet3d:training -f docker/Dockerfile .
