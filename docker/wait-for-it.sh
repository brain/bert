#!/bin/bash
# Note: this is a generated file, please modify it in github.com/brain/docker/templates directory.

# usage
# ./wait-for-it.sh <container-name>

for CONTAINER in "$@"
do
  echo ">> Waiting ... $CONTAINER"
  WAIT=0
  docker inspect --format "{{json .State.Health }}" ${CONTAINER}|grep -q healthy
  echo $?
  while ! docker inspect --format "{{json .State.Health }}" ${CONTAINER}|grep -q healthy ; do
    sleep 1
    WAIT=$(($WAIT + 1))
    if [ "$WAIT" -gt 60 ]; then
      echo "Error: Timeout"
      exit 1
    fi
  done
done
