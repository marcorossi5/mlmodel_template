#!/bin/bash

# Jump to "self" folder
cd "$(dirname "$0")"

docker compose -f docker/docker-compose.yml stop mlflow
