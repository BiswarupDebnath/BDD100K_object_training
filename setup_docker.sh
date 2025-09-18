#!/bin/bash

DATA_PATH="/Users/biswarupdebnath/Desktop/laptop_backup/laptop_Documents/personal_documents/job_prep/bosch/assignment_data_bdd/"
CODE_PATH="/Users/biswarupdebnath/Desktop/laptop_backup/laptop_Documents/personal_documents/job_prep/bosch/assignment"

# Docker build
docker build -t datascience_torch_jupyter -f ./docker/Dockerfile .


# Docker run
docker run --name bosch_assignment_bdd -d -p 8890:8888 -v ${DATA_PATH}:/home/jovyan/assignment_data_bdd/ -v ${CODE_PATH}:/home/jovyan/code/  datascience_torch_jupyter jupyter lab --NotebookApp.token=''
