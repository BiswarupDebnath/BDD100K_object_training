# BDD100K_object_training

1. Download the BBD100K object detection dataset:
   https://drive.google.com/file/d/1NgWX5YfEKbloAKX9l8kUVJFpWFlUO8UT/view

2. Update the `DATA_PATH` in the setup_docker.sh file. It should point to the downloaded data path

3. Run the following command to setup the docker container  
   ```sh ./setup_docker.sh``` 

4. In any browser, launch `http://localhost:8890/`. It will open jupyter lab which serves as the dashboard for data analysis, model training and evaluation.
   

