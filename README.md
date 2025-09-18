# BDD100K_object_training

1. Download the BBD100K object detection dataset:
   https://drive.google.com/file/d/1NgWX5YfEKbloAKX9l8kUVJFpWFlUO8UT/view

2. Update the `DATA_PATH` in the setup_docker.sh file. It should point to the downloaded data path.

3. Run the following command to setup the docker container:  
   ```sh ./setup_docker.sh``` 

4. In any browser, launch `http://localhost:8890/`. It will open jupyter lab which serves as the dashboard for data analysis, model training and evaluation.
   

### Data analysis

Launch `1_data_analysis/data_analysis.ipynb` in a notebook. This notebook loads the data and analyses various aspects of it to generate useful inferences which can be used while preparing the data for training the model.

### Model training

Launch `2_model_training/train.ipynb` in a notebook. This module trains our data on custom lightweight version of YoloV5 model. This can also be done by running the train.py file.

### Evaluation and visualization

Launch `3_evaluation_and_visualisation/eval_and_visualise.ipynb` in a notebook. This loads the saved model and generates visualisations of the model output on the validation data.

