# Disaster Response Pipeline Project
## Introduction:

This is a disaster response project. THe main funciton is to create a webapp that uses data cleaning and machine learning techniques to classify 
disaster related tweets and classify them on the basis of the disaster they pertain to. It can have many utilities such as forwarding the tweet to 
appropriate disaster response centers and increase the reach of the tweet. Along with that the webapp also gives the overview of the data on which the model is trained.

## Repository structure:
- app
    - | - template
    - |     - | - master.html # main page of web app
    - |     - | - go.html # classification result page of web app
    - | - run.py # Flask file that runs app
- data
    - | - disaster_categories.csv # data to process
    - | - disaster_messages.csv # data to process
    - | - process_data.py
    - | - InsertDatabaseName.db # database to save clean data to
- models
    - | - train_classifier.py
    - | - classifier.pkl # saved model
- README.md
- ETL Pipeline Preparation.ipynb # jupyter notebook for designing and testing the ELT pipeline
- ML Pipeline Preparation.ipynb # jupyter notebook for designing and training the ML models


### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/disaster_texts.db 
`
    - Two types are models are present for training:
        - Random Forest without gridsearch:
        ` python models/train_classifier.py data/disaster_texts.db models/gridsearch_model_xgb.pkl fast rand`
        - Random Forest with gridsearch:
        ` python models/train_classifier.py data/disaster_texts.db models/gridsearch_model_xgb.pkl grid rand`
        - XGBoost without gridsearch:
        ` python models/train_classifier.py data/disaster_texts.db models/gridsearch_model_xgb.pkl fast xgb`
        - XGBoost with gridsearch:
        ` python models/train_classifier.py data/disaster_texts.db models/gridsearch_model_xgb.pkl grid xgb`
        

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

## Example Images

![Training data overview](https://github.com/rohit18115/disaster_response/blob/main/data/Training_data_overview.png)

![Classification result example](https://github.com/rohit18115/disaster_response/blob/main/data/classification_results.png)

## to-do
- Deploy app on heroku
