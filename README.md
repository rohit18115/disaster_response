# Disaster Response Pipeline Project

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

## Example Images

![Training data overview](https://github.com/rohit18115/disaster_response/blob/main/data/Training_data_overview.png)

![Classification result example](https://github.com/rohit18115/disaster_response/blob/main/data/classification_results.png)

## to-do
- Make a better readme
- Deploy app on heroku
