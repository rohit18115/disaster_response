import sys
import pandas as pd
from sqlalchemy import create_engine
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import xgboost as xgb

stop_words = stopwords.words("english")
lemmatizer = WordNetLemmatizer()
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report
import numpy as np
import pickle


def load_data(database_filepath):
    """Loads database from given filepath
    and merge them in a dataframe.

    Parameters:
        database_filepath: the file path for message data

    Returns:
        dataframe

    """
    engine = create_engine("sqlite:///{}".format(database_filepath))
    df = pd.read_sql_table("disaster_texts", con=engine)
    # Drop any nan rows as it will cause hinderance in training the model
    df.dropna(subset=["related"], inplace=True)
    X = df["message"]
    Y = df.iloc[:, 4:]
    return X, Y


def tokenize(text):
    """NLP pipeline for processing the text and converting it into
    a form which is suitable for training a model.

    Parameters:
        text: non standardized text

    Returns:
        tokenized text

    """
    # normalize case and remove punctuation
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())

    # tokenize text
    tokens = word_tokenize(text)

    # lemmatize andremove stop words
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]

    return tokens


def build_model(fast="fast", XGB="xgb"):
    """Builds a training pipeline and finds optimal hyperparameters for the
    given data.

    Parameters:
        fast(str): If true the model will use the best parameters on which
        the model was trained previously.
        XGB(str): Uses the XGBoost model.

    Returns:
        Model

    """
    if XGB != "xgb":

        pipeline = Pipeline(
            [
                ("vect", CountVectorizer(tokenizer=tokenize)),
                ("tfidf", TfidfTransformer()),
                ("clf", MultiOutputClassifier(RandomForestClassifier())),
            ]
        )
    else:
        model = xgb.XGBClassifier()
        pipeline = Pipeline(
            [
                ("vect", CountVectorizer(tokenizer=tokenize)),
                ("tfidf", TfidfTransformer()),
                ("clf", MultiOutputClassifier(model)),
            ]
        )
    if XGB != "xgb":
        if fast != "fast":
            # Number of trees in random forest
            n_estimators = [int(x) for x in np.linspace(start=50, stop=100, num=2)]
            # Number of features to consider at every split
            max_features = ["auto", "sqrt"]
            # Maximum number of levels in tree
            max_depth = [int(x) for x in np.linspace(10, 30, num=2)]
            max_depth.append(None)
            # Minimum number of samples required to split a node
            min_samples_split = [5, 10]
            # Minimum number of samples required at each leaf node
            min_samples_leaf = [1, 3]
            # Create the random grid
            random_grid = {
                "clf__estimator__n_estimators": n_estimators,
                "clf__estimator__max_features": max_features,
                "clf__estimator__max_depth": max_depth,
                "clf__estimator__min_samples_split": min_samples_split,
                "clf__estimator__min_samples_leaf": min_samples_leaf,
            }
            cv = GridSearchCV(pipeline, param_grid=random_grid)

            return cv
        else:

            random_grid = {
                "clf__estimator__n_estimators": 50,
                "clf__estimator__max_features": "auto",
                "clf__estimator__max_depth": 10,
                "clf__estimator__min_samples_split": 5,
                "clf__estimator__min_samples_leaf": 1,
            }
            pipline.set_params(
                clf__estimator__n_estimators=50,
                clf__estimator__max_features="auto",
                clf__estimator__max_depth=10,
                clf__estimator__min_samples_split=5,
                clf__estimator__min_samples_leaf=1,
            )

            return pipeline
    else:
        if fast != "fast":
            random_grid = {
                "clf__estimator__max_depth": [5, 7, 10],
                "clf__estimator__n_estimators": [50, 100],
            }
            cv = GridSearchCV(pipeline, param_grid=random_grid)
            return cv
        else:
            # random_grid = {
            #     "clf__estimator__max_depth": 5,
            #     "clf__estimator__n_estimators": 50,
            # }
            pipeline.set_params(
                clf__estimator__max_depth=5, clf__estimator__n_estimators=50
            )

            return pipeline


def evaluate_model(model, X_test, Y_test):
    """Tests the model and prints the  accuracy, precision, and recall
    for each category in the label.

    Parameters:
        model: trained model
        X_test: Testing data
        Y_test: Testing labels
        category_names: Name of the categories in the labels

    Returns:
        None

    """
    predictions = model.predict(X_test)
    for i, col in enumerate(Y_test):
        print(col, classification_report(Y_test[col], predictions[:, i]))


def save_model(model, model_filepath):
    """
    Saves the model at the specified path
    """
    pickle.dump(model, open(model_filepath, "wb"))


def main():
    if len(sys.argv) == 5:
        database_filepath, model_filepath, fast, XGB = sys.argv[1:]
        print("Fast and XGB", fast, XGB)
        print("Loading data...\n    DATABASE: {}".format(database_filepath))
        X, Y = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        print("Building model...")
        model = build_model(fast, XGB)

        print("Training model...")
        model.fit(X_train, Y_train)

        print("Evaluating model...")
        evaluate_model(model, X_test, Y_test)

        print("Saving model...\n    MODEL: {}".format(model_filepath))
        save_model(model, model_filepath)

        print("Trained model saved!")

    else:
        print(
            "Please provide the filepath of the disaster messages database "
            "as the first argument and the filepath of the pickle file to "
            "save the model to as the second argument. \n\nExample: python "
            "train_classifier.py ../data/DisasterResponse.db classifier.pkl"
        )


if __name__ == "__main__":
    main()
