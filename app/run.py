import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar, Heatmap
from sklearn.externals import joblib
from sqlalchemy import create_engine


app = Flask(__name__)


def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def get_redundant_pairs(df):
    """Get diagonal and lower triangular pairs of correlation matrix"""
    pairs_to_drop = set()
    cols = df.columns
    for i in range(0, df.shape[1]):
        for j in range(0, i + 1):
            pairs_to_drop.add((cols[i], cols[j]))
    return pairs_to_drop


def get_top_abs_correlations(df, n=5):
    au_corr = df.corr().abs().unstack()
    labels_to_drop = get_redundant_pairs(df)
    au_corr = au_corr.drop(labels=labels_to_drop).sort_values(ascending=False)
    return au_corr[0:n]


# load data
engine = create_engine("sqlite:///../data/disaster_texts.db")
df = pd.read_sql_table("disaster_texts", engine)

categories_engine = create_engine("sqlite:///../data/categories_database.db")
categories = pd.read_sql_table("categories", categories_engine)

# load model
model = joblib.load("../models/gridsearch_model_xgb.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route("/")
@app.route("/index")
def index():

    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby("genre").count()["message"]
    genre_names = list(genre_counts.index)

    categories_temp = pd.DataFrame(categories.mean() * 100).reset_index()
    categories_temp.columns = ["category", "mean_percent"]

    top_correlations = get_top_abs_correlations(categories, 10).reset_index()
    top_correlations["pairs"] = (
        top_correlations["level_0"] + "-" + top_correlations["level_1"]
    )
    top_correlations = top_correlations.drop(["level_0", "level_1"], axis=1)
    top_correlations.columns = ["correlation", "pairs"]

    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        {
            "data": [Bar(x=genre_names, y=genre_counts)],
            "layout": {
                "title": "Distribution of Message Genres",
                "yaxis": {"title": "Count"},
                "xaxis": {"title": "Genre"},
            },
        },
        {
            "data": [Bar(x=categories_temp.category, y=categories_temp.mean_percent)],
            "layout": {
                "title": "Distribution of nultioutput category labels",
                "yaxis": {"title": "Mean Percent"},
                "xaxis": {"title": "Category"},
            },
        },
        {
            "data": [
                Heatmap(z=categories.corr(), x=categories.columns, y=categories.columns)
            ],
            "layout": {
                "title": "Correlation heatmap of the category labels",
            },
        },
        {
            "data": [Bar(x=top_correlations.pairs, y=top_correlations.correlation)],
            "layout": {
                "title": "Correlation values of the top 10 pairs of category labels",
                "yaxis": {"title": "Correlation values"},
                "xaxis": {"title": "Category label pairs"},
            },
        },
    ]

    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    # render web page with plotly graphs
    return render_template("master.html", ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route("/go")
def go():
    # save user input in query
    query = request.args.get("query", "")

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file.
    return render_template(
        "go.html", query=query, classification_result=classification_results
    )


def main():
    app.run(host="0.0.0.0", port=3001, debug=True)


if __name__ == "__main__":
    main()
