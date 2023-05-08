from flask import Flask, jsonify, request, render_template, send_file, redirect, url_for
import pandas as pd
from bertopic import BERTopic

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html", content="Testing")

topic_model = BERTopic.load("model/topic_model")
data_subset = pd.read_csv("clean_data_subset.csv")
docs = data_subset['text_cleaned'].to_list()
x = topic_model.get_document_info(docs)

@app.route("/predict")
def predict():
    return render_template("layout.html")


@app.route("/predict", methods=["POST"])
def do_prediction():
    topic = request.json["topic"]
    # docs = data_subset['text_cleaned'].to_list()
    # x = topic_model.get_document_info(docs)
    ind_list = x[x.Topic==topic_model.find_topics(topic)[0][0]].sort_values('Probability', ascending=False)[0:5].index
    result = data_subset[['title', 'summary', 'url']].loc[ind_list]
    return result.to_json(orient="records")



if __name__ == "__main__":
    app.run(host='0.0.0.0') #debug=True updates website automatically
