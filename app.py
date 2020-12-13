import flask
import time
from sentiment.src.predict import SentimentPredictor
from ner.src.predict import NERPredictor
from flask import Flask
from flask import request

app = Flask(__name__)
app.config['JSON_SORT_KEYS'] = False

sentiment_predictor = SentimentPredictor()
ner_predictor = NERPredictor()


@app.route("/process")
def process():
    start_time = time.time()
    text = request.args.get("text")

    positive_prediction = sentiment_predictor.predict(text)
    negative_prediction = sentiment_predictor.predict_inverse(text)
    result = ner_predictor.predict(text)

    response = {
        "response": {
            "time_taken": str(time.time() - start_time),
            "positive": str(positive_prediction),
            "negative": str(negative_prediction),
            "text": str(text),
            "tokens": result
        }
    }

    return flask.jsonify(response)


@app.route("/sentiment")
def sentiment_predict():
    start_time = time.time()
    text = request.args.get("text")

    positive_prediction = sentiment_predictor.predict(text)
    negative_prediction = sentiment_predictor.predict_inverse(text)

    response = {
        "response": {
            "time_taken": str(time.time() - start_time),
            "positive": str(positive_prediction),
            "negative": str(negative_prediction),
            "text": str(text),
        }
    }

    return flask.jsonify(response)


@app.route("/ner")
def ner_predict():
    start_time = time.time()
    text = request.args.get("text")

    result = ner_predictor.predict(text)

    response = {
        "response": {
            "time_taken": str(time.time() - start_time),
            "text": str(text),
            "result": result
        }
    }

    return flask.jsonify(response)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port="9999")
