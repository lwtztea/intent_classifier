import json
from flask import Flask, request, abort, Response
from svc_utils import preprocess_texts, predict_intent, vectorize_texts

app = Flask(__name__)

with open('lib/data/categories.json') as f:
    categories = json.load(f)


@app.route('/', methods=['GET'])
def start():
    return 'Hello! I am waiting for your query...\n'


@app.route('/predict', methods=['POST'])
def classify():
    if not (request.json and 'query' in request.json):
        abort(Response("Your request should be in JSON format: {'query':[texts]}\n"))
    user_query = request.json['query']
    preprocessed_texts = preprocess_texts(user_query)
    vectorized_texts = vectorize_texts(preprocessed_texts)
    predictions = predict_intent(vectorized_texts)
    predictions = [categories[p] for p in predictions]
    return {'predictions': predictions}


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
