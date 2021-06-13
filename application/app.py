import json
import torch
from flask import Flask, request, abort, Response
from utils import tokenize_text, predict_intent
from model import BertForMultiLabelClassification
from transformers import BertTokenizer

app = Flask(__name__)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = BertForMultiLabelClassification()
model.to(device)

model_path = 'lib/model/classifier.pt'
model.classifier.load_state_dict(torch.load(model_path, map_location=device))

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

with open('lib/data/categories.json') as f:
    categories = json.load(f)


@app.route('/', methods=['GET'])
def start():
    return 'Hello! I am waiting for your query...'


@app.route('/predict', methods=['POST'])
def classify():
    if not (request.json and 'query' in request.json):
        abort(Response("Your request should be in JSON format: {'query':[texts]}"))
    user_query = request.json['query']

    input_ids, mask = tokenize_text(user_query, tokenizer)
    input_ids, mask = input_ids.to(device), mask.to(device)

    predictions = predict_intent(input_ids, mask, model)
    predictions = [categories[p] for p in predictions]

    return {'prediction': predictions}


if __name__ == '__main__':
    app.run()
