import re
import pickle

model_filename = 'lib/model/svc_model.sav'
vectorizer_filename = 'lib/model/vectorizer.pickle'
SVC_model = pickle.load(open(model_filename, 'rb'))
vectorizer = pickle.load(open(vectorizer_filename, 'rb'))


def _to_lover(x: str):
    return x.lower()


def _remove_punctuation(x: str):
    return re.sub('[^a-z ]', '', x)


def preprocess_texts(texts):
    return list(map(_remove_punctuation, list(map(_to_lover, texts))))


def vectorize_texts(texts):
    return vectorizer.transform(texts)


def predict_intent(input_ids):
    return SVC_model.predict(input_ids)
