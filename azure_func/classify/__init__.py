import logging
from .predict import predict_sentence_from_url
import azure.functions as func
import json

def main(req: func.HttpRequest) -> func.HttpResponse:
    sentence = req.params.get('sentence')
    target = req.params.get('target')
    results = predict_sentence_from_url(sentence,target)
    headers = {
        "Content-type": "application/json",
        "Access-Control-Allow-Origin": "*"
    }
    return func.HttpResponse(json.dumps(results), headers = headers)
