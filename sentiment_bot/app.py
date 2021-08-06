from flask import Flask, render_template, request

import requests
import json
from requests.api import post

url = 'https://sentianalysis.azurewebsites.net/api/classify?code=MTPS8hzUaKL/yA7KGCkxTtGqFomtJyh5CiHIeyzrYueOnV81lTroJw=='

app = Flask(__name__)

@app.route('/')
def hello():
    return render_template('index.html')

@app.route('/classify/')
def classify():
    sentence = request.args.get('sentence')
    target= int(request.args.get('target'))
    result = getSentiment(sentence, target)
    return result


def getSentiment(sentence, target):
    params =  {'sentence': sentence, 'target': target}
    return requests.post(url,params=params).text

