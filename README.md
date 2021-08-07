The project involed building a machine learning model for a target based sentiment analysis.
The sentiment analysis can be processed in web application with a chat box interface.
The model is delpoyed to azure function. 
And the chat box web application is delpoyed to web appication in azure. Here is the website.
https://sentbot.azurewebsites.net/

The web application will collect the user's text and target word. Then the send these data to the machine learning model though an API azure function.
After ananlysing the result will send back to the web accplication.

The system able to capture the level of different sentiment.
![alt text](https://i.ibb.co/LY86QV8/Screen-Shot-2021-08-07-at-11-49-35-pm.png)

It is also able to capture sentiment of ambiguous meaning
![alt text](https://i.ibb.co/PDwxwxw/Screen-Shot-2021-08-07-at-11-52-45-pm.png)

I have only tested the web app in google chrome. It might take some time to establish connection in the first time running the application.

The model is builded from tensorflow library.
The training of the model is done in the train.py script.

To run,
Due to the upload size limmit, I was not able to upload the data files.
Download the following files and unzip and put the data in the /data
https://nlp.stanford.edu/data/glove.twitter.27B.zip
```
python3 -m venv .venv 
source .venv/bin/activiate
pip install -r requirements.txt
python -m spacy download en_core_web_sm 
python train.py
```

The implementation of the model is docuemented in the sentimentAnalysisLSTM.pdf

The trained model is saved in models/
The predict.py in models is used to test the model

The graphs of loss and accuracy each models located in result/

The azure_func/ contain the script to delpy the model to azure function.

The sentiment_bot folder contain the the web application built in flask and python3.
To run locally,
```
cd setniment_bot
python3 -m venv .venv 
source .venv/bin/activiate
pip install -r requirements.txt
flask run
```
The website usually host in http://127.0.0.1:5000/

The data file contains the training data and testing data.
