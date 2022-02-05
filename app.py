import numpy as np
import tensorflow as tf
from flask import Flask, request, make_response
import json
import pickle
from flask_cors import cross_origin
from keras.models import Model
from keras.models import load_model
from nltk.tokenize import word_tokenize
# Keras pre-processing
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
# Text preprocessing and stopwords
from text_preprocess_py import *
import random, re

app = Flask(__name__)
model = load_model("finalized_keras_model.h5")

@app.route('/')
def hello():
    return 'Hello World'

# geting and sending response to dialogflow
@app.route('/webhook', methods=['POST'])
@cross_origin()
def webhook():

    req = request.get_json(silent=True, force=True)

    #print("Request:")
    #print(json.dumps(req, indent=4))

    res = processRequest(req)

    res = json.dumps(res, indent=4)
    #print(res)
    r = make_response(res)
    r.headers['Content-Type'] = 'application/json'
    return r


# processing the request from dialogflow
def processRequest(req):

    #sessionID=req.get('responseId')
    result = req.get("queryResult")
    #user_says=result.get("queryText")
    #log.write_log(sessionID, "User Says: "+user_says)
    parameters = result.get("parameters")
    description=parameters.get("description")
   
    intent = result.get("intent").get('displayName')
    
    if (intent=='Accident_Description'):
        prediction = predict(description)
          
        fulfillmentText= "Predicted Accident Level  {} !".format(prediction)
        #log.write_log(sessionID, "Bot Says: "+fulfillmentText)
        return {
            "fulfillmentText": fulfillmentText
        }
    #else:
    #    log.write_log(sessionID, "Bot Says: " + result.fulfillmentText)

if __name__ == '__main__':
    app.run()

#Predict:
def predict(input_str):
	loaded_model = load_model("finalized_keras_model.h5")
	ip = clean_input(input_str)
	tokenizer_obj = Tokenizer()
	tokenizer_obj.fit_on_texts(ip)
	sequences = tokenizer_obj.texts_to_sequences(ip)
	# pad sequences
	word_index = tokenizer_obj.word_index
	# print('\n Found %s unique tokens.' % len(word_index))

	desc_pad = pad_sequences(sequences, maxlen=100)

	ip_padded = desc_pad
	y_pred = loaded_model.predict(ip_padded, batch_size=32, verbose=1)
	y_pred_bool = np.argmax(y_pred, axis=0)
	prediction = (np.argmax(y_pred_bool))
	return prediction

def clean_input(input_txt):
  input_txt = input_txt.lower()
  input_txt = replace_words(input_txt)
  input_txt = remove_punctuation(input_txt)
  input_txt = lemmatize(input_txt)
  input_txt = re.sub(' +', ' ', input_txt)
  input_txt = remove_stopwords(input_txt)
  return input_txt