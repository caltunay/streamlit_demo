import streamlit as st
import numpy as np
import tensorflow as tf
from transformers import TFDistilBertForSequenceClassification
from transformers import DistilBertTokenizer
from pickle import load 
from sklearn.preprocessing import LabelEncoder

def predict_proba(string_list, model, tokenizer):
    
    # tokenize text
    encodings = tokenizer(string_list, truncation = True, padding = True)

    # transform to tf.Dataset
    dataset = tf.data.Dataset.from_tensor_slices((dict(encodings)))

    # predict
    preds = model.predict(dataset.batch(1)).logits

    # transformto array with probabilities
    preds = tf.nn.softmax(preds, axis = 1).numpy()
    
    # convert to label
    results = np.argmax(preds, axis = 1)

    return results


header = st.container()

path = 'model'

loaded_tokenizer = DistilBertTokenizer.from_pretrained(path)
loaded_model = TFDistilBertForSequenceClassification.from_pretrained(path) 
loaded_label_encoder = load(open(path + '/label_encoder.pkl', 'rb'))

with header:
	st.title('Submit your post below!')
	user_input = st.text_area('')
	res = predict_proba(string_list = [user_input], model = loaded_model, tokenizer = loaded_tokenizer)
	worded_res = loaded_label_encoder.inverse_transform(res)
	st.text(worded_res[0])




	# with open('model', 'r') as file:
	# 	content = file.read()

	# 	for idx, line in enumerate(content):
	# 		st.text(idx)
	# 		st.text(line)
