import streamlit as st
import pandas as pd
import numpy as np
import pickle

data = pd.read_csv("Bengaluru_House_Data.csv")

model = pickle.load(open('Real-Estate-Price-Prediction/model.pkl', 'rb'))
X = pickle.load(open('Real-Estate-Price-Prediction/clean_csv.pkl', 'rb'))
location = pickle.load(open('Real-Estate-Price-Prediction/location.pkl', 'rb'))


def predict_price(a, b, c, d):
    loc_index = np.where(X.columns == a)[0][0]
    x = np.zeros(len(X.columns))
    x[0] = b
    x[1] = c
    x[2] = d
    if loc_index >= 0:
        x[loc_index] = 1

    result = model.predict([x])[0]
    result = round(result, 4)
    st.header(f'{result} lakhs')


st.title("Real Estate Price Prediction")

loc = st.selectbox('Location', location)

sqft = st.number_input('Area in sqft ')

bath = int(st.number_input('No of Bathroom '))

bhk = int(st.number_input('No of BHK'))

if st.button('Predict'):
    predict_price(loc, sqft, bath, bhk)
