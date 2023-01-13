#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 12 21:44:49 2023

@author: hongjiang
"""


import streamlit as st
import pickle
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt


#load the model

model = pickle.load(open("model.pkl","rb"))

st.title("Customer Churn Perdiction")
st.subheader('E-Commerce')

#Tenure
Tenure = st.slider('Tenure', 0,70, 5)

#CashbackAmount
CashbackAmount = st.number_input(label='Cash back Amount per month', value=200.00)

#Complain
Complain = st.selectbox('Complain', ['Yes','No'])

#DaySinceLastOrder
DaySinceLastOrder =  st.slider('Day Since Last Order in Last Month', 0, 31, 10)

#WarehouseToHome
WarehouseToHome = st.slider('Ware house To Home (miles)', 0, 100, 30)

#NumberOfAddress
NumberOfAddress = st.slider('Number Of Address', 0, 20, 1)

#SatisfactionScore
SatisfactionScore = st.selectbox('SatisfactionScore', ['Very Satisfied','Satisfied', 'Neutral', 'Disappointed', 'Very Disappointed'])

#OrderAmountHikeFromlastYear
OrderAmountHikeFromlastYear =  st.slider('Order Amount Hike From last Year in percentage', 0, 30, 10)

#OrderCount
OrderCount = st.slider('Order Count', 0, 30, 10)

#NumberOfDeviceRegistered
NumberOfDeviceRegistered = st.slider('Number Of Device Registered', 0, 30, 3)


if st.button('Predict Churn'):
    #complain
    if Complain == "Yes":
        complain = 1
    else:
        complain = 0
        
    #SatisfactionScore   
    if SatisfactionScore == 'Very Satisfied':
        score = 1

    elif SatisfactionScore == "Satisfied":
        score = 2

    elif SatisfactionScore == "Neutral":
        score = 3
        
    elif SatisfactionScore == 'Disappointed':
        score = 4
        
    else:
        score = 5
        
    query = np.array([Tenure, CashbackAmount, complain, DaySinceLastOrder, \
                                      WarehouseToHome, NumberOfAddress, score, \
                                      OrderAmountHikeFromlastYear, OrderCount, NumberOfDeviceRegistered], dtype=object)
    

    query = query.reshape(1, 10)
    print(query)
    prediction = round(model.predict_proba(query)[0][-1], 3)
    
    st.subheader("The churn probability of this customer next month is " + prediction)

    shap.initjs()

    #set the tree explainer as the model of the pipeline
    explainer = shap.TreeExplainer(model)

    #get Shap values from preprocessed data
    shap_values = explainer.shap_values(query)

    #plot the feature importance
    fig = shap.force_plot(explainer.expected_value, shap_values, query, matplotlib=True,show=False, \
                          feature_names=['Tenure', 'CashbackAmount', 'Complain', 'DaySinceLastOrder', \
                                      'WarehouseToHome', 'NumberOfAddress', 'SatisfactionScore', \
                                      'OrderAmountHikeFromlastYear', 'OrderCount', 'NumberOfDeviceRegistered'])
    st.pyplot(fig)



