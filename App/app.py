# Import packages
import streamlit as st
import altair as at
import plotly.express as px
import pandas as pd
import numpy as np
import joblib

# Load the model
pipeline_lr = joblib.load(open("../models/emotion_classifier.pkl", "rb"))
pipeline_rf = joblib.load(open("../models/emotion_classifier_rf.pkl", "rb"))
pipeline_xgboost = joblib.load(open("../models/emotion_classifier_xgboost.pkl", "rb"))

# Functions for Logistic Regression
def predict_emotions_lr(text):
    results = pipeline_lr.predict([text])
    return results[0]

def predict_probability_lr(text):
    results = pipeline_lr.predict_proba([text])
    return results

# Functions for Random Forest
def predict_emotions_rf(text):
    results = pipeline_rf.predict([text])
    return results[0]

def predict_probability_rf(text):
    results = pipeline_rf.predict_proba([text])
    return results

# Functions for XGBoost
def predict_emotions_xgboost(text):
    results = pipeline_xgboost.predict([text])
    return results[0]

def predict_probability_xgboost(text):
    results = pipeline_xgboost.predict_proba([text])
    return results

st.title("Know Your Emotions")
menu = ["Logistic Regression", "Random Forest", "XGBoost"]
choice = st.sidebar.selectbox("Selct Classifier from the Menu", menu)

# Logistic Regression
if choice=="Logistic Regression":
    st.subheader("Logistic Regression - Emotion in Text")
    with st.form(key="emotion_clf_form"):
        raw_text = st.text_area("Type Here")
        submit_text = st.form_submit_button(label="Submit")
        
    if submit_text:
        col1, col2 = st.beta_columns(2)
        
        prediction = predict_emotions_lr(raw_text)
        probability = predict_probability_lr(raw_text)
        
        with col1:
            st.success("Original Text")
            st.write(raw_text)
            
            st.success("Prediction")
            st.write(prediction)
            st.write("Confidence: ", np.max(probability))
            
        with col2:
            st.success("Prediction Probability")
            prob_df = pd.DataFrame(probability, columns=pipeline_lr.classes_)
            prob_df_clean = prob_df.T.reset_index()
            prob_df_clean.columns = ['emotions','probability']
            fig = at.Chart(prob_df_clean).mark_bar().encode(x="emotions", y="probability", color="emotions")
            st.altair_chart(fig, use_container_width=True)
            
# Random Forest
elif choice=="Random Forest":
    st.subheader("Random Forest - Emotion in Text")
    with st.form(key="emotion_clf_form"):
        raw_text = st.text_area("Type Here")
        submit_text = st.form_submit_button(label="Submit")
        
    if submit_text:
        col1, col2 = st.beta_columns(2)
        
        prediction = predict_emotions_rf(raw_text)
        probability = predict_probability_rf(raw_text)
        
        with col1:
            st.success("Original Text")
            st.write(raw_text)
            
            st.success("Prediction")
            st.write(prediction)
            st.write("Confidence: ", np.max(probability))
            
        with col2:
            st.success("Prediction Probability")
            prob_df = pd.DataFrame(probability, columns=pipeline_rf.classes_)
            prob_df_clean = prob_df.T.reset_index()
            prob_df_clean.columns = ['emotions','probability']
            fig = at.Chart(prob_df_clean).mark_bar().encode(x="emotions", y="probability", color="emotions")
            st.altair_chart(fig, use_container_width=True)

# XGBoost
elif choice=="XGBoost":
    st.subheader("XGBoost - Emotion in Text")
    with st.form(key="emotion_clf_form"):
        raw_text = st.text_area("Type Here")
        submit_text = st.form_submit_button(label="Submit")
        
    if submit_text:
        col1, col2 = st.beta_columns(2)
        
        prediction = predict_emotions_xgboost(raw_text)
        probability = predict_probability_xgboost(raw_text)
        
        with col1:
            st.success("Original Text")
            st.write(raw_text)
            
            st.success("Prediction")
            st.write(prediction)
            st.write("Confidence: ", np.max(probability))
            
        with col2:
            st.success("Prediction Probability")
            prob_df = pd.DataFrame(probability, columns=pipeline_xgboost.classes_)
            prob_df_clean = prob_df.T.reset_index()
            prob_df_clean.columns = ['emotions','probability']
            fig = at.Chart(prob_df_clean).mark_bar().encode(x="emotions", y="probability", color="emotions")
            st.altair_chart(fig, use_container_width=True)
