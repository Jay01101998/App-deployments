import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
plt.rcParams['figure.figsize']=[10,5]
from PIL import Image
import streamlit as st

menu=st.sidebar.radio("Menu",["Exploring the data","Predicting Heart attack risk"])

if menu=="Exploring the data":
    st.title("HEART ATTACK RISK Analysis")
    st.image("download.jpeg")
    st.markdown("Cardiovascular diseases (CVDs) are the number 1 cause of death globally, taking an estimated 17.9 million lives each year, which accounts for 31% of all deaths worldwide.")
    st.markdown("Four out of 5CVD deaths are due to heart attacks and strokes, and one-third of these deaths occur prematurely in people under 70 years of age. ")
    st.markdown("Heart failure is a common event caused by CVDs and this dataset contains 11 features that can be used to predict a possible heart disease.")
    st.markdown("People with cardiovascular disease or who are at high cardiovascular risk (due to the presence of one or more risk factors such as hypertension, diabetes, hyperlipidaemia or already established disease) need early detection and management wherein a machine learning model can be of great help.")
    st.markdown("Reference: https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction")
    
    st.header("Let's check then variables and data present in the dataset")


    df=pd.read_csv(r"C:\Users\HP\Desktop\StreamLit\heart_disease\heart_statlog_cleveland_hungary_final.csv")
    st.write("Contents of the data:",df.head())


    st.write("The shape of the data is:",df.shape)

    st.write("Statistical Description of Data:",df.describe())

    st.write("Distribution of varibles are:")
    st.image("output.png")

if menu=="Predicting Heart attack risk":
    st.title("Predicting the heart attack risk")
    
    st.image('heart_diagnosis.jpeg')

    df=pd.read_csv(r"C:\Users\HP\Desktop\StreamLit\heart_disease\heart_statlog_cleveland_hungary_final.csv")
    
    st.header("Lets predict the heart attack risk")

    y=df['target']
    x=df.drop('target',axis=1)

    from sklearn.ensemble import RandomForestClassifier
    rf=RandomForestClassifier()
    rf.fit(x,y)

    age=st.number_input("Enter the age",min_value=28,max_value=77,step=1)
    gender=st.selectbox("Select the gender",(0,1))
    chest_pain=st.select_slider("Select the extent of chest pain",(1,2,3,4))
    resting_bp=st.number_input("Enter the resting blood pressure value",min_value=0,max_value=200,step=1)
    cholesterol=st.number_input("Enter the value of cholesterol",min_value=0,max_value=603,step=1)
    fasting_bs=st.selectbox("Select the fasting blood sugar",(0,1))
    resting_ecg=st.select_slider("Select the resting ECG type",(0,1,2))
    max_heart_rate=st.number_input("Enter the maximum heart rate",min_value=60,max_value=202,step=1)
    exercise_angina=st.selectbox("Enter the chest pain due to exercise",(0,1))
    old_peak=st.number_input("Enter the old peak value",min_value=-3,max_value=7,step=1)
    slope_st=st.selectbox("Enter the slope category of ST",(0,1,2))
    
    value=[age,gender,chest_pain,resting_bp,cholesterol,fasting_bs,resting_ecg,max_heart_rate,exercise_angina,old_peak,slope_st]


    prediction=rf.predict([value])[0]


    if st.button("Predict the heart attack risk"):
        if prediction==0:
            st.subheader("No RISK")
            st.balloons()
        else:
            st.warning("RISK DETECTED!!")
        









    
    


