import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
plt.rcParams['figure.figsize']=[10,5]

import streamlit as st


menu=st.sidebar.radio("Menu",["Data Description","Insurance Prediction"])

if menu=="Data Description":
    st.title("Data Description")
    st.header("Medical Insurance Prediction")
    st.image("insurance.jpeg")
    df=pd.read_csv(r'C:\Users\HP\Downloads\expenses.csv')
    st.write("The data contents:")
    st.table(df.head())
    st.write("The Statistical summary of the categorical data is: ")
    st.table(df.describe(include='object'))

    st.write("The Statistical summary of the numerical data is: ")
    st.table(df.describe(include=np.number))
    
    st.header("Data Visualization")

    st.image('output.png')
    
if menu=='Insurance Prediction':
    st.header("Medical Insurance Prediction")

    st.subheader("Let's make a predictive model")

    df=pd.read_csv(r'C:\Users\HP\Downloads\expenses.csv')

    df['smoker']=df['smoker'].replace({'yes':1,'no':0})
    from sklearn.preprocessing import LabelEncoder
    le=LabelEncoder()

    df['sex']=le.fit_transform(df['sex'])
    df['region']=le.fit_transform(df['region'])

    st.table(df.head())

    

    y=df['charges']
    x=df.drop('charges',axis=1)

    from sklearn.ensemble import RandomForestRegressor

    rf=RandomForestRegressor()

    rf_model=rf.fit(x,y)

    age=st.number_input("Age",15,70,20)

    st.write("For Female enter 0 and for Male 1")
    sex=st.number_input("Sex",0,1,1)

    bmi=st.number_input("BMI",15,54,16)

    children=st.number_input("Children",0,3,1)

    smoker=st.number_input("Smoker",0,1,1)

    st.write("For southwest enter 3; for southeast enter 2; for northwest enter 1; for northeast enter 0")
    region=st.number_input("Region",0,3,1)

    value=[age,sex,bmi,children,smoker,region]

    prediction=rf_model.predict([value])[0]


    if st.button("Predict"):
        st.subheader(round(prediction,4))








    
