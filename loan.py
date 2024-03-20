import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
plt.rcParams['figure.figsize']=[10,5]

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
from sklearn.preprocessing import MinMaxScaler
mmax=MinMaxScaler()


menu=st.sidebar.radio("Menu",["Home","Loan Prediction"])


if menu=='Home':
   st.title("Home")
 
   st.image("loan.jpeg")

   df=pd.read_csv(r"C:\Users\HP\Downloads\loan_data.csv")
   
   st.write("Lets see the contents of the dataset",df.head(15))

   st.write("The shape of the dataset is",df.shape)

   st.header("Data Visualization")


   graph=st.selectbox("SELECT THE PLOT",["KDE Plot","Countplot"])
   if graph=='Countplot':
    #st.pyplot(sns.countplot(x=df['Property_Area']))
      fig,ax=plt.subplots(figsize=(10,5))
      sns.countplot(x=df['Property_Area'])
      st.pyplot(fig)
      sns.countplot(x=df['Loan_Status'])
      st.pyplot(fig)
      sns.countplot(x=df['Gender'])
      st.pyplot(fig)
      sns.countplot(x=df['Married'])
      st.pyplot(fig)
      sns.countplot(x=df['Dependents'])
      st.pyplot(fig)
      sns.countplot(x=df['Education'])
      st.pyplot(fig)
      sns.countplot(x=df['Self_Employed'])
      st.pyplot(fig)
   if graph=='KDE Plot':
      fig,ax=plt.subplots(figsize=(10,5))
      sns.kdeplot(x=df['CoapplicantIncome'])
      st.pyplot(fig)
      sns.kdeplot(x=df['ApplicantIncome'])
      st.pyplot(fig)
      sns.kdeplot(x=df['LoanAmount'])
      st.pyplot(fig)
      sns.kdeplot(x=df['Loan_Amount_Term'])
      st.pyplot(fig)
      sns.countplot(x=df['Dependents'])
      st.pyplot(fig)
      sns.kdeplot(x=df['Credit_History'])
      st.pyplot(fig)
    
if menu=='Loan Prediction':
    st.title("The Loan will be granted or not?")

    df=pd.read_csv(r"C:\Users\HP\Downloads\loan_data.csv")

    df=df.drop('Loan_ID',axis=1)
    df=df.dropna()

    from sklearn.preprocessing import LabelEncoder
    le=LabelEncoder()
    
    df['Gender']=le.fit_transform(df['Gender'])
    df['Dependents']=le.fit_transform(df['Dependents'])
    df['Property_Area']=le.fit_transform(df['Property_Area'])
    df['Married']=le.fit_transform(df['Married'])
    df['Education']=le.fit_transform(df['Education'])
    df['Self_Employed']=le.fit_transform(df['Self_Employed'])

    cols=['ApplicantIncome','CoapplicantIncome','LoanAmount','Loan_Amount_Term']
    
    from sklearn.preprocessing import MinMaxScaler
    mmax=MinMaxScaler()

    df[cols]=mmax.fit_transform(df[cols])
    
    from sklearn.ensemble import RandomForestClassifier
    rf=RandomForestClassifier()

    y=df['Loan_Status']

    y=y.replace({'Y':1,'N':0})

    x=df.drop(['Loan_Status'],axis=1)

    rf_model=rf.fit(x,y)

    gender=st.number_input("Gender",0,1,1)
    married=st.number_input("Married",0,1,1) 
    dependent=st.number_input("Dependents",0,4,1)
    education=st.number_input("Education",0,1,1)
    self_employed=st.number_input("Self Employed",0,1,1) 
    applicant_income=st.number_input("Applicant Income",0,1000,1)
    coapplicant_income=st.number_input("Co-applicant Income",0,1000,1)
    loan_amount=st.number_input("Loan Amount",0,100,1)
    loan_term=st.number_input("Loan Term",0,4,1) 
    credit_history=st.number_input("Credit_History",0,1,1) 
    property_area=st.number_input("Property_area",0,1,1)

    value=[gender,married,dependent,education,self_employed,applicant_income,coapplicant_income,loan_amount,loan_term,credit_history,property_area]
    prediction=rf_model.predict([value])[0]
    if st.button("Loan Status"):
       if prediction==0:
          st.subheader("Rejected")
       else:
          st.subheader("Approved")
       
       

    
    

