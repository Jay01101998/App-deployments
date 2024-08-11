import streamlit as st

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
#import seaborn as sns

import datetime as dt
import scipy.stats as stats
st.set_page_config(page_title="Credit Data",layout='wide')

import plotly.figure_factory as ff
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,classification_report,silhouette_score
import time
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

df=pd.read_csv(r"https://github.com/Jay01101998/App-deployments/blob/main/credit_train.csv")
#df=pd.read_csv(r"C:\Users\HP\Desktop\self-projects\credit_train.csv")

st.sidebar.title("Navigating the App")
options= st.sidebar.radio("INDEX",["Data Study","Modelling and Prediction"])



if options=="Data Study":
    st.title("LOAN CREDIT STATUS PREDICTION")

    st.write(''' 
A loan is a financial arrangement where a lender provides money, property, or other assets to a borrower, who agrees to repay the principal amount along with interest over a specified period. The interest rate can be fixed or variable, impacting the total repayment amount. Loans can be secured, backed by collateral like a house or car, or unsecured, relying solely on the borrower’s creditworthiness. Common types include mortgages, auto loans, personal loans, student loans, and business loans. The loan application process involves submitting an application, undergoing a credit check, and, if approved, signing a loan agreement. Loans offer benefits like access to substantial funds and credit-building opportunities but also come with risks such as debt burden, interest costs, and potential default consequences. Regulatory measures are in place to ensure fair practices and protect both parties involved in the lending process.

Lets's study about Loan
         
         ''')
    
    df=df.dropna()
    df=df.reset_index()
    df=df.drop('index',axis=1)
    df=df.drop(['Loan ID','Customer ID'],axis=1)
    st.write(df)

    st.subheader("Let's check the parameters for evaluating the eligibility for the loan.")

    cols=st.selectbox("Select the Feature to understand.",df.columns)

    if cols:
        st.text(df[cols].describe())
                
                
            # Cleaning data

        df['Current Loan Amount']=df['Current Loan Amount'].replace(99999999,df['Current Loan Amount'].median())
        df['Purpose']=df['Purpose'].replace('other','Other')

        col_1=st.selectbox("Enter the column to be studied",df.select_dtypes(include='object').columns)
        if col_1:
            col_2=st.selectbox("Enter the metric to be studied",df.select_dtypes(exclude='object').columns)
            stat_metric=st.selectbox("Enter the statistical mesure to be studied",["mean","median","count",'sum'])
            if stat_metric=="mean":
                    st.bar_chart(df.groupby(col_1)[col_2].mean())
            if stat_metric=="median":
                    st.bar_chart(df.groupby(col_1)[col_2].median())
            if stat_metric=="count":
                    st.bar_chart(df.groupby(col_1)[col_2].count())
            if stat_metric=="sum":
                    st.bar_chart(df.groupby(col_1)[col_2].sum())

                

            st.subheader("Let's study the dependency of numerical variables")

            col_3=st.selectbox("Enter the first numerical variable",df.select_dtypes(exclude='object').columns)

            col_4=st.selectbox("Enter the second numerical variable",df.select_dtypes(exclude='object').columns)

            hue=st.selectbox("Select the categorical variable to displayed",df.select_dtypes(include='object').columns)

            chart_data=pd.DataFrame(df[[col_3,col_4]])
            #st.scatter_chart(chart_data,x=f"{col_3}",y=f"{col_4}")


            import altair as alt


            c = alt.Chart(df).mark_circle().encode(
            x=col_3, y=col_4, color=hue)

            #col_left,col_right=st.columns(2)

            #with col_left:
            st.altair_chart(c, use_container_width=True)

            #with col_right:
            st.metric("Correlation",round(df[col_3].corr(df[col_4]),3))

if options=="Modelling and Prediction":
     st.title("Let's clean the data and make a model")
     df=df.dropna()
     df=df.reset_index()
     df=df.drop('index',axis=1)
     df=df.drop(['Loan ID','Customer ID'],axis=1)
     st.write(df.head())

    #  st.text("Categorical columns")
 
    #  cols=st.selectbox("Select the column",df.select_dtypes(include="object").columns)
    #  st.text(df[cols].unique())

    #  st.text("Numerical columns")
 
    #  cols=st.selectbox("Select the column",df.select_dtypes(exclude="object").columns)
    #  st.text(df[cols].unique())
     

     y=df['Loan Status']
     x=df.drop('Loan Status',axis=1)
     
     st.subheader("Statistical Significance")
     
     sel_col=st.selectbox("Select the columns",x.columns)
     sig_cols=[]
     insig_cols=[]
     if x[sel_col].dtype=="object":
          stat,pvalue,dof,exp= stats.chi2_contingency(pd.crosstab(y,x[sel_col]))
          with st.spinner("Checking Significance"):
               time.sleep(5)
          if pvalue<0.05:
               
               st.warning("Feature Insignificant" )
               insig_cols.append(sel_col)
          else:
               st.text("Significant")
               sig_cols.append(sel_col)

     else:
          a=df[df['Loan Status']=='Fully Paid'][sel_col]
          b=df[df['Loan Status']=='Charged Off'][sel_col]
          stat,pvalue=stats.ttest_ind(a,b)
          with st.spinner("Checking Significance"):
               time.sleep(5)
          if pvalue<0.05:
               st.warning("Feature Insignificant" )
               insig_cols.append(sel_col)
          else:
               st.text("Feature Significant")
               sig_cols.append(sel_col)


     x['Maximum Open Credit']=x['Maximum Open Credit'].replace(798255370.0,x['Maximum Open Credit'].median())
     x['Purpose']=x['Purpose'].replace({'Other':'other'})
     x['Current Loan Amount']=x['Current Loan Amount'].replace(99999999.0,np.nan)
     x['Current Loan Amount']=x['Current Loan Amount'].replace(np.nan,x['Current Loan Amount'].median())
    #  x['Maximum Open Credit']=pd.Series(stats.yeojohnson(x['Maximum Open Credit'])[0])
    #  x['Credit Score']=pd.Series(stats.yeojohnson(x['Credit Score'])[0])
    #  x['Annual Income']=pd.Series(stats.yeojohnson(x['Annual Income'])[0])
     x['Term']=x['Term'].replace({'Short Term':0,'Long Term':1})
     x['Years in current job']=x['Years in current job'].replace({'8 years':8,'< 1 year':0,'2 years':2,'3 years':3,'10+ years':10,'4 years':4,'6 years':6,'7 years':7,'1 year':1,'9 years':9,'5 years':5})
     x['Home Ownership']=x['Home Ownership'].map(x['Home Ownership'].value_counts(normalize=True))
     x['Purpose']=x['Purpose'].map(x['Purpose'].value_counts(normalize=True))
     
     y=y.replace({'Fully Paid':1,'Charged Off':0})
     

     rf=RandomForestClassifier()
     rf_model=rf.fit(x,y)

     st.header("Predict the Loan Status")
     

     var1=st.number_input(x.columns[0],min_value=x[x.columns[0]].min(),max_value=x[x.columns[0]].max())
     var2=st.number_input(x.columns[1],min_value=x[x.columns[1]].min(),max_value=x[x.columns[1]].max())
     var3=st.number_input(x.columns[2],min_value=x[x.columns[2]].min(),max_value=x[x.columns[2]].max())
     var4=st.number_input(x.columns[3],min_value=x[x.columns[3]].min(),max_value=x[x.columns[3]].max())
     var5=st.number_input(x.columns[4],min_value=x[x.columns[4]].min(),max_value=x[x.columns[4]].max())
     var6=st.number_input(x.columns[5],min_value=x[x.columns[5]].min(),max_value=x[x.columns[5]].max())
     var7=st.number_input(x.columns[6],min_value=x[x.columns[6]].min(),max_value=x[x.columns[6]].max())
     var8=st.number_input(x.columns[7],min_value=x[x.columns[7]].min(),max_value=x[x.columns[7]].max())
     var9=st.number_input(x.columns[8],min_value=x[x.columns[8]].min(),max_value=x[x.columns[8]].max())
     var10=st.number_input(x.columns[9],min_value=x[x.columns[9]].min(),max_value=x[x.columns[9]].max())
     var11=st.number_input(x.columns[10],min_value=x[x.columns[10]].min(),max_value=x[x.columns[10]].max())
     var12=st.number_input(x.columns[11],min_value=x[x.columns[11]].min(),max_value=x[x.columns[11]].max())
     var13=st.number_input(x.columns[12],min_value=x[x.columns[12]].min(),max_value=x[x.columns[12]].max())
     var14=st.number_input(x.columns[13],min_value=x[x.columns[13]].min(),max_value=x[x.columns[13]].max())
     var15=st.number_input(x.columns[14],min_value=x[x.columns[14]].min(),max_value=x[x.columns[14]].max())
     var16=st.number_input(x.columns[15],min_value=x[x.columns[15]].min(),max_value=x[x.columns[15]].max())

     prediction=rf_model.predict([[var1,var2,var3,var4,var5,var6,var7,var8,var9,var10,var11,var12,var13,var14,var15,var16]])
     with st.spinner("PREDICTING THE LOAN STATUS"):
            time.sleep(5)
     if prediction==0:
          st.warning("CHARGED OFF")
     else:
          st.success("FULLY PAID")
          st.balloons()


     
     









     




     




