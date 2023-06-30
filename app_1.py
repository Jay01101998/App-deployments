import streamlit as st


st.title("WELCOME TO MINI E-GUIDE FOR MACHINE LEARNING")
from PIL import Image

image = Image.open('ml.jpeg','r')

st.image(image,width=80,use_column_width='always')


st.header("Machine Learning: A Statistical Application")


st.write('''Machine learning is a method of data analysis that automates analytical model building. It is a branch of artificial intelligence based on the idea that systems can learn from data, identify patterns and make decisions with minimal human intervention.

Do you get automatic recommendations on Netflix and Amazon Prime about the movies you should watch next? Or maybe you get options for People You may know on Facebook or LinkedIn? You might also use Siri, Alexa, etc. on your phones. That’s all Machine Learning! This is a technology that is becoming more and more popular. Chances are that Machine Learning is used in almost every technology around you!''')

st.subheader('Types of Machine Learning: ')

st.text("1.  Regression")
st.text("2.  Classification")
st.text("3.  Clustering")


#st.checkbox("Select the folowing:",options=["Regression","Classification","Clustering"])


#st.selectbox('Select the algorithm for information',('Regression', 'Classification', 'Clustering'))

a = st.selectbox('Select the algorithm for information',('Regression', 'Classification', 'Clustering'))

if a=="Regression":
    st.write("Regression analysis is a form of predictive modelling technique which investigates the relationship between a dependent (target) and independent variable (s) (predictor). This technique is used for forecasting, time series modelling and finding the causal effect relationship between the variables. For example, relationship between rash driving and number of road accidents by a driver is best studied through regression.")
    st.latex('''
    Y = a + b*X1 + c*X2 + d*X3 + ϵ''')
    image = Image.open('linreg.jpeg','r')
    st.image(image,width=80,use_column_width='always')

elif a=="Classification":
    st.write("Classification is a task in data mining that involves assigning a class label to each instance in a dataset based on its features. The goal of classification is to build a model that accurately predicts the class labels of new instances based on their features.")
    image = Image.open('classification.jpeg','r')
    st.image(image,width=80,use_column_width='always')
else:
    st.write("Clustering is the task of dividing the population or data points into a number of groups such that data points in the same groups are more similar to other data points in the same group and dissimilar to the data points in other groups. It is basically a collection of objects on the basis of similarity and dissimilarity between them.")
    image = Image.open('cluster.jpeg','r')
    st.image(image,width=80,use_column_width='always')