import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import warnings
from PIL import Image
warnings.filterwarnings("ignore")
plt.rcParams['figure.figsize']=[10,5]

st.title("GLOBAL SUPERSTORE")

st.image("superstore_img.jpeg",output_format='JPEG')

st.markdown('''The sample Dataset includes data for the Sales of multiple products sold by the store along with subsequent information related to geography, Product categories, and subcategories, sales, and profits, segmentation amongst the consumers, etc. 
            This sample Dataset presents a common use case, from which one can gather useful insights from the Sales data in order to improve the Marketing and Sales strategies. 
            One can learn about various operations and elements using this sample Dataset and come up with better strategies to improve and grow the business more.''')

df=pd.read_csv(r"C:\Users\HP\Desktop\self-projects\Superstore\Global_Superstore.csv")

st.subheader("Let's take a look at the dataset")

st.write(df.head())

st.write(f"The no.of Rows are : {df.shape[0]}")

st.write(f"The no.of Columns are : {df.shape[1]}")

st.subheader("Let's discuss the columns and its contents")

column=st.selectbox("Select the column",options=df.columns)

if df[column].dtype=='object':
   st.write(f"The unique values in {column.upper()} is {df[column].nunique()}")
   st.write(f"The most frequent value in {column.upper()} is {df[column].mode()[0]}")

else:
   st.write(f"The mean of {column.upper()} is {df[column].mean()}")
   st.write(f"The median of {column.upper()} is {df[column].median()}")
   st.write(f"The mode of {column.upper()} is {df[column].mode()[0]}")
   st.write(f"The skewness of {column.upper()} is {df[column].skew()}")



st.header("Let's visualize the quantitative metrics over time")

order_data=df.set_index('Order_Date')
order_data.index=pd.to_datetime(order_data.index)

st.write("Sales")
st.line_chart(order_data.groupby(order_data.index)['Sales'].sum(1000))


st.write("Quantity")
st.line_chart(order_data.groupby(order_data.index)['Quantity'].sum(1000))


st.write("Profit")
st.line_chart(order_data.groupby(order_data.index)['Profit'].sum(1000))


ship_data=df.set_index('Ship_Date')
ship_data.index=pd.to_datetime(ship_data.index)

duration=[]
for i in (ship_data.index-order_data.index):
    duration.append(float(str(i).split(" ")[0]))

delay_df=pd.DataFrame({'delay':duration,'profit':df['Profit'],'sales':df['Sales'],'quantity':df['Quantity']})




def metric_trend(metric):
   st.write(f"Date wise {metric}")
   if df[metric].dtype=='object':
      st.line_chart(order_data.groupby(order_data.index)[metric].count().head(1000))
   else:
      st.line_chart(order_data.groupby(order_data.index)[metric].mean().head(1000))

text=st.selectbox("Enter the metric to be studied over time",options=['Customer_ID','Order_ID','Product_ID'])

if text=='Customer_ID':
   metric_trend("Customer_ID")
if text=='Order_ID':
   metric_trend("Order_ID")
if text=='Profit':
   metric_trend("Profit")
if text=='Sales':
   metric_trend("Sales")
if text=='Quantity':
   metric_trend("Quantity")


selector=st.selectbox("Select the metric to be visualized",options=["Category","Sub Category","Market","Region","Segment"])
    
if selector=='Category':
    st.subheader("Category wise Metrics")

    left,right=st.columns(2)

    with left:
      st.bar_chart(df.groupby('Category')['Order_ID'].count())
      st.bar_chart(df.groupby('Category')['Sales'].sum())

    with right:
      st.bar_chart(df.groupby('Category')['Quantity'].sum())
      st.bar_chart(df.groupby('Category')['Profit'].sum())



if selector=='Sub Category':
    st.subheader("Sub Category wise Metrics")

    left,right=st.columns(2)

    with left:
      st.bar_chart(df.groupby('Sub_Category')['Order_ID'].count())
      st.bar_chart(df.groupby('Sub_Category')['Sales'].sum())

    with right:
      st.bar_chart(df.groupby('Sub_Category')['Quantity'].sum())
      st.bar_chart(df.groupby('Sub_Category')['Profit'].sum())

if selector=='Market':
    st.subheader("Market wise Metrics")

    left,right=st.columns(2)

    with left:
      st.bar_chart(df.groupby('Market')['Order_ID'].count())
      st.bar_chart(df.groupby('Market')['Sales'].sum())

    with right:
      st.bar_chart(df.groupby('Market')['Quantity'].sum())
      st.bar_chart(df.groupby('Market')['Profit'].sum())

if selector=='Region':
    st.subheader("Region wise Metrics")

    left,right=st.columns(2)

    with left:
      st.bar_chart(df.groupby('Region')['Order_ID'].count())
      st.bar_chart(df.groupby('Region')['Sales'].sum())

    with right:
      st.bar_chart(df.groupby('Region')['Quantity'].sum())
      st.bar_chart(df.groupby('Region')['Profit'].sum())


if selector=='Segment':
    st.subheader("Segment wise Metrics")

    left,right=st.columns(2)

    with left:
      st.bar_chart(df.groupby('Segment')['Order_ID'].count())
      st.bar_chart(df.groupby('Segment')['Sales'].sum())

    with right:
      st.bar_chart(df.groupby('Segment')['Quantity'].sum())
      st.bar_chart(df.groupby('Segment')['Profit'].sum())



import altair as alt

st.subheader("Quantitative dependencies")

def relation(metric):
   c = alt.Chart(df).mark_circle().encode(
   x='Sales', y='Profit', size='Quantity', tooltip=['Sales', 'Profit', 'Quantity'],color=metric)

   st.altair_chart(c, use_container_width=True)

select=st.selectbox("Select the qualitative variable to check upon the relation",options=["Category","Sub Category","Market","Region","Segment"])

if select=='Category':
   relation("Category")
if select=='Sub Category':
   relation("Sub_Category")
if select=='Market':
   relation("Market")
if select=='Region':
   relation("Region")
if select=='Segment':
   relation("Segment")



st.write("Distribution of Quantity")
fig, ax = plt.subplots()
ax.hist(df['Quantity'], bins=20)

st.pyplot(fig)

st.write("Distribution of Sales")
fig, ax = plt.subplots()
ax.hist(df['Sales'], bins=20)

st.pyplot(fig)

st.write("Distribution of Profit")
fig, ax = plt.subplots()
ax.hist(df['Profit'], bins=20)

st.pyplot(fig)


st.subheader("Quantitative metrics for each city")




country=st.text_input("Enter the country")


if country in df['Country'].unique():
   state = st.selectbox(f"Enter the state for the {country}",df[df['Country']==country]['State'].unique())
   city = st.selectbox(f"Enter the city for the {state}",df[df['State']==state]['City'].unique())
   st.write(f"The mean Profit is {df[df['City']==city]['Profit'].mean()}")
   st.write(f"The most frequent Quantity is {df[df['City']==city]['Quantity'].mode()[0]}")
   st.write(f"The mean Sales is {df[df['City']==city]['Sales'].mean()}")

   st.table(df[df['City']==city][['Discount','Profit','Quantity','Sales','Shipping_Cost']].describe())
  

