import nltk.sentiment
import streamlit as st
from streamlit_chat import message
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pypdf import PdfReader
import altair as alt
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize,sent_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')  
nltk.download('vader-lexicon') 
from nltk.sentiment import SentimentIntensityAnalyzer 
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
import time
import plotly.figure_factory as ff

def preprocessing_text(text):
    stemmer = PorterStemmer()
    #lemmatizer = WordNetLemmatizer()
    #emoji_pattern = r'^(?:[\u2700-\u27bf]|(?:\ud83c[\udde6-\uddff]){1,2}|(?:\ud83d[\udc00-\ude4f]){1,2}|[\ud800-\udbff][\udc00-\udfff]|[\u0021-\u002f\u003a-\u0040\u005b-\u0060\u007b-\u007e]|\u3299|\u3297|\u303d|\u3030|\u24c2|\ud83c[\udd70-\udd71]|\ud83c[\udd7e-\udd7f]|\ud83c\udd8e|\ud83c[\udd91-\udd9a]|\ud83c[\udde6-\uddff]|\ud83c[\ude01-\ude02]|\ud83c\ude1a|\ud83c\ude2f|\ud83c[\ude32-\ude3a]|\ud83c[\ude50-\ude51]|\u203c|\u2049|\u25aa|\u25ab|\u25b6|\u25c0|\u25fb|\u25fc|\u25fd|\u25fe|\u2600|\u2601|\u260e|\u2611|[^\u0000-\u007F])+$'
    text= text.lower()
    text = text.split()
    text = [word for word in text if word not in stopwords.words('english')]
    #text = [lemmatizer.lemmatize(word) for word in text]
    #text = [stemmer.stem(word) for word in text]
    text = ' '.join(text)  
    text = re.sub(r'[0-9]+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    #text = re.sub(emoji_pattern, '', text)
    text= re.sub(r'\s+', ' ', text)
    return text

st.set_page_config(page_title="WebMed Data Analysis",layout='wide')


st.title("WebMed Data Analysis")


df=pd.read_csv(r"C:\Users\HP\Desktop\self-projects\NLP datasets\drug_reviews_scrapped_data.csv\drug_reviews_scrapped_data.csv")

sia=SentimentIntensityAnalyzer()

st.subheader("The dataset contains the details of Drugs, the conditions they are administered and reviews for the same.")

drug=st.sidebar.selectbox("Select the drug",df['Drug'].unique())
if drug:
   condition=st.sidebar.selectbox("Select the condition",df[df['Drug']==drug]['Condition'].unique())
   if condition:
      st.write(f"For the condition {condition}, the reviews are: ")
      review=df[df['Condition']==condition]['Review']
      st.table(review.head())
      
      # clean=st.button("Clean the text")
      # if clean:
      #    #st.write("Cleaning the text")
      #    with st.spinner(text="Cleaning the text"):
      #       time.sleep(5)

      #    review=review.apply(preprocessing_text)

         #st.table(review.head())
      select=st.selectbox("Select the review",review.values)
      avg_score=[]
      if select:
         res=sia.polarity_scores(select)
         # st.write("The average compound sentiment compound score is:")
         # st.subheader(res['compound'])
         st.bar_chart(res)
      
      for i in review:
         res=sia.polarity_scores(i)
         avg_score.append(res['compound'])

      st.sidebar.write("Average Score sentiment score is :".upper())
      st.sidebar.header(np.mean(avg_score))
         

      # tf=TfidfVectorizer()
      # vec=tf.fit_transform(review).toarray()

         #st.button(label="Generate Components")
         # with st.spinner(text="Generating the components"):
         #       time.sleep(5)

         # comp=2
         # pca=PCA(n_components=comp,random_state=42)
         # x_pca=pd.DataFrame(pca.fit_transform(vec),columns=['PC'+str(i) for i in range(1,comp+1)])
         # st.metric(label="Correlation",value=(x_pca['PC1'].corr(x_pca['PC2'])))

         # st.write(x_pca.head())

         # st.scatter_chart(x_pca[['PC1','PC2']])
