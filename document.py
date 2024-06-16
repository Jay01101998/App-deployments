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
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
import time

#from langchain.vectorstores import faiss
from langchain.text_splitter import RecursiveCharacterTextSplitter
#st.set_option('deprecation.showPyplotGlobalUse', False)
import gensim
from gensim.models import Word2Vec
from gensim.models import KeyedVectors

st.set_page_config(layout='wide',page_title="Word Similarity using Word2Vec")
st.title("Word Similarity using Word2Vec")

st.caption("This application can be used to display the content of a document and study the similarity between the words present in the document content. In order to achieve the same WORD2VEC model for Gensim is used.")

def preprocessing_text(text):
    #stemmer = PorterStemmer()
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


uploader=st.file_uploader("Upload the document",accept_multiple_files=True)

if uploader:

    texts=""
    for upload in uploader:
        reader=PdfReader(upload)
        content="\n".join(page.extract_text() for page in reader.pages)
        texts+=content
        texts=preprocessing_text(texts)

        splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)

    text_split=splitter.split_text(texts)
    # for i in text_split:
    #     st.text(i)
    
    sentences=[i.split(" ") for i in text_split]

    st.subheader("Document Content")
    
    with st.spinner(text="Displaying the content....."):
        time.sleep(4)
        st.write(texts)

    model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=4)

    model.save("word2vec.model")

    
    st.header("Let's study the similarity between the words:")
    

    col1,col2=st.columns(2)
    with col1:
       st.subheader("Select the words")
       word_1=st.selectbox("Select the word 1",preprocessing_text(content).split(" "))
       vec_1 = model.wv[word_1]
   
       word_2=st.selectbox("Select the word 2",preprocessing_text(content).split(" "))
       vec_2 = model.wv[word_2]
       
       score=model.wv.similarity(word_1,word_2)
       st.text(f"The similarity between {word_1} and {word_2} is {score}")


    with col2:
       st.subheader("Let's visualize the similarity")
       chart_data = pd.DataFrame(zip(vec_1,vec_2), columns=["word_1","word_2"])
       st.scatter_chart(chart_data,x=f"word_1",y=f"word_2")
    
    # list_1=[]
    # a=preprocessing_text(content).split(" ")
    # for i in set(a):
    #     list_1.append(i)

    # list_2=[]
    # b=preprocessing_text(content).split(" ")
    # for i in set(b):
    #     list_2.append(i)  

    
    
    # for i,j in list(zip(list_1,list_2)):
    #     score=model.wv.similarity(model.wv[i],model.wv[j])
    #     st.write(i,j,": ",score)
