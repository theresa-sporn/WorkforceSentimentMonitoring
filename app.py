import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import matplotlib.pyplot as plt

## get the dataframe

from WorkforceSentimentMonitoring.data import get_data, merge

## initial dataframe

submission, train, test = get_data()
df = merge(submission, train, test)

categories = ['work-balance', 'culture-values', 'career-opportunities','comp-benefits', 'senior-mgmt', 'overall']
# topics = ['work-balance', 'culture-values', 'career-opportunities','comp-benefits', 'senior-mgmt']
df = df[categories]


# count pos, neg, neutral reviews

pos_counts = df[df>=4].count()
neg_counts = df[df<=2].count()
neutral_counts = df[df==3].count()

counts = ['positive', 'negative', 'neutral']

df_counts = pd.DataFrame([pos_counts, neg_counts, neutral_counts], index=counts)
total_counts = df.shape[0]


for category in categories:
    df_counts[f'{category}_counts'] = (df_counts[f'{category}']/total_counts)


# creating pie charts according to categories
figures = []
for category in categories:

    labels = ['positive', 'negative', 'neutral']
    values = list(df_counts[f'{category}'])
    index_max = np.argmax(values)
    pull = [0,0,0]
    pull[index_max] = 0.2
    #marker = ['#e6f2ff', '#ff6600', '#ff0000']


    fig = go.Figure(data=[go.Pie(labels=labels, values=values, pull=pull, textinfo='label+percent', title=category)])
    #fig.show()
    #st.plotly_chart(fig)
    figures.append(fig)


# streamlit

st.title('Workforce Sentiment Analysis')
st.markdown('### Welcome to your **monitoring dashboard**')

st.markdown('# Overall sentiment of your employees')

if not st.checkbox('Hide Graph', False, key=1):
    st.plotly_chart(figures[categories.index('overall')])

   # if

st.sidebar.title("Visualization Selector")

#st.dataframe(df[:10])


import streamlit as st
import streamlit.components.v1 as components


st.header("test html import")

import os

path = os.path.join(os.getcwd(), './notebooks/lda.html')

HtmlFile = open(path, 'r', encoding='utf-8')
source_code = HtmlFile.read()
print(source_code)
components.html(source_code, height=1000, width=1600)


st.success('This is a success!')
st.info('This is an info')
st.warning('This is a semi success')
st.error('Let\'s keep positive, this might be pretty close to a success!')




if not st.checkbox('Hide Graph', False, key=1):
    st.plotly_chart(figures[categories.index('overall')])


if st.sidebar.checkbox("Show Analysis of selected Category", False, key=1):
    select = st.sidebar.selectbox('Select a Category', topics)


if st.sidebar.checkbox("Show Analysis of selected Category", False, key=2):
    st.markdown("## **Analysis**")
    st.markdown("### %s " % (select))
    if not st.checkbox('Hide Graph', False, key=2):

        'Review topic:', select

        for topic in topics:
            if topic == select:
                st.plotly_chart(figures[topics.index(topic)])

#figures


# #get the state selected in the selectbox
# state_data = df[df['state'] == select]
# select_status = st.sidebar.radio("Covid-19 patient's status", ('Confirmed',
# 'Active', 'Recovered', 'Deceased'))
