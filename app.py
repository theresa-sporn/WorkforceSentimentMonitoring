import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import matplotlib.pyplot as plt

from WorkforceSentimentMonitoring.data import get_data, merge

st.title('Workforce Sentiment Dashbord')

st.sidebar.title("Visualization Selector")

# get the dataframe
submission, train, test = get_data()
df = merge(submission, train, test)

categories = ['work-balance', 'culture-values', 'career-opportunities','comp-benefits', 'senior-mgmt', 'overall']
df = df[categories]

# display in streamlit
st.dataframe(df[:1])

topics = ['work-balance', 'culture-values', 'career-opportunities','comp-benefits', 'senior-mgmt']

# get the dataframe
submission, train, test = get_data()
df = merge(submission, train, test)

categories = ['work-balance', 'culture-values', 'career-opportunities','comp-benefits', 'senior-mgmt', 'overall']
topics = ['work-balance', 'culture-values', 'career-opportunities','comp-benefits', 'senior-mgmt']
df = df[categories]

pos_counts = df[df>=4].count()
neg_counts = df[df<=2].count()
neutral_counts = df[df==3].count()

counts = ['positive', 'negative', 'neutral']

df_counts = pd.DataFrame([pos_counts, neg_counts, neutral_counts], index=counts)
total_counts = df.shape[0]


for category in categories:
    df_counts[f'{category}_counts'] = (df_counts[f'{category}']/total_counts)




st.sidebar.checkbox("Show Analysis by Category", True, key=1)
select = st.sidebar.selectbox('Select a Category', topics)


if st.sidebar.checkbox("Show Analysis by Category", True, key=2):
    st.markdown("## **Analysis**")
    #st.markdown("###" (select))
    if not st.checkbox('Hide Graph', False, key=1):

        figures = []

        for category in categories:

            labels = ['positive', 'negative', 'neutral']
            values = list(df_counts[f'{category}'])
            # print(values)
            index_max = np.argmax(values)
            # print(index_max)
            pull = [0,0,0]
            pull[index_max] = 0.2
            # print(pull)
            fig = go.Figure(data=[go.Pie(labels=labels, values=values, pull=pull)])
            #fig.show()
            st.plotly_chart(fig)
            figures.append(fig)

#figures


# #get the state selected in the selectbox
# state_data = df[df['state'] == select]
# select_status = st.sidebar.radio("Covid-19 patient's status", ('Confirmed',
# 'Active', 'Recovered', 'Deceased'))
