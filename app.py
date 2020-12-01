import streamlit as st
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go

import streamlit.components.v1 as components
import os

## get the dataframe
##

# from WorkforceSentimentMonitoring.data import get_data, merge
# from WorkforceSentimentMonitoring.data import get_prepaired_data

# submission, train, test = get_data()
# df = merge(submission, train, test)

df = pd.read_csv('./raw_data/pred_sample.csv').drop('Unnamed: 0', axis=1)


columns = ['work-balance',
           'culture-values',
           'career-opportunities',
           'comp-benefits',
           'senior-mgmt',
           'overall'
           ]

df = df[columns]

# reorganize columns
categories = ['overall',
              'work-balance',
              'culture-values',
              'career-opportunities',
              'comp-benefits',
              'senior-mgmt'
              ]

df= df.reindex(columns=categories)


# count pos, neg, neutral reviews
# pos_counts = df[df>=4].count()
# neg_counts = df[df<=2].count()
# neutral_counts = df[df==3].count()

pos_counts = df[df==2].count()
neg_counts = df[df==0].count()
neutral_counts = df[df==1].count()

sentiment = ['positive',
             'negative',
             'neutral'
            ]


df_counts = pd.DataFrame([pos_counts, neg_counts, neutral_counts], index=sentiment)
total_counts = df.shape[0]
for category in categories:
    df_counts[f'{category}_counts'] = (df_counts[f'{category}']/total_counts)**100


# creating pie charts according to categories
figures = []
for category in categories:

    labels = ['positive', 'negative', 'neutral']
    values = list(df_counts[f'{category}'])
    index_max = np.argmax(values)
    pull = [0,0,0]
    #pull[index_max] = 0.2
    marker = {'colors': [
                     '#ceeed8',
                     '#ffd5d5',
                     'lightblue',
                    ]}

    fig = go.Figure(data=[go.Pie(labels=labels, values=values, pull=pull, textinfo='label+percent', title=category, marker=marker)])
    figures.append(fig)
    #st.plotly_chart(fig)

 # reorganize df for bar charts

df_bar = pd.DataFrame()
for category in categories:
    df_bar[f'{category}'] = df_counts[f'{category}']
df_bar = df_bar.transpose()

df_bar.negative = (df_bar.negative/df_bar.negative.sum())*100
df_bar.positive = (df_bar.positive/df_bar.positive.sum())*100
df_bar.neutral = (df_bar.neutral/df_bar.neutral.sum())*100
df_bar.reset_index(inplace=True)
df_bar.rename(columns = {'index':'review topics','positive':'positive [%]', 'negative':'negative [%]', 'neutral':'neutral [%]'}, inplace=True)






### streamlit

st.sidebar.title("Visualization Selector")
st.sidebar.write('Monitoring categories:')
st.sidebar.write('- **Overall satisfaction**')
st.sidebar.write('- **Work-Life-Balance**')
st.sidebar.write('- **Company culture**')
st.sidebar.write('- **Career opportunities**')
st.sidebar.write('- **Company benefits**')
st.sidebar.write('- **Senior management**')

select = st.sidebar.selectbox('Select category you want to visualize', categories)


st.markdown("""
    # Workforce Sentiment Analysis
    ### Hello :wave: and welcome to your **monitoring dashboard** :chart_with_upwards_trend:
    #### :white_check_mark:  When your employees are happy, they feel invested in the organisation's goals and are more compelled to their work
    #### :white_check_mark:  Find out about your employees happiness
    #### :white_check_mark:  Improve and boost your working environment :rocket:
""")

space = '''<br>'''

components.html(space, height=50, width=1200)






st.markdown('## Sentiment Analysis')
for category in categories:
    if select == category:
        st.info(f'Employees\' satisfaction: **{category}**')


        if st.checkbox('Show Graph', True, key=104):
            labels = ['positive', 'negative', 'neutral']
            values = list(df_counts[f'{category}'])
            index_max = np.argmax(values)
            pull = [0,0,0]
            #pull[index_max] = 0.2
            marker = {'colors': [
                             '#ceeed8',
                             '#ffd5d5',
                             'lightblue',
                            ]}

            fig = go.Figure(data=[go.Pie(labels=labels, values=values, pull=pull, textinfo='label+percent', title=category, marker=marker)])
            st.plotly_chart(fig)
            # if index_max == 0:
            #     st.success('This is a success!')
            # elif index_max == 1:
            #     st.error('Let\'s keep positive, this might be pretty close to a success!')
            # else: st.warning('This is a semi success')




        if select == categories[0]:
          st.markdown(f'## Indepth sentiment analysis')
          option = st.selectbox('Select a line to filter', labels, key=100)
          if option == 'positive':
              color = '#dffde9'
              fig = px.bar(df_bar, x='review topics', y='positive [%]', title="Positive Reviews")
              fig.update_traces(marker_color=color)
              st.plotly_chart(fig)
          elif option == 'negative':
              color = 'red'
              fig = px.bar(df_bar, x='review topics', y='negative [%]', title="Negative Reviews")
              fig.update_traces(marker_color=color)
              st.plotly_chart(fig)
          else:
              color = 'lightblue'
              fig = px.bar(df_bar, x='review topics', y='neutral [%]', title="Neutral Reviews")
              fig.update_traces(marker_color=color)
              st.plotly_chart(fig)





        st.header("Review Topics")


        topics = ['positive topics', 'negative topics', 'neutral topics']
        option_2 = st.selectbox('Select a line to filter', topics, key=101)
        if option_2 == 'negative topics':
            path = os.path.join(os.getcwd(), './notebooks/lda.html')
            HtmlFile = open(path, 'r', encoding='utf-8')
            source_code = HtmlFile.read()
            print(source_code)
            components.html(source_code, height=1000, width=1600)



#expander = st.beta_expander('Overall')






















# ## feedback
# st.success
# st.info('This is an info')
# st.warning('This is a semi success')
# st.error('Let\'s keep positive, this might be pretty close to a success!')









    #if st.sidebar.checkbox("Show Analysis of selected Category", False, key=2):

#figures


# #get the state selected in the selectbox
# state_data = df[df['state'] == select]
# select_status = st.sidebar.radio("Covid-19 patient's status", ('Confirmed',
# 'Active', 'Recovered', 'Deceased'))
