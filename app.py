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

from WorkforceSentimentMonitoring.data import get_data, merge

## initial dataframe

submission, train, test = get_data()
df = merge(submission, train, test)

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
pos_counts = df[df>=4].count()
neg_counts = df[df<=2].count()
neutral_counts = df[df==3].count()

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


# st.title('Workforce Sentiment Analysis')
# st.markdown('### Welcome to your **monitoring dashboard**: Find out how happy your employees are :white_check_mark:')

# txt = st.text_area('Text to analyze', '''
#     It was the best of times, it was the worst of times, it was
#     the age of wisdom, it was the age of foolishness, (...)
#     ''')


    # **bold** or *italic* text with [links](http://github.com/streamlit) and:
    # - bullet points



#st.sidebar.checkbox("Show Analysis by Category", True, key=1)


st.markdown("""
    # Workforce Sentiment Analysis
    ### Hello :wave: and welcome to your **monitoring dashboard** :chart_with_upwards_trend:
    ### :white_check_mark:  When your employees are happy, they feel invested in the organisation's goals and are more compelled to their work
    ### :white_check_mark:  Find out about your employees happiness
    ### :white_check_mark:  Improve and boost your working environment :rocket:
""")

st.info('Find below indepth information on the:')
st.write('- **Overall** happiness of your company')
st.write('''- Indepth analysis of topics:
                - **Work-Life-Balance**
                - **Company culture**
                - **Career opportunities**
                - **Company benefits**
                - **Senior management**
        ''')

st.sidebar.title("Visualization Selector")
select = st.sidebar.selectbox('Select a Category', categories)

for category in categories:
    if select == category:
        st.markdown(f'## {category} sentiment of your employees')

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





        st.markdown('## Overall sentiment of your employees')
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
