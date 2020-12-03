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

# from WorkforceSentimentMonitoring.data import get_data, merge
# from WorkforceSentimentMonitoring.data import get_prepaired_data

# submission, train, test = get_data()
# df = merge(submission, train, test)

#df = pd.read_csv('./raw_data/pred_sample.csv').drop('Unnamed: 0', axis=1)



# page_bg_img = '''
# <style>
# body {
# background-image: url("https://lh3.googleusercontent.com/proxy/sZTTe5weHc6DQ4yQHEhFMhtfaiZsYnnM38wB1JTbeYMEvaVg_BZtZrPtBo5GrA4itBkWxmUftHeKsYDCxTs1Brt2PwGX1uo8Uw4KBklORULlu942cCxP1G0LMlqEBXuudzXJwr5wvYc");
# background-size: cover;
# }
# </style>
# '''

# st.markdown(page_bg_img, unsafe_allow_html=True)



# ----------------------------------
#      STREAMLIT FRONTEND START
# ----------------------------------


columns = ['work-balance',
             'culture-values',
             'career-opportunities',
             'comp-benefits',
             'senior-mgmt',
             'overall'
             ]

# reorganize columns order

categories = ['overall',
              'work-balance',
              'culture-values',
              'career-opportunities',
              'comp-benefits',
              'senior-mgmt'
              ]

categories_new = ['Overall Satisfaction',
              'Work-Life-Balance',
              'Culture values',
              'Career opportunities',
              'Company benefits',
              'Senior management'
              ]

topics = ['Work-Life-Balance',
          'Culture values',
          'Career opportunities',
          'Company benefits',
          'Senior management'
          ]

# ----------------------------------
#     Introduction
# ----------------------------------

st.sidebar.title("Visualization Selector")
st.sidebar.write('Monitoring categories:')
st.sidebar.write('- **Overall satisfaction**')
st.sidebar.write('- **Work-Life-Balance**')
st.sidebar.write('- **Culture values**')
st.sidebar.write('- **Career opportunities**')
st.sidebar.write('- **Company benefits**')
st.sidebar.write('- **Senior management**')

select = st.sidebar.selectbox('Select category you want to visualize', categories_new)


st.markdown("""
      # Workforce Sentiment Analysis
      ### Hello :wave: and welcome to your **monitoring dashboard** :chart_with_upwards_trend:
      ### :white_check_mark:  When your employees are happy, they feel invested in the organisation's goals and are more compelled to their work
      ### :white_check_mark:  Find out about your employees happiness
      ### :white_check_mark:  Improve and boost your working environment :rocket:
  """)

space = '''<br>'''

components.html(space, height=50, width=1200)


# ----------------------------------
#     File import
# ----------------------------------

my_expander = st.beta_expander('File upload')
my_expander.info('Please upload a *CSV* file :open_file_folder:')


def try_read_df(f):
    try:
        return pd.read_csv(f)
    except:
        return pd.read_excel(f)

df = pd.DataFrame()
uploaded_file = my_expander.file_uploader("Choose a file")
try:
  if uploaded_file is not None:
    df = try_read_df(uploaded_file)
    df = df[columns]
    df = df.reindex(columns=categories)
    for cat, cat_new in zip (categories,categories_new):
      df = df.rename(columns={f'{cat}':f'{cat_new}'})
    my_expander.success('You have succesfully uploaded your *CSV* file')
    #st.dataframe(uploaded_file)
    clicked = my_expander.button('Check file format')
    if clicked:
      my_expander.write(df)
except ImportError:
  my_expander.error('**Error** : You did not upload the correct file format')


# df = pd.DataFrame()
# uploaded_file = my_expander.file_uploader("Choose a file")
# if uploaded_file is not None:
#   df = pd.read_csv(uploaded_file)
#   df = df[columns]
#   df = df.reindex(columns=categories)
#
# else:
#   my_expander.error('**Error** : You did not upload the correct file format')
    # except ParserError:
    # try:
    #   df = pd.read_csv(uploaded_file)
    #   df = df[columns]
    #   df= df.reindex(columns=categories)
    #   my_expander.success('You have succesfully uploaded your.csv file')
    # except ParserError:
    # #   my_expander.error('**Error** : You did not upload the correct file format')

components.html(space, height=50, width=1200)


# ----------------------------------
#     Visualizations
# ----------------------------------


if df.shape[1] == len(categories):

  ## here prediction of scores 1-5

  # count pos, neg, neutral reviews
  # pos_counts = df[df>=4].count()
  # neg_counts = df[df<=2].count()
  # neutral_counts = df[df==3].count()

  ## here prediction of pos:2 , neutral:1 ,  neg:0

  # pos_counts = df[df==2].count()
  # neg_counts = df[df==0].count()
  # neutral_counts = df[df==1].count()

  ## here prediction of pos:1 , neg:0

  neg_counts = df[df==0].count()
  pos_counts = df[df==1].count()
  neutral_counts = df[df==2].count()

  sentiment = ['positive',
               'negative',
               'neutral'
              ]


  df_counts = pd.DataFrame([pos_counts, neg_counts, neutral_counts], index=sentiment)
  total_counts = df.shape[0]
  for category in categories_new:
      df_counts[f'{category}_counts'] = (df_counts[f'{category}']/total_counts)**100


  # add this line if we are doing a binary classification
  df_counts = df_counts.iloc[:2]

  # creating pie charts according to categories
  figures = []
  for category in categories_new:

      #labels = ['positive', 'negative', 'neutral']

      # for binary classification task
      labels = ['positive', 'negative']
      values = list(df_counts[f'{category}'])
      index_max = np.argmax(values)
      pull = [0,0,0]
      pull[index_max] = 0.2
      marker = {'colors': [
                       'green',
                       'red',
                       'lightblue',
                      ]}

      fig = go.Figure(data=[go.Pie(labels=labels, values=values, pull=pull, textinfo='label+percent', title=category, marker=marker)])
      figures.append(fig)
      #st.plotly_chart(fig)

   # reorganize df for bar charts

  df_bar = pd.DataFrame()
  for category in categories_new:
      df_bar[f'{category}'] = df_counts[f'{category}']
  df_bar = df_bar.transpose()

  df_bar.negative = (df_bar.negative/df_bar.negative.sum())*100
  df_bar.positive = (df_bar.positive/df_bar.positive.sum())*100
  #df_bar.neutral = (df_bar.neutral/df_bar.neutral.sum())*100
  df_bar.reset_index(inplace=True)
  #df_bar.rename(columns = {'index':'review topics','positive':'positive [%]', 'negative':'negative [%]', 'neutral':'neutral [%]'}, inplace=True)
  df_bar.rename(columns = {'index':'review topics','positive':'positive [%]', 'negative':'negative [%]'}, inplace=True)
  df_bar = df_bar.drop(0)
  st.write(df_bar)

  components.html(space, height=50, width=1200)



  my_expander_2 = st.beta_expander('Sentiment Analysis')

  my_expander_2.markdown('## **Overall Satisfaction**')
  labels = ['positive', 'negative']
  values = list(df_counts['Overall Satisfaction'])
  index_max = np.argmax(values)
  pull = [0,0,0]
  #pull[index_max] = 0.2
  marker = {'colors': ['green',
                       'red',
                       'lightblue',
                        ]}

  fig = go.Figure(data=[go.Pie(labels=labels, values=values, pull=pull, textinfo='label+percent', marker=marker)])
  my_expander_2.plotly_chart(fig)

  my_expander_3 = st.beta_expander('Topic-specific Sentiment')
  if select == categories_new[0]:
    option = my_expander_3.selectbox('Select a line to filter', ['Positive Reviews', 'Negative Reviews'], key=100)
    if option == 'Positive Reviews':
      # my_expander_3.markdown(f'## **Positive sentiment analysis**')
      color = 'green'
      fig = px.bar(df_bar, x='review topics', y='positive [%]', title="Positive Reviews")
      fig.update_traces(marker_color=color)
      my_expander_3.plotly_chart(fig)
    elif option == 'Negative Reviews':
      # my_expander_3.markdown(f'## **Negative sentiment analysis**')
      color = 'red'
      fig = px.bar(df_bar, x='review topics', y='negative [%]', title="Negative Reviews")
      fig.update_traces(marker_color=color)
      my_expander_3.plotly_chart(fig)
    else:
      components.html(space, height=50, width=1200)

#     color = 'lightblue'
#     fig = px.bar(df_bar, x='review topics', y='neutral [%]', title="Neutral Reviews")
#     fig.update_traces(marker_color=color)
#     my_expander_3.plotly_chart(fig)



  for topic in topics:
      if select == topic:
          # st.info(f'Employees\' satisfaction: **{category}**')


          if st.checkbox('Show Graph', True, key=104):
              labels = ['positive', 'negative']
              values = list(df_counts[f'{topic}'])
              index_max = np.argmax(values)
              pull = [0,0,0]
              #pull[index_max] = 0.2
              marker = {'colors': [
                               'green',
                               'red',
                               'lightblue',
                              ]}

              fig = go.Figure(data=[go.Pie(labels=labels, values=values, pull=pull, textinfo='label+percent', title=topic, marker=marker)])
              st.plotly_chart(fig)
              # if index_max == 0:
              #     st.success('This is a success!')
              # elif index_max == 1:
              #     st.error('Let\'s keep positive, this might be pretty close to a success!')
              # else: st.warning('This is a semi success')

          components.html(space, height=50, width=1200)

          my_expander_4 = st.beta_expander('Review topics')
          my_expander_4.header("**Relevant Topics**")


          selection = ['Positive topics', 'Negative topics']
          option_2 = my_expander_4.selectbox('Select a line to filter', selection, key=101)
          if option_2 == 'Negative topics':
              path = os.path.join(os.getcwd(), './notebooks/pyLDAvis2.html')
              HtmlFile = open(path, 'r', encoding='utf-8')
              source_code = HtmlFile.read()
              print(source_code)
              components.html(source_code, height=1000, width=1600)
else:

  components.html(space, height=50, width=1200)


