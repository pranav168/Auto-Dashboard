import warnings

from pandas.core.dtypes.missing import notnull
warnings.filterwarnings("ignore")
from numpy.testing._private.utils import suppress_warnings
import streamlit as st
import openpyxl

import datetime
import pandas as pd
import numpy as np
import streamlit as st 
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits import mplot3d

import pandasql as ps
import sqlite3

import plotly.express as px

from sklearn.ensemble import IsolationForest, RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler

st.set_option('deprecation.showPyplotGlobalUse', False)
st.set_page_config(layout="wide")


st.title("Auto Dashboard")
st.markdown("The dashboard will help a researcher to get to know \
more about the given datasets and it's output")

file_buffer = st.file_uploader("Upload an excel file", type=["xlsx"])

#server.maxUloadSize -- Increase File Size

@st.cache(allow_output_mutation=True)
def load_data(file_buffer=file_buffer):
    # try:
    df= openpyxl.load_workbook(file_buffer)
    # except:
    #     df=pd.read_csv(file_buffer)
    #     datatoexcel=pd.ExcelWriter('Dataframe.xlsx')
    #     df=df.to_excel(datatoexcel)
    return df


def dtype_check(column_name, PRIMARY_KEY='', SYMBOL='🟡'):
    if str(df[column_name].dtype)== 'int64':
        st.sidebar.text(f'{SYMBOL}|📂 {column_name} (int) {PRIMARY_KEY}')
    elif str(df[column_name].dtype)== 'float64':
        st.sidebar.text(f'{SYMBOL}|📂 {column_name} (float) {PRIMARY_KEY} ')
    elif str(df[column_name].dtype)== 'bool':
        st.sidebar.text(f'{SYMBOL}|📂 {column_name} (bool) {PRIMARY_KEY} ')
    elif str(df[column_name].dtype)== 'object':
        st.sidebar.text(f'{SYMBOL}|📂 {column_name} (str) {PRIMARY_KEY} ') 
    elif str(df[column_name].dtype)== 'datetime64':
        st.sidebar.text(f'{SYMBOL}|📂 {column_name} (datetime) {PRIMARY_KEY} ')
    elif str(df[column_name].dtype)== 'category':
        st.sidebar.text(f'{SYMBOL}|📂 {column_name} (Lists\ Dict) {PRIMARY_KEY} ')  
    else:
        pass
    return np.nan

def factored_addition(x):
        sum=0
        while x>0:
            x-=1
            sum+=x
        return sum

def analysis(df):    

    st.title("Statistical Analysis")  
    try:
        st.table(df.describe())
    except:
        st.text('No Numerical Data found')
    
    df.dropna(inplace=True)
    try:
        st.title('Correlation Graph')
        corr = df.corr()                                                                #plotting co-relation chart
        plt.figure(figsize=(25,15))
        sns.heatmap(corr, annot=True)
        st.pyplot()
    except:
        'Unable to make Correlation Graph'

    # if len(int_list)>0:
    st.title('Plots')

    charts=[]

    final_list=df.columns

    x=len(final_list)
    for i in range(1,factored_addition(x)+1):
        charts.append(f'chart_{i}')

    for i in range(0,len(charts),2):
        try:
            charts[i],charts[i+1]=st.columns(2)
        except:
            try:
                charts[-1]=st.columns(1)
            except:
                pass
    
    count=0

    for n,i in enumerate(final_list):
        plot_df=pd.DataFrame()
        plot_df[i]=df[i]

        try:
        
            for j in final_list[n+1:]:
                if i != j:
                    plot_df[j]=df[j]
                    
                    with charts[count]:
                        count+=1
                        try:
                            # st.bar_chart(x=plot_df[i],y=plot_df[j])
                            fig = px.bar(plot_df, x=i, y=j)
                            fig.update_layout(width=1100,height=600) #width=2200,height=900
                            st.plotly_chart(fig,width=1100,height=600) #,width=2200,height=900
                            # st.plotly_chart(fig)
                        except:
                            try:
            
                                fig = px.line(plot_df, x=i, y=j)
                                st.plotly_chart(fig)
                            except:
                                pass
                    plot_df=pd.DataFrame()
                    plot_df[i]=df[i]
        
        except:
            pass

def sql_query(user_input):
            q1=str(user_input)
            querried_df= ps.sqldf(q1)
            return querried_df

@st.cache(allow_output_mutation=True)
def create_df(sheet_name):
    created_df= pd.read_excel(file_buffer,sheet_name=sheet_name)
    return created_df

if file_buffer:
    st.sidebar.title('Schema')
    schema=st.sidebar.checkbox('CheckBox to Show Schema',True)
    st.sidebar.title('Dataset')
    sheets=st.sidebar.checkbox('CheckBox to Show Dataset', True)
 
    if schema:
        df=load_data()
        for i in df.get_sheet_names():
            st.sidebar.title(f'📁 {i.replace(" ","_")}')
            df= pd.read_excel(file_buffer,sheet_name=i)
            for j in df.columns:  
                if len(df[j].value_counts())== len(df[j]):
                    dtype_check(column_name=j,PRIMARY_KEY='(PRIMARY KEY)', SYMBOL='🔴')
                else:
                    dtype_check(column_name=j)

    if sheets:
        df=load_data()
        sheet_names=df.get_sheet_names()
        
        dataframe=[]

        for i in sheet_names:
            j=i.replace(' ','_')
            dataframe.append(j)
        
        input_sheets=st.sidebar.multiselect('Select Table', options=sheet_names)

        dict_dataset={}
        
        for i in dataframe:
                try:
                    dict_dataset[i]=create_df(i)
                except:
                    dict_dataset[i]=create_df(i.replace('_',' '))
                
        if input_sheets: 
            for i in input_sheets:
                i=i.replace(' ','_')
                st.header(i)
                st.dataframe(dict_dataset[i])
                df=dict_dataset[i]
                # st.text(dataframe)
                analysis(df)
        
        for num,i in enumerate(dict_dataset):
            globals()[i]=dict_dataset[dataframe[num]]
        
        user_input = st.sidebar.text_area("Type SQL Query")   
        if user_input:
            try:
                querried_df=sql_query(user_input)
                st.dataframe(querried_df)
            except:
                st.text('Invalid Query')
            try:
                analysis(querried_df)
            except:
                pass


            
            

                

            


            




    