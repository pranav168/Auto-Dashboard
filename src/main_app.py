import warnings

from pandas.core.dtypes.missing import notnull
warnings.filterwarnings("ignore")
from numpy.testing._private.utils import suppress_warnings


import datetime

import pandas as pd
import pandasql as ps

import streamlit as st 

import AutoDashboard as ad
from AutoDashboard import dtype_check, load_data, factored_addition, analysis, create_df


st.set_option('deprecation.showPyplotGlobalUse', False)
st.set_page_config(layout="wide")


st.title("Auto Dashboard")
st.markdown("The dashboard will help a researcher to get to know \
more about the given datasets and it's output")


file_buffer = st.file_uploader("Upload an excel file", type=["xlsx"])

#server.maxUloadSize -- Increase File Size

if file_buffer:
    st.sidebar.title('Schema')
    schema=st.sidebar.checkbox('CheckBox to Show Schema',True)
    st.sidebar.title('Dataset')
    sheets=st.sidebar.checkbox('CheckBox to Show Dataset', True)
        
    if schema:
        df=load_data(file_buffer)
        for i in df.get_sheet_names():
            st.sidebar.title(f'📁 {i.replace(" ","_")}')
            df= pd.read_excel(file_buffer,sheet_name=i)
            for j in df.columns:  
                if len(df[j].value_counts())== len(df[j]):
                    dtype_check(df=df,column_name=j,PRIMARY_KEY='(PRIMARY KEY)', SYMBOL='🔴')
                else:
                    dtype_check(df=df,column_name=j)

    if sheets:
        df=load_data(file_buffer)
        sheet_names=df.get_sheet_names()
        
        dataframe=[]

        for i in sheet_names:
            j=i.replace(' ','_')
            dataframe.append(j)
        
        input_sheets=st.sidebar.multiselect('Select Table', options=sheet_names)

        dict_dataset={}
        
        for i in dataframe:
                try:
                    dict_dataset[i]=create_df(file_buffer,i)
                except:
                    dict_dataset[i]=create_df(file_buffer,i.replace('_',' '))
                
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

        def sql_query(user_input):
            q1=str(user_input)
            querried_df= ps.sqldf(q1)
            return querried_df
        
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


            
            

                

            


            




    