import warnings

from pandas.core.dtypes.missing import notnull
warnings.filterwarnings("ignore")
from numpy.testing._private.utils import suppress_warnings
import streamlit as st

import pandas as pd
import numpy as np
import streamlit as st 
from colorama import Fore
import matplotlib.pyplot as plt
import seaborn as sns

st.set_option('deprecation.showPyplotGlobalUse', False)
st.set_page_config(layout="wide")


st.title("Auto Dashboard")
st.markdown("The dashboard will help a researcher to get to know \
more about the given datasets and it's output")

file_buffer = st.file_uploader("Upload an excel file", type=["xlsx","csv"])
if file_buffer:
    try:
        df= pd.read_excel(file_buffer)
    except:
        df=pd.read_csv(file_buffer)

    column_list=list(df.columns)
    int_list=[]
    cat_list=[]
    for i in column_list:
        try:
            a=df[i][1]/1.0
            int_list.append(i)
        except:
            cat_list.append(i)


st.sidebar.title("Build Your Custom Charts")
st.sidebar.markdown("Select the Charts/Plots accordingly:")

auto_mode=st.sidebar.checkbox("Auto Dashboard", True, key = 1)
custom_mode=st.sidebar.checkbox("Make Custom Plots", False, key = 2)
  
chart_visual = st.sidebar.selectbox('Select Charts/Plot type', 
                                    ('Line Chart', 'Bar Chart','Scatter Plot', 'Box Plot'))
  
if file_buffer:
    if chart_visual=='3D Scatter Plot':
        scatter_features=[]
        for i in range(3):
            temp = st.sidebar.selectbox(f'Select Feature {i+1}',
                                            options =set(int_list)-set(scatter_features) )
            scatter_features.append(temp)
    else:
        numlevel = st.sidebar.slider("Select the number of Numerical Features", 1, 2)
        # if numlevel==0:
        #     catlevel =2
        if numlevel<2:
            catlevel =1
        else:
            catlevel=0

        num_features=[]
        cat_features=[]
    
        for i in range(numlevel):
            temp = st.sidebar.selectbox(f'Numerical Feature {i+1}',
                                            options =set(int_list)-set(num_features) )
            num_features.append(temp)

        for i in range(catlevel):
            temp2 = st.sidebar.selectbox(f'Categorical Features {i+1}',
                                            options =set(cat_list)-set(cat_features) )
            cat_features.append(temp2)

    if auto_mode:
        try:
            count_dataset=pd.DataFrame()
            distinct_features=[]                                                                                          #Empty list to know the number of distict features,sum of all these values, and sum of values top 10 comprises
            for i in column_list:                                                                                               
                count_dataset[i]= pd.Series(df[i].value_counts().sort_values(ascending=False).head(10).index)      
                count_dataset[f'{i}_count']=pd.Series(df[i].value_counts().sort_values(ascending=False).head(10).values).astype('int')   
                distinct_features.append((len(df[i].value_counts().index),df[i].value_counts().sum(),df[i].value_counts().sort_values(ascending=False).head(10).sum())) 
                final_tally=list(zip(column_list,distinct_features))                                                           #Zipping with column_list
                def Nan_as_black(val):
                    if str(val)=='nan':
                        color = 'black'
                    return 'color: %s' % color
            count_dataset=count_dataset.style.apply(lambda x: pd.DataFrame(col_ref, index=count_dataset.index, columns=count_dataset.columns).fillna(''), axis=None).highlight_null('black').applymap(Nan_as_black)
            st.table(count_dataset)
        except:
            pass
        st.title("Dataset Head")
        st.table(df.head())

        st.title("Statistical Analysis")  
        st.table(df.describe())

        st.title('Corelation Graph')
        corr = df.corr()                                                                #plotting co-relation chart
        plt.figure(figsize=(25,15))
        sns.heatmap(corr, annot=True)
        st.pyplot()

        if len(int_list)>0:
            st.title('Plots for Numerical Features')
            for i in int_list:
                sns.distplot(df[i]) 
                st.pyplot()                                        #shows the overall distribution 
                sns.countplot(df[i])
                st.pyplot()
        if len(cat_list)>0:
            st.title('Plots for Categorical Features')
            for i in cat_list:
                sns.countplot(df[i])
                st.pyplot()

    if custom_mode:
        plot_df=pd.DataFrame()
        if num_features==0:
            plot_df[cat_features[0]]=df[cat_features[0]]
            plot_df[cat_features[1]]=df[cat_features[1]]
            # else:
            #     st.sidebar.text(f'{chart_visual} not Possible between 2 Categorical Features')
        
        elif len(num_features)==1:
            plot_df[num_features[0]]=df[num_features[0]]
            plot_df[cat_features[0]]=df[cat_features[0]]

        elif len(num_features)==2:
            plot_df[num_features[0]]=df[num_features[0]]
            plot_df[num_features[1]]=df[num_features[1]]

        if chart_visual=='Line Chart':
            try:
                st.line_chart(plot_df)
            except:
                st.text('Chart Could Not be Plot')

        if chart_visual=='Bar Chart':
            if numlevel==2:
                st.bar_chart(plot_df)
            elif numlevel==1:
                fig=sns.catplot(y=num_features[0],x=cat_features[0],data=plot_df, kind="bar")
                st.pyplot(fig)

        if chart_visual=='Scatter Plot':
            if numlevel==2:
                fig=sns.scatterplot(data=plot_df, x=num_features[0], y=num_features[1])
                st.pyplot(fig)
            elif numlevel==1:
                fig=sns.catplot(y=num_features[0],x=cat_features[0],data=plot_df)
                st.pyplot(fig)

        if chart_visual=='Box Plot':
            if numlevel==2:
                sns.boxplot(data=plot_df, x=num_features[0], y=num_features[1])
                st.pyplot()
            elif numlevel==1:
                sns.boxplot(y=num_features[0],x=cat_features[0],data=plot_df)
                st.pyplot()

