import streamlit as st
import pandas as pd
from plotly import express as px

st.title("EDA on Training Data 🚊")
st.caption("Exploratory Data Analysis on the Training Data")

train_df = pd.read_csv('data/raw/train.csv')

with st.expander('Raw 🥩 Train Data'):
    st.dataframe(train_df)

with st.expander('Summary 📊 Statistics'):
    st.write(train_df.describe())

with st.expander('Sales 📈 Over Time'):
    daily_sales = train_df.groupby('date')['sales'].sum()
    st.line_chart(daily_sales)

with st.expander('Sales 🏪 By Store'):
    store_sales = train_df.groupby('store_nbr')['sales'].sum()
    st.bar_chart(store_sales)

with st.expander('Sales 🪒 By Family'):
    item_sales = train_df.groupby('family')['sales'].sum()
    st.bar_chart(item_sales)