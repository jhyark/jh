import streamlit as st
import sqlite3
import pandas as pd
from datetime import datetime
import plotly.express as px
import pyupbit

def get_connection():
    return sqlite3.connect('trading_coin.db')

def load_data():
    conn = get_connection()    
    query = " SELECT * FROM decisions"    
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df

def cal_init_investment(df):
    init_btc_balance = df.iloc[0]['btc_balance']
    init_krw_balance = df.iloc[0]['krw_balance']
    init_btc_price = df.iloc[0]['btc_krw_price']
    init_total_inv = init_krw_balance + (init_btc_balance * init_btc_price)
    return init_total_inv

def cal_curr_investment(df):
    curr_btc_balance = df.iloc[-1]['btc_balance']
    curr_krw_balance = df.iloc[-1]['krw_balance']
    curr_btc_price = pyupbit.get_current_price("KRW-BTC")
    curr_total_inv = curr_krw_balance + (curr_btc_balance * curr_btc_price)                                               
    return curr_total_inv

def main():

    st.title("**  Bitcoin Trades Viewer  **")
    df = load_data()
    if df.empty:
        st.warning("No trade data available.")
        return

    init_investment = cal_init_investment(df)
    curr_investment = cal_curr_investment(df)

    profit_rate = (curr_investment - init_investment) / init_investment * 100
    st.header(f"Current Profit Rate: {profit_rate:.2f}%")

    st.header("Basic Statistics")
    st.write(f"Total number of trades :  {len(df)}")
    st.write(f"First trade date :  {df['timestamp'].min()}")
    st.write(f"Last trade date  :  {df['timestamp'].max()}")
                      
    st.header("Trade History")
    st.dataframe(df)

    st.header("Trade Decision Distribution")
    decision_count = df['decision'].value_counts()
    if not decision_count.empty:
        fig = px.pie(values=decision_count.values, names=decision_count.index,title='Trade Decisions')
        st.plotly_chart(fig) 
    else:
        st.write("No decision data available.")
        
    st.header("BTC Balance Over Time") # btc 잔액변화
    fig = px.line(df, x='timestamp', y='btc_balance', title='BTC balance')
    st.plotly_chart(fig) 

    st.header("KRW Balance Over Time") # krw 잔액변화
    fig = px.line(df, x='timestamp', y='krw_balance', title='KRW balance')
    st.plotly_chart(fig) 

    st.header("BTC average price over time") # btc 평균 매수가 변화
    fig = px.line(df, x='timestamp', y='btc_avg_buy_price', title='BTC average buy price')
    st.plotly_chart(fig) 
        
    st.header("BTC price over time") # btc 가격 변화
    fig = px.line(df, x='timestamp', y='btc_krw_price', title='BTC price(KRW)')
    st.plotly_chart(fig) 

if __name__ == '__main__':
    main()    
    