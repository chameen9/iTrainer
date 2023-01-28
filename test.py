import streamlit as st
import sqlite3
conn = sqlite3.connect('data.db',check_same_thread=False)
cur = conn.cursor()

def form():
    st.write('Title')
    with st.form(key='Information Form'):
        name = st.text_input('Enter your name')
        age = st.number_input('Enter your age')
        date = st.date_input('Enter Date of Birth')
        submit = st.form_submit_button('Submit')
        if submit == True:
            addData(name,age,date)

def addData(a,b,c):
    #cur.execute("""CREATE TABLE IF NOT EXISTS user(NAME TEXT(50),AGE TEXT(20),DOB TEXT(30));""")
    cur.execute("INSERT INTO User VALUES (?,?,?)",(a,b,c))
    conn.commit()
    conn.close()
    st.success('Submited')

form()