import streamlit as st
import pandas as pd


# Directory of the pages


# Load available modules from the pages directory



st.set_page_config(
    page_title="TFAP",
    layout="wide",
    initial_sidebar_state="auto"
)



if all(key not in st.session_state.keys() for key in ('model', 'num_features', 'score')):
    st.session_state['num_features'] = []
    st.session_state['score'] = []
    st.session_state.clicked = False
    st.session_state.clicked2 = False
    st.session_state.df = None
    st.session_state.date = ""
    st.session_state.target = ""

def display_df():
    df = pd.DataFrame({"Model": st.session_state['model'],
                       "Number of features": st.session_state['num_features'],
                       "F1-Score": st.session_state['score']})
    
    sorted_df = df.sort_values(by=['F1-Score'], ascending=False).reset_index(drop=True)

    st.write(sorted_df)




if __name__ == "__main__":

    st.title("TFAP - Timeseries Forecasting AI playground")

    st.subheader(
        "Вітаю, це сторінка проєкту, який є програмним застосунком для експерементаторів у сфері прогнозування часових рядів використовуючи методи машинного навчання. Щоб почати експерементувати натисніть розділ Data, щоб програма отримала данні з якими ви хочете працювати)")

    st.page_link("pages/Дані.py", label="Data")

