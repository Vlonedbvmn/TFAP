import streamlit as st
import pandas as pd

# Initialize session state variables
if 'clicked' not in st.session_state:
    st.session_state.clicked = False
if 'clicked2' not in st.session_state:
    st.session_state.clicked2 = False
if 'submitted' not in st.session_state:
    st.session_state.submitted = False
if 'df' not in st.session_state:
    st.session_state.df = None
if 'date' not in st.session_state:
    st.session_state.date = None
if 'target' not in st.session_state:
    st.session_state.target = None
if 'name' not in st.session_state:
    st.session_state.name = None




# Set Streamlit page config
st.set_page_config(
    page_title="Дані",
    layout="wide",
    initial_sidebar_state="auto"
)

st.markdown("""
    <style>
    .main-title {
        font-size: 40px;
        font-weight: bold;
        color: #f0f0f0;
        text-align: center;
        margin-bottom: 30px;
    }

    .button-container {
        display: flex;
        justify-content: center;
        gap: 20px;
        margin-bottom: 30px;
    }

    .button-container button {
        font-size: 18px;
        font-weight: bold;
        padding: 12px 24px;
        border-radius: 8px;
        background-color: #4CAF50;
        color: white;
        border: none;
        cursor: pointer;
    }

    .button-container button:hover {
        background-color: #45a049;
    }

    .section-title {
        font-size: 24px;
        font-weight: bold;
        color: #f0f0f0;
        text-align: center;
        margin-bottom: 10px;
    }

    .data-button-container {
        display: flex;
        justify-content: center;
        gap: 15px;
    }

    .data-button-container button {
        font-size: 16px;
        padding: 10px 20px;
        background-color: #333;
        color: #f0f0f0;
        border-radius: 5px;
        border: 1px solid #555;
    }

    .data-button-container button:hover {
        background-color: #444;
    }
    </style>
    """, unsafe_allow_html=True)

# Define button click functions
def click_button():
    st.session_state.clicked = True
    st.session_state.clicked2 = False
    st.session_state.submitted = False  # Reset submitted state on new button click

def click_button2():
    st.session_state.clicked = False
    st.session_state.clicked2 = True
    st.session_state.submitted = False  # Reset submitted state on new button click

def submit_data(dataframe, date_col, target_col, name):
    st.session_state.df = dataframe
    st.session_state.date = date_col
    st.session_state.target = target_col
    st.session_state.name = name
    st.session_state.submitted = True

# Main function
if __name__ == "__main__":
    
    with st.container():
        st.title("Оберіть з якими даними Ви бажаєте працювати")

    # Create two columns for buttons
    st.markdown('<div class="button-container">', unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)

    # Button for selecting experimental data
    with col1:
        st.button(label="Обрати тестувальні", on_click=click_button)
    st.markdown('</div>', unsafe_allow_html=True)
    # Button for selecting own data
    with col4:
        st.button(label="Обрати свої", on_click=click_button2)

    # If experimental data button is clicked, show additional options
    if st.session_state.clicked:
        st.markdown("### Ви обрали тестувальні дані. Це тестові набори даних, за допомогою яких ви можете ознайомитись з функціоналом проєкту та зрозуміти, яка модель підійде найкраще.")
        st.write("Оберіть тестовий набір даних:")
        st.markdown('<div class="data-button-container">', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1:
            t1 = st.button(label="Тестовий набір даних 1")
        with c2:
            t2 = st.button(label="Тестовий набір даних 2")
        with c3:
            t3 = st.button(label="Тестовий набір даних 3")
        st.markdown('</div>', unsafe_allow_html=True)
        # Test 1: Load sales.csv and allow submission
        if t1:
            dataframe = pd.read_csv("sales.csv")
            st.markdown(
                "### Тестовий набір даних 1 - це штучно-згенерозаний датасет, що представляє собою що-денну зміну акцій умовної компанії.")
            st.markdown("<a href='https://www.kaggle.com/datasets/sudipmanchare/simulated-sales-data-with-timeseries-features'>Посилання на датасет</a>")
            st.write(dataframe)
            st.button(label="Підтвердити", key="submit1", on_click=submit_data, args=(dataframe, "date", "target", "Тестовий набір даних 1"))

        # Test 2: Load AXISBANK-BSE.csv and allow submission
        if t2:
            dataframe = pd.read_csv("Weather_dataset.csv")
            st.markdown(
                "### Тест 2 - це показовий датасет, що являє собою часовий ряд по-годинної зміни температури в Німеччині.")
            st.write(dataframe)
            st.button(label="Підтвердити", key="submit2", on_click=submit_data, args=(dataframe, "date", "target", "Тестовий набір даних 2"))

        # Test 3: Load electricityConsumptionAndProduction.csv and allow submission
        if t3:
            dataframe = pd.read_csv("electricityConsumptionAndProductioction.csv")
            st.markdown(
                "### Тест 3 - це показовий датасет, що являє собою часовий ряд по-годинної зміни кількості попотребування електроенергії в Румунії.")
            st.write(dataframe)
            st.button(label="Підтвердити", key="submit3", on_click=submit_data, args=(dataframe, "date", "target", "Тестовий набір даних 3"))

    # If own data button is clicked, allow file upload
    if st.session_state.clicked2:
        st.markdown("### Ви обрали свої дані. Наразі основні вимоги до даних це:")
        st.markdown("### • Завжди повинні бути 2 колонки: час та значення, які будуть прогнозуватися")
        st.markdown("### • Бажано офрмити колонки з часом під формат timestamp, date, або datetime")
        st.markdown("### • Бажано, щоб було не було пропусків між записами значень, бо пропуски будуть заміщуватися, а отже якість прогнозування може падати")
        st.markdown("###  ")
        uploaded_file = st.file_uploader("Оберіть файл (Підтриуються формати .csv та .xlsx)", type=["csv", "xlsx"])
        if uploaded_file is not None:
            if uploaded_file.name[-4:] == "xlsx":
                dataframe = pd.read_excel(uploaded_file)
                st.write(dataframe)
            else:
                dataframe = pd.read_csv(uploaded_file)
                st.write(dataframe)

            option = st.selectbox(
            'Оберіть назву колонки з данними про дати',
            tuple(dataframe.columns.values))

            option2 = st.selectbox(
                'Оберіть назву колонки з данними про значення які ви хочете передбачити',
                tuple(dataframe.columns.values))


            st.button(label="Підтвердити", key="submit_own", on_click=submit_data, args=(dataframe, option, option2, uploaded_file.name))

    # After submission, show the dataframe and success message
    if st.session_state.submitted:
        st.success(f"Дані датасету {st.session_state.name} успішно завантажені! Тепер можете перейти до розділу 'Налаштування моделі'")

    st.divider()
    if st.session_state.name is not None:
        st.header(f"Наразі обраний датасет: {st.session_state.name}")
        with st.expander("Подивитися обраний датасет:"):
            st.write(st.session_state.df)
