import streamlit as st
import pandas as pd
from neuralforecast.models import KAN, TimeLLM, TimesNet, NBEATSx, TimeMixer, PatchTST
from neuralforecast import NeuralForecast
import plotly.figure_factory as ff
import plotly.express as px
from sklearn.metrics import mean_squared_error

st.set_page_config(
    page_title="Модель",
    layout="wide",
    initial_sidebar_state="auto"
)

means = {"Місяць": "M",
"Година": "h",
"Рік": "Y",
"Хвилина": "T",
"Секунда": "S",
"День": "D",
}

if 'fig' not in st.session_state:
    st.session_state.fig = None
if 'mse' not in st.session_state:
    st.session_state.mse = None
if 'inst_name' not in st.session_state:
    st.session_state.inst_name = None
if 'model_forecast' not in st.session_state:
    st.session_state.model_forecast = None
if 'df_forpred' not in st.session_state:
    st.session_state.df_forpred = None
if 'horiz' not in st.session_state:
    st.session_state.horiz = None
if 'date_not_n' not in st.session_state:
    st.session_state.date_not_n = None
# @st.cache_data(show_spinner="Loading data...")
# def load_data():
#     wine_data = load_wine()
#     wine_df = pd.DataFrame(data=wine_data.data, columns=wine_data.feature_names)
#
#     wine_df['target'] = wine_data.target
#
#     return wine_df
#
# wine_df = load_data()
#
# @st.cache_data
# def split_data(df):
#
#     X = df.drop(['target'], axis=1)
#     y = df['target']
#
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=True, stratify=y, random_state=42)
#
#     return X_train, X_test, y_train, y_test
#
# X_train, X_test, y_train, y_test = split_data(wine_df)
#
# @st.cache_data()
# def select_features(X_train, y_train, X_test, k):
#     selector = SelectKBest(mutual_info_classif, k=k)
#     selector.fit(X_train, y_train)
#
#     sel_X_train = selector.transform(X_train)
#     sel_X_test = selector.transform(X_test)
#
#     return sel_X_train, sel_X_test
#
# @st.cache_data(show_spinner="Training and evaluating model")
# def fit_and_score(model, k):
#
#     if model == "Baseline":
#         clf = DummyClassifier(strategy="stratified", random_state=42)
#     elif model == "Decision Tree":
#         clf = DecisionTreeClassifier(random_state=42)
#     elif model == "Random Forest":
#         clf = RandomForestClassifier(random_state=42)
#     else:
#         clf = GradientBoostingClassifier(random_state=42)
#
#     sel_X_train, sel_X_test = select_features(X_train, y_train, X_test, k)
#
#     clf.fit(sel_X_train, y_train)
#
#     preds = clf.predict(sel_X_test)
#
#     score = round(f1_score(y_test, preds, average='weighted'),3)
#
#     return score
#
# def save_performance(model, k):
#
#     score = fit_and_score(model, k)
#
#     st.session_state['model'].append(model)
#     st.session_state['num_features'].append(k)
#     st.session_state['score'].append(score)

@st.cache_data(show_spinner="Робимо передбачення...")
def submit_data_auto(datafra, iter, horizon, rarety):
    try:
        st.session_state.horiz = horizon
        q = int(round(len(datafra) * 0.01, 0))
        fcst = NeuralForecast(
            models=[
                KAN(h=horizon,
                    input_size=horizon*q,
                    # output_size=horizon,
                    max_steps=iter,
                    scaler_type='standard',
                    start_padding_enabled=True
                    ),
                TimesNet(h=horizon,
                    input_size=horizon * q,
                    # output_size=horizon,
                    max_steps=iter,
                    scaler_type='standard',
                    start_padding_enabled=True
                    ),
                TimeMixer(h=horizon,
                    input_size=horizon * q,
                    # output_size=horizon,
                    max_steps=iter,
                    scaler_type='standard',
                    # start_padding_enabled=True,
                    n_series=1
                    ),
                PatchTST(h=horizon,
                    input_size=horizon * q,
                    # output_size=horizon,
                    max_steps=iter,
                    scaler_type='standard',
                    start_padding_enabled=True
                    ),
                NBEATSx(h=horizon,
                    input_size=horizon * q,
                    # output_size=horizon,
                    max_steps=iter,
                    scaler_type='standard',
                    start_padding_enabled=True
                    ),

            ],
            freq=rarety
        )

        Y_train_df = datafra[:-horizon]
        Y_test_df = datafra[-horizon:]
        fcst.fit(df=Y_train_df)
        forecasts = fcst.predict(futr_df=Y_test_df)
        print(forecasts)
        results = {}
        for i in ["KAN", "TimesNet", "TimeMixer", "PatchTST", "NBEATSx"]:
            results[i] = mean_squared_error(Y_test_df["y"], forecasts[i])

        key_with_min_value = min(results, key=results.get)

        if key_with_min_value == "KAN":
            fcst = NeuralForecast(
                models=[
                    KAN(h=horizon,
                        input_size=horizon * q,
                        # output_size=horizon,
                        max_steps=iter,
                        scaler_type='standard',
                        start_padding_enabled=True
                        )

                ],
                freq=rarety
            )

            Y_train_df = datafra[:-horizon]
            Y_test_df = datafra[-horizon:]
            fcst.fit(df=Y_train_df)
            forecasts = fcst.predict(futr_df=Y_test_df)
            st.session_state.mse = results["KAN"]
            st.session_state.inst_name = "KAN"
            st.session_state.model_forecast = fcst
            print(f'Mean Squared Error: {st.session_state.mse}')
            print(len(forecasts["KAN"]), len(Y_test_df["y"]))
            print(forecasts.columns)
            forecasts = forecasts.reset_index(drop=True)
            print(forecasts.columns.values)
            print(forecasts["KAN"].values.tolist())
            dpred = pd.DataFrame()
            dpred["real"] = Y_test_df["y"]
            dpred["pred"] = forecasts["KAN"].values.tolist()
            dpred["unique_id"] = [i for i in range(1, len(dpred) + 1)]
            # Create distplot with custom bin_size
            st.session_state.fig = px.line(dpred, x='unique_id', y=['real', 'pred'],
                                           labels={'value': 'Y values', 'x': 'X values'})
            print(dpred)
        if key_with_min_value == "TimesNet":
            fcst = NeuralForecast(
                models=[
                    TimesNet(h=horizon,
                             input_size=horizon * q,
                             # output_size=horizon,
                             max_steps=iter,
                             scaler_type='standard',
                             start_padding_enabled=True
                             ),

                ],
                freq=rarety
            )

            Y_train_df = datafra[:-horizon]
            Y_test_df = datafra[-horizon:]
            fcst.fit(df=Y_train_df)
            forecasts = fcst.predict(futr_df=Y_test_df)
            st.session_state.mse = results["TimesNet"]
            st.session_state.inst_name = "TimesNet"
            st.session_state.model_forecast = fcst
            print(f'Mean Squared Error: {st.session_state.mse}')
            print(len(forecasts["TimesNet"]), len(Y_test_df["y"]))
            print(forecasts.columns)
            forecasts = forecasts.reset_index(drop=True)
            print(forecasts.columns.values)
            print(forecasts["TimesNet"].values.tolist())
            dpred = pd.DataFrame()
            dpred["real"] = Y_test_df["y"]
            dpred["pred"] = forecasts["TimesNet"].values.tolist()
            dpred["unique_id"] = [i for i in range(1, len(dpred) + 1)]
            # Create distplot with custom bin_size
            st.session_state.fig = px.line(dpred, x='unique_id', y=['real', 'pred'],
                                           labels={'value': 'Y values', 'x': 'X values'})
            print(dpred)
        if key_with_min_value == "TimeMixer":
            fcst = NeuralForecast(
                models=[
                    TimeMixer(h=horizon,
                              input_size=horizon * q,
                              # output_size=horizon,
                              max_steps=iter,
                              scaler_type='standard',
                              # start_padding_enabled=True,
                              n_series=1
                              )

                ],
                freq=rarety
            )

            Y_train_df = datafra[:-horizon]
            Y_test_df = datafra[-horizon:]
            fcst.fit(df=Y_train_df)
            forecasts = fcst.predict(futr_df=Y_test_df)
            st.session_state.mse = results["TimeMixer"]
            st.session_state.inst_name = "TimeMixer"
            st.session_state.model_forecast = fcst
            print(f'Mean Squared Error: {st.session_state.mse}')
            print(len(forecasts["TimeMixer"]), len(Y_test_df["y"]))
            print(forecasts.columns)
            forecasts = forecasts.reset_index(drop=True)
            print(forecasts.columns.values)
            print(forecasts["TimeMixer"].values.tolist())
            dpred = pd.DataFrame()
            dpred["real"] = Y_test_df["y"]
            dpred["pred"] = forecasts["TimeMixer"].values.tolist()
            dpred["unique_id"] = [i for i in range(1, len(dpred) + 1)]
            # Create distplot with custom bin_size
            st.session_state.fig = px.line(dpred, x='unique_id', y=['real', 'pred'],
                                           labels={'value': 'Y values', 'x': 'X values'})
            print(dpred)
        if key_with_min_value == "PatchTST":
            fcst = NeuralForecast(
                models=[
                    PatchTST(h=horizon,
                             input_size=horizon * q,
                             # output_size=horizon,
                             max_steps=iter,
                             scaler_type='standard',
                             start_padding_enabled=True
                             )

                ],
                freq=rarety
            )

            Y_train_df = datafra[:-horizon]
            Y_test_df = datafra[-horizon:]
            fcst.fit(df=Y_train_df)
            forecasts = fcst.predict(futr_df=Y_test_df)
            st.session_state.mse = results["PatchTST"]
            st.session_state.inst_name = "PatchTST"
            st.session_state.model_forecast = fcst
            print(f'Mean Squared Error: {st.session_state.mse}')
            print(len(forecasts["PatchTST"]), len(Y_test_df["y"]))
            print(forecasts.columns)
            forecasts = forecasts.reset_index(drop=True)
            print(forecasts.columns.values)
            print(forecasts["PatchTST"].values.tolist())
            dpred = pd.DataFrame()
            dpred["real"] = Y_test_df["y"]
            dpred["pred"] = forecasts["PatchTST"].values.tolist()
            dpred["unique_id"] = [i for i in range(1, len(dpred) + 1)]
            # Create distplot with custom bin_size
            st.session_state.fig = px.line(dpred, x='unique_id', y=['real', 'pred'],
                                           labels={'value': 'Y values', 'x': 'X values'})
            print(dpred)

        if key_with_min_value == "NBEATSx":
            fcst = NeuralForecast(
                models=[
                    NBEATSx(h=horizon,
                            input_size=horizon * q,
                            # output_size=horizon,
                            max_steps=iter,
                            scaler_type='standard',
                            start_padding_enabled=True
                            ),

                ],
                freq=rarety
            )

            Y_train_df = datafra[:-horizon]
            Y_test_df = datafra[-horizon:]
            fcst.fit(df=Y_train_df)
            forecasts = fcst.predict(futr_df=Y_test_df)
            st.session_state.inst_name = "NBEATSx"
            st.session_state.mse = results["NBEATSx"]
            st.session_state.model_forecast = fcst
            print(f'Mean Squared Error: {st.session_state.mse}')
            print(len(forecasts["NBEATSx"]), len(Y_test_df["y"]))
            print(forecasts.columns)
            forecasts = forecasts.reset_index(drop=True)
            print(forecasts.columns.values)
            print(forecasts["NBEATSx"].values.tolist())
            dpred = pd.DataFrame()
            dpred["real"] = Y_test_df["y"]
            dpred["pred"] = forecasts["NBEATSx"].values.tolist()
            dpred["unique_id"] = [i for i in range(1, len(dpred) + 1)]
            # Create distplot with custom bin_size
            st.session_state.fig = px.line(dpred, x='unique_id', y=['real', 'pred'], labels={'value': 'Y values', 'x': 'X values'})
            print(dpred)
    except:
        st.warning('Надано не коректні гіперпараметри', icon="⚠️")

    # st.session_state.mse = mean_squared_error(Y_test_df["y"], forecasts["KAN"])
    # st.session_state.model_forecast = fcst
    # print(f'Mean Squared Error: {st.session_state.mse}')
    # print(len(forecasts["KAN"]), len(Y_test_df["y"]))
    # print(forecasts.columns)
    # forecasts = forecasts.reset_index(drop=True)
    # print(forecasts.columns.values)
    # print(forecasts["KAN"].values.tolist())
    # dpred = pd.DataFrame()
    # dpred["real"] = Y_test_df["y"]
    # dpred["pred"] = forecasts["KAN"].values.tolist()
    # dpred["unique_id"] = [i for i in range(1, len(dpred) + 1)]
    # # Create distplot with custom bin_size
    # st.session_state.fig = px.line(dpred, x='unique_id', y=['real', 'pred'], labels={'value': 'Y values', 'x': 'X values'})
    # print(dpred)


@st.cache_data(show_spinner="Робимо передбачення...")
def submit_data_KAN(datafra, iter, horizon, rarety, inp):
    if st.session_state.date_not_n:
        start_date = pd.to_datetime('2024-01-01')
        datafra['ds'] = start_date + pd.to_timedelta(datafra['ds'] - 1, rarety)

    datafra = datafra.set_index('ds').asfreq(rarety)    
    datafra = datafra.reset_index()
    print("s;kgfoshdisdifsdf")
    print(datafra)
        
    try:
        st.session_state.horiz = horizon
        q = int(round(len(datafra) * 0.01, 0))
        fcst = NeuralForecast(
            models=[
                KAN(h=horizon,
                    input_size=int(round(inp, 0)),
                    # output_size=horizon,
                    max_steps=iter,
                    scaler_type='standard',
                    start_padding_enabled=True
                    ),
            ],
            freq=rarety
        )

        Y_train_df = datafra[:-horizon]
        Y_test_df = datafra[-horizon:]
        fcst.fit(df=Y_train_df)
        forecasts = fcst.predict(futr_df=Y_test_df)
        st.session_state.mse = mean_squared_error(Y_test_df["y"], forecasts["KAN"])
        st.session_state.inst_name = "KAN"
        st.session_state.model_forecast = fcst
        print(f'Mean Squared Error: {st.session_state.mse}')
        print(len(forecasts["KAN"]), len(Y_test_df["y"]))
        print(forecasts.columns)
        forecasts = forecasts.reset_index(drop=True)
        print(forecasts.columns.values)
        print(forecasts["KAN"].values.tolist())
        dpred = pd.DataFrame()
        dpred["real"] = Y_test_df["y"]
        dpred["pred"] = forecasts["KAN"].values.tolist()
        dpred["unique_id"] = [i for i in range(1, len(dpred) + 1)]
        # Create distplot with custom bin_size
        st.session_state.fig = px.line(dpred, x='unique_id', y=['real', 'pred'], labels={'value': 'Y values', 'x': 'X values'})
        print(dpred)
    except Exception as ex:
        print(ex)
        st.warning('Надано не коректні гіперпараметри', icon="⚠️")
@st.cache_data(show_spinner="Робимо передбачення...")
def submit_data_TN(datafra, iter, horizon, rarety, inp):
    try:
        st.session_state.horiz = horizon
        q = int(round(len(datafra) * 0.1, 0))
        fcst = NeuralForecast(
            models=[
                TimesNet(h=horizon,
                    input_size=int(round(inp, 0)),
                    # output_size=horizon,
                    max_steps=iter,
                    scaler_type='standard',
                    start_padding_enabled=True
                    ),
            ],
            freq=rarety
        )

        Y_train_df = datafra[:-horizon]
        Y_test_df = datafra[-horizon:]
        fcst.fit(df=Y_train_df)
        forecasts = fcst.predict(futr_df=Y_test_df)
        st.session_state.mse = mean_squared_error(Y_test_df["y"], forecasts["TimesNet"])
        st.session_state.inst_name = "TimesNet"
        st.session_state.model_forecast = fcst
        print(f'Mean Squared Error: {st.session_state.mse}')
        print(len(forecasts["TimesNet"]), len(Y_test_df["y"]))
        print(forecasts.columns)
        forecasts = forecasts.reset_index(drop=True)
        print(forecasts.columns.values)
        print(forecasts["TimesNet"].values.tolist())
        dpred = pd.DataFrame()
        dpred["real"] = Y_test_df["y"]
        dpred["pred"] = forecasts["TimesNet"].values.tolist()
        dpred["unique_id"] = [i for i in range(1, len(dpred) + 1)]
        # Create distplot with custom bin_size
        st.session_state.fig = px.line(dpred, x='unique_id', y=['real', 'pred'], labels={'value': 'Y values', 'x': 'X values'})
        print(dpred)
    except:
        st.warning('Надано не коректні гіперпараметри', icon="⚠️")
@st.cache_data(show_spinner="Робимо передбачення...")
def submit_data_TM(datafra, iter, horizon, rarety, inp):
    try:
        st.session_state.horiz = horizon
        q = int(round(len(datafra) * 0.1, 0))
        fcst = NeuralForecast(
            models=[
                TimeMixer(h=horizon,
                    input_size=int(round(inp, 0)),
                    # output_size=horizon,
                    max_steps=iter,
                    scaler_type='standard',
                    start_padding_enabled=True,
                    n_series=1
                    ),
            ],
            freq=rarety
        )

        Y_train_df = datafra[:-horizon]
        Y_test_df = datafra[-horizon:]
        fcst.fit(df=Y_train_df)
        forecasts = fcst.predict(futr_df=Y_test_df)
        st.session_state.mse = mean_squared_error(Y_test_df["y"], forecasts["TimeMixer"])
        st.session_state.inst_name = "TimeMixer"
        st.session_state.model_forecast = fcst
        print(f'Mean Squared Error: {st.session_state.mse}')
        print(len(forecasts["TimeMixer"]), len(Y_test_df["y"]))
        print(forecasts.columns)
        forecasts = forecasts.reset_index(drop=True)
        print(forecasts.columns.values)
        print(forecasts["TimeMixer"].values.tolist())
        dpred = pd.DataFrame()
        dpred["real"] = Y_test_df["y"]
        dpred["pred"] = forecasts["TimeMixer"].values.tolist()
        dpred["unique_id"] = [i for i in range(1, len(dpred) + 1)]
        # Create distplot with custom bin_size
        st.session_state.fig = px.line(dpred, x='unique_id', y=['real', 'pred'], labels={'value': 'Y values', 'x': 'X values'})
        print(dpred)
    except:
        st.warning('Надано не коректні гіперпараметри', icon="⚠️")
@st.cache_data(show_spinner="Робимо передбачення...")
def submit_data_PTST(datafra, iter, horizon, rarety, inp):
    try:
        st.session_state.horiz = horizon
        q = int(round(len(datafra) * 0.1, 0))
        fcst = NeuralForecast(
            models=[
                PatchTST(h=horizon,
                    input_size=int(round(inp, 0)),
                    # output_size=horizon,
                    max_steps=iter,
                    scaler_type='standard',
                    start_padding_enabled=True
                    ),
            ],
            freq=rarety
        )

        Y_train_df = datafra[:-horizon]
        Y_test_df = datafra[-horizon:]
        fcst.fit(df=Y_train_df)
        forecasts = fcst.predict(futr_df=Y_test_df)
        st.session_state.mse = mean_squared_error(Y_test_df["y"], forecasts["PatchTST"])
        st.session_state.inst_name = "PatchTST"
        st.session_state.model_forecast = fcst
        print(f'Mean Squared Error: {st.session_state.mse}')
        print(len(forecasts["PatchTST"]), len(Y_test_df["y"]))
        print(forecasts.columns)
        forecasts = forecasts.reset_index(drop=True)
        print(forecasts.columns.values)
        print(forecasts["PatchTST"].values.tolist())
        dpred = pd.DataFrame()
        dpred["real"] = Y_test_df["y"]
        dpred["pred"] = forecasts["PatchTST"].values.tolist()
        dpred["unique_id"] = [i for i in range(1, len(dpred) + 1)]
        # Create distplot with custom bin_size
        st.session_state.fig = px.line(dpred, x='unique_id', y=['real', 'pred'], labels={'value': 'Y values', 'x': 'X values'})
        print(dpred)
    except:
        st.warning('Надано не коректні гіперпараметри', icon="⚠️")
@st.cache_data(show_spinner="Робимо передбачення...")
def submit_data_NBx(datafra, iter, horizon, rarety, inp):
    try:
        st.session_state.horiz = horizon
        q = int(round(len(datafra) * 0.1, 0))
        fcst = NeuralForecast(
            models=[
                NBEATSx(h=horizon,
                    input_size=int(round(inp, 0)),
                    # output_size=horizon,
                    max_steps=iter,
                    scaler_type='standard',
                    start_padding_enabled=True
                    ),
            ],
            freq=rarety
        )

        Y_train_df = datafra[:-horizon]
        Y_test_df = datafra[-horizon:]
        fcst.fit(df=Y_train_df)
        forecasts = fcst.predict(futr_df=Y_test_df)
        st.session_state.mse = mean_squared_error(Y_test_df["y"], forecasts["NBEATSx"])
        st.session_state.inst_name = "NBEATSx"
        st.session_state.model_forecast = fcst
        print(f'Mean Squared Error: {st.session_state.mse}')
        print(len(forecasts["NBEATSx"]), len(Y_test_df["y"]))
        print(forecasts.columns)
        forecasts = forecasts.reset_index(drop=True)
        print(forecasts.columns.values)
        print(forecasts["NBEATSx"].values.tolist())
        dpred = pd.DataFrame()
        dpred["real"] = Y_test_df["y"]
        dpred["pred"] = forecasts["NBEATSx"].values.tolist()
        dpred["unique_id"] = [i for i in range(1, len(dpred) + 1)]
        # Create distplot with custom bin_size
        st.session_state.fig = px.line(dpred, x='unique_id', y=['real', 'pred'], labels={'value': 'Y values', 'x': 'X values'})
        print(dpred)
    except:
        st.warning('Надано не коректні гіперпараметри', icon="⚠️")


if __name__ == "__main__":

    if st.session_state.df is not None:
        ds_for_pred = pd.DataFrame()
        ds_for_pred["y"] = st.session_state.df[st.session_state.target]
        try:
            ds_for_pred["ds"] = st.session_state.df[st.session_state.date]
            st.session_state.date_not_n = False
        except:
            st.session_state.date_not_n = True
            ds_for_pred['ds'] = [i for i in range(1, len(ds_for_pred)+1)]

        ds_for_pred["unique_id"] = [0 for i in range(1, len(ds_for_pred)+1)]

        print(ds_for_pred)
        st.session_state.df_forpred = ds_for_pred
        with st.container():
            st.title("🧪 Обрання та налашутвання моделі прогнозування")


        model = st.selectbox("Оберіть модель для передбачення", ["KAN", "TimesNet", "NBEATSx", "TimeMixer", "PatchTST", "Авто-вибір"])

        try:
            if model == "KAN":
                st.markdown("## Ви обрали модель KAN")
                st.markdown("### KAN — це нейронна мережа, що застосовує апроксимаційну теорему Колмогорова-Арнольда, яка стверджує, що сплайни можуть апроксимувати складніші функції.")
                st.divider()
                horizon = st.select_slider(
                    "Оберіть горизонт передбачення:",
                    options=[i for i in range(1, int(round(len(ds_for_pred)*0.1, 0)))]
                )
                iter = st.select_slider(
                    "Оберіть к-ть ітерацій начання моделі:",
                    options=[i for i in range(5,101)]
                )
                fr = st.selectbox("Оберіть частоту запису даних в ряді:",
                                     ["Місяць", "День", "Рік", "Хвилина", "Секунда", "Година"])

                inp = st.number_input("Оберіть к-ть попередніх значень з ряду для кроку прогнозу:")

                st.button(label="Підтвердити", key="kan", on_click=submit_data_KAN,
                          args=(ds_for_pred, iter, horizon, means[fr], inp))
        except Exception as ex:
            print(ex)
            st.warning('Надано не коректні гіперпараметри', icon="⚠️")

        try:
            if model == "NBEATSx":
                st.markdown("## Ви обрали модель NBEATSx")
                st.markdown(
                    "### NBEATSx — це глибока нейронна архітектура на основі MLP, яка використовує прямі та зворотні залишкові зв'язки.")
                st.divider()
                horizon = st.select_slider(
                    "Оберіть горизонт передбачення:",
                    options=[i for i in range(1, int(round(len(ds_for_pred)*0.1, 0)))]
                )
                iter = st.select_slider(
                    "Оберіть к-ть ітерацій начання моделі:",
                    options=[i for i in range(5, 101)]
                )
                fr = st.selectbox("Оберіть частоту запису даних в ряді:",
                                  ["Місяць", "День", "Рік", "Хвилина", "Секунда", "Година"])

                inp = st.number_input("Оберіть к-ть попередніх значень з ряду для кроку прогнозу:")
                st.button(label="Підтвердити", key="kan", on_click=submit_data_NBx,
                          args=(ds_for_pred, iter, horizon, means[fr], inp))
        except:
            st.warning('Надано не коректні гіперпараметри', icon="⚠️")
        try:
            if model == "TimesNet":
                st.markdown("## Ви обрали модель TimesNet")

                st.markdown("### TimesNet — це модель на основі CNN, яка ефективно вирішує завдання моделювання як внутрішньоперіодних, так і міжперіодних змін у часових рядах.")
                st.divider()
                horizon = st.select_slider(
                    "Оберіть горизонт передбачення:",
                    options=[i for i in range(1, int(round(len(ds_for_pred)*0.1, 0)))]
                )
                iter = st.select_slider(
                    "Оберіть к-ть ітерацій начання моделі:",
                    options=[i for i in range(5, 101)]
                )
                fr = st.selectbox("Оберіть частоту запису даних в ряді:",
                                  ["Місяць", "День", "Рік", "Хвилина", "Секунда", "Година"])
                inp = st.number_input("Оберіть к-ть попередніх значень з ряду для кроку прогнозу:")
                st.button(label="Підтвердити", key="kan", on_click=submit_data_TN,
                          args=(ds_for_pred, iter, horizon, means[fr], inp))
        except:
            st.warning('Надано не коректні гіперпараметри', icon="⚠️")
        try:
            if model == "TimeMixer":
                st.markdown("## Ви обрали модель TimeMixer")
                st.markdown(
                    "### TimeMixer - модель, яка поєднує елементи архітектури Transformers і CNN для досягнення високої точності в прогнозах, обробляючи залежності як в просторі, так і в часі.")
                st.divider()
                horizon = st.select_slider(
                    "Оберіть горизонт передбачення:",
                    options=[i for i in range(1, int(round(len(ds_for_pred)*0.1, 0)))]
                )
                iter = st.select_slider(
                    "Оберіть к-ть ітерацій начання моделі:",
                    options=[i for i in range(5, 101)]
                )
                fr = st.selectbox("Оберіть частоту запису даних в ряді:",
                                  ["Місяць", "День", "Рік", "Хвилина", "Секунда", "Година"])
                inp = st.number_input("Оберіть к-ть попередніх значень з ряду для кроку прогнозу:")
                st.button(label="Підтвердити", key="kan", on_click=submit_data_TM,
                          args=(ds_for_pred, iter, horizon, means[fr], inp))
        except:
            st.warning('Надано не коректні гіперпараметри', icon="⚠️")
        try:
            if model == "PatchTST":
                st.markdown("## Ви обрали модель PatchTST")
                st.markdown(
                    "### PatchTST — це високоефективна модель на основі Transformer, призначена для багатовимірного прогнозування часових рядів.")
                st.divider()
                horizon = st.select_slider(
                    "Оберіть горизонт передбачення:",
                    options=[i for i in range(1, int(round(len(ds_for_pred)*0.1, 0)))]
                )
                iter = st.select_slider(
                    "Оберіть к-ть ітерацій начання моделі:",
                    options=[i for i in range(5, 101)]
                )
                fr = st.selectbox("Оберіть частоту запису даних в ряді:",
                                  ["Місяць", "День", "Рік", "Хвилина", "Секунда", "Година"])
                inp = st.number_input("Оберіть к-ть попередніх значень з ряду для кроку прогнозу:")
                st.button(label="Підтвердити", key="kan", on_click=submit_data_PTST,
                          args=(ds_for_pred, iter, horizon, means[fr], inp))
        except:
            st.warning('Надано не коректні гіперпараметри', icon="⚠️")
        try:
            if model == "Авто-вибір":
                st.divider()
                horizon = st.select_slider(
                    "Оберіть число кількості часових величин, яке найблище до числа часових величин на які ви хочете робити передбачення:",
                    options=[i for i in range(1, int(round(len(ds_for_pred)*0.1, 0)))]
                )
                iter = st.select_slider(
                    "Оберіть к-ть ітерацій начання моделі (чим більше, тим довше):",
                    options=[i for i in range(5, 101)]
                )
                fr = st.selectbox("Оберіть частоту запису даних в ряді:",
                                  ["Місяць", "День", "Рік", "Хвилина", "Секунда", "Година"])
                st.button(label="Підтвердити", key="kan", on_click=submit_data_auto,
                          args=(ds_for_pred, iter, horizon, means[fr]))
        except:
            st.warning('Надано не коректні гіперпараметри', icon="⚠️")
        st.divider()
        if st.session_state.fig is not None:
            if st.session_state.inst_name != "Авто-вибір":
                st.markdown(f"## Середньоквадратичне відхилення обраної моделі ({st.session_state.inst_name}): {round(st.session_state.mse, 3)}")
                st.plotly_chart(st.session_state.fig, use_container_width=True)
            else:
                st.markdown(
                    f"## Середньоквадратичне відхилення обраної моделі авто-вибором ({st.session_state.inst_name}): {round(st.session_state.mse, 3)}")
                st.plotly_chart(st.session_state.fig, use_container_width=True)

        # st.button("Train", type="primary", on_click=save_performance, args=((model, k)))
        #
        # with st.expander("See full dataset"):
        #     st.write(wine_df)
        #
        # if len(st.session_state['score']) != 0:
        #     st.subheader(f"The model has an F1-Score of: {st.session_state['score'][-1]}")
    else:
        st.warning('Для початки роботи з моделями, оберіть дані', icon="⚠️")





