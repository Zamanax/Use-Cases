import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression

from streamlit_utils import load_data, load_model, date_convert_to_timestamp

# Lancement et titre de l'application
st.title('Prédiction immobilière en Seine-et-Marne')

st.dataframe(["Maxime Zamani", "Jasmine Rodriguez", "Aela Le Sommer", "Lilas Thorel", "Nada Fakihani"], hide_index=True)

data = load_data()
model = load_model()

st.session_state['data'] = data
st.session_state['model'] = model

st.write('### Sélectionnez les caractéristiques du bien')
# Sélection des caractéristiques du bien
surface = st.slider('Surface (m²)', 0, 500, 100)
nb_pieces = st.slider('Nombre de pièces', 0, 10, 3)

# Sélection de la ville
ville = st.selectbox('Ville', data['new_commune'].unique())

# Sélection du type de bien
type_bien = st.selectbox('Type de bien', data['Type local'].unique())

date_debut_prediction = st.date_input('Début de la prédiction', value=pd.to_datetime("2022-01-01"), min_value=pd.to_datetime("2019-01-01"))

date_fin_prediction = st.date_input('Fin de la prédiction', value=pd.to_datetime("2023-07-01"))

# prédiction du prix
st.write('### Prédiction du prix')
# Création du dataframe
fh = pd.date_range(start=date_debut_prediction, end=date_fin_prediction, freq="M")

to_predictDf = pd.DataFrame({"new_commune": [ville]*len(fh), "Type local": [type_bien]*len(fh), "Surface Totale": [surface]*len(fh), "Nombre pieces principales": [nb_pieces]*len(fh), "Date mutation": fh})

# Prédictions
predictions = pd.Series(model.predict(to_predictDf), index=fh)
lm = LinearRegression()
lm.fit(predictions.index.map(date_convert_to_timestamp).to_numpy().reshape(-1, 1), predictions)
trend = pd.Series(lm.predict(predictions.index.map(date_convert_to_timestamp).to_numpy().reshape(-1, 1)), index=fh)

results = pd.DataFrame({"Prédiction": predictions, "Tendance": trend})
results = results.round(2)

# Affichage des prédictions
st.line_chart(results)

st.write("Croissance du prix par mois dans la commune: **" + str(round(lm.coef_[0], 5) * 100) + " %**")

# # Prévision avec Prophet
# st.write('### Prédiction avec Prophet')

# # Création du modèle
# prophet = Prophet()
# prophet.fit(results.reset_index().rename(columns={"index": "ds", "Prédiction": "y"})[["ds", "y"]])
# future = prophet.make_future_dataframe(periods=12, freq='M')[-13:]
# st.dataframe(future)
# forecast = prophet.predict(future)

# # Dataframe results["Prédiction"] + forecast["yhat"]
# future_prediction = pd.concat([results[["Prédiction"]].assign(type="Prédiction"), forecast.set_index("ds")[["yhat"]].rename(columns={"yhat": "Prédiction"}).assign(type="Prophet")])

# # Affichage des prédictions suivies des prévisions
# st.line_chart(future_prediction.reset_index(), x="index", y="Prédiction", color="type")

# st.dataframe(forecast)
# st.dataframe(future_prediction.reset_index())

