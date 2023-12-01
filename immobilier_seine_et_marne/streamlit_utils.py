
# Fonctions nécessaires au modèle
import pickle
from sklearn.pipeline import FunctionTransformer
import streamlit as st
import os
import pandas as pd

def date_convert_to_timestamp(X):
    return X.values.astype(int) // 10 ** 9

date_transformer = FunctionTransformer(date_convert_to_timestamp)

@st.cache_resource
def load_model():
    # Chargement du modèle
    try:
        model = pickle.load(open('immobilier_seine_et_marne/long_training_model.pkl', 'rb'))
        # st.info('Modèle chargé')
        return model
    except FileNotFoundError:
        # Si le modèle n'existe pas, on le crée
        st.error('Le modèle n\'a pas été trouvé')
        os.system("jupyter notebook --execute immobilier_seine_et_marne/preprocess_creation_modele_habitation.ipynb")
        st.info('Le modèle a été créé. Relancez l\'application.')

@st.cache_data
def load_data():
    # Chargement des données
    try:
        data = pd.read_csv('immobilier_seine_et_marne/data/seine_et_marne.csv')
        # st.info('Données chargées')
        return data
    except FileNotFoundError:
        # Si les données n'existent pas, on les crée
        st.error('Les données n\'ont pas été trouvées')
        os.system("jupyter notebook --execute immobilier_seine_et_marne/preprocess_creation_modele_habitation.ipynb")
        st.info('Les données ont été créées. Relancez l\'application.')