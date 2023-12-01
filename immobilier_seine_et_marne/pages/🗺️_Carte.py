import streamlit as st
import geopandas as gpd
import streamlit.components.v1 as components

from streamlit_utils import load_data

st.title("Carte des demandes de valeurs foncières en Seine-et-Marne")

if 'data' not in st.session_state:
    data = load_data()
else :
    data = st.session_state['data']

@st.cache_data
def load_map():
    map = gpd.read_file(
    "https://france-geojson.gregoiredavid.fr/repo/departements/77-seine-et-marne/communes-77-seine-et-marne.geojson")
    map["code"] = map["code"].astype("Int64")
    return map

map = load_map()

# Map of Seine-et-Marne by mean "Valeur fonciere" by "Code postal"

map = map.merge(data.groupby("new_commune")[
                  "Valeur fonciere"].mean().round(2).reset_index(), left_on="code", right_on="new_commune")

components.html(
    map.explore("Valeur fonciere", legend=False)._repr_html_(),
    height=600,
    width=600,
)