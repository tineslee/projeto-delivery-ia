import os
import streamlit as st
import pandas as pd

# Importa o módulo de recomendações
from recommendations import build_recommender, recommend_for_user  # o arquivo deve ser recommendations.py dentro de src/

st.set_page_config(page_title="IA para Delivery – Recomendações", layout="wide")
st.title("🍽️ Recomendações personalizadas (inspirado no iFood)")

DATA_PATH = os.path.join("data", "historico_pedidos.csv")

@st.cache_data
def load_components():
    df, user_item, user_sim = build_recommender(DATA_PATH)
    return df, user_item, user_sim

df, user_item, user_sim = load_components()

st.sidebar.header("Configurações")
min_id = int(df["cliente_id"].min())
max_id = int(df["cliente_id"].max())
cliente_id = st.sidebar.number_input("Cliente ID", min_value=min_id, max_value=max_id, value=min_id, step=1)
top_n = st.sidebar.slider("Quantidade de recomendações", 1, 10, 5)

if st.button("Gerar recomendações"):
    try:
        recs_df = recommend_for_user(df, user_item, user_sim, cliente_id=int(cliente_id), top_n=top_n)
        st.subheader(f"Recomendações para cliente {int(cliente_id)}")
        st.write(recs_df[["nome", "categoria", "score"]])

        st.subheader("Categorias sugeridas")
        st.bar_chart(recs_df.groupby("categoria")["score"].mean())
    except Exception as e:
        st.error(f"Erro ao gerar recomendações: {e}")
        st.info("Verifique se o cliente existe e se os dados foram gerados corretamente.")
