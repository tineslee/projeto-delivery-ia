import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def load_data(path="data/historico_pedidos.csv"):
    df = pd.read_csv(path)
    return df

def build_user_item_matrix(df):
    """
    Retorna uma matriz (clientes x restaurantes) com ratings médios.
    """
    pivot = df.pivot_table(
        index="cliente_id",
        columns="restaurante_id",
        values="rating",
        aggfunc="mean"
    ).fillna(0.0)
    return pivot

def compute_user_similarity(user_item):
    """
    Similaridade de cosseno entre clientes.
    """
    sim = cosine_similarity(user_item.values)
    sim_df = pd.DataFrame(sim, index=user_item.index, columns=user_item.index)
    return sim_df

def recommend_for_user(df, user_item, user_sim, cliente_id, top_n=5):
    """
    Recomenda restaurantes não avaliados pelo cliente com base em:
    - média ponderada pelos vizinhos mais similares (colaborativa)
    - reforço por preferência de categoria do cliente (conteúdo)
    Retorna um DataFrame com restaurante_id, nome, categoria e score.
    """
    if cliente_id not in user_item.index:
        raise ValueError("cliente_id não encontrado nos dados")

    # Restaurantes já avaliados pelo cliente
    rated_mask = user_item.loc[cliente_id].values > 0
    rated_ids = set(np.where(rated_mask)[0] + 1)  # +1 pois restaurante_id começa em 1

    all_restaurants = set(user_item.columns)
    not_rated = sorted(list(all_restaurants - rated_ids))

    # Top vizinhos (exclui ele mesmo)
    neighbors = user_sim.loc[cliente_id].drop(cliente_id).sort_values(ascending=False)
    top_neighbors = neighbors.head(20)  # 20 vizinhos

    # Score colaborativo: média ponderada dos ratings dos vizinhos
    coll_scores = {}
    for rid in not_rated:
        neighbor_ratings = user_item.loc[top_neighbors.index, rid]
        weights = top_neighbors.values
        if weights.sum() > 0:
            score = float(np.dot(neighbor_ratings, weights) / weights.sum())
        else:
            score = float(neighbor_ratings.mean())
        coll_scores[rid] = score

    # Preferência por categoria do cliente (conteúdo)
    user_df = df[df["cliente_id"] == cliente_id]
    pref_cat = user_df.groupby("categoria")["rating"].mean()

    # Mapas auxiliares: categoria e nome por restaurante_id
    rest_info = df.drop_duplicates(subset=["restaurante_id"])[["restaurante_id", "categoria", "nome"]].set_index("restaurante_id")
    rest_cat = rest_info["categoria"]
    rest_name = rest_info["nome"]

    # Combina colaborativa + conteúdo
    final = []
    for rid in not_rated:
        base = coll_scores.get(rid, 0.0)
        cat = rest_cat.get(rid, None)
        boost = 0.0
        if cat is not None:
            boost = pref_cat.get(cat, 0.0) - 3.0  # acima de 3 indica preferência
        score = base + 0.2 * boost  # peso moderado do conteúdo
        nome = rest_name.get(rid, f"Restaurante #{rid}")
        final.append((rid, nome, cat, score))

    # Ordena por score e retorna top_n
    final_sorted = sorted(final, key=lambda x: x[3], reverse=True)[:top_n]
    recommendations = pd.DataFrame(final_sorted, columns=["restaurante_id", "nome", "categoria", "score"])
    return recommendations

def build_recommender(path="data/historico_pedidos.csv"):
    df = load_data(path)
    user_item = build_user_item_matrix(df)
    user_sim = compute_user_similarity(user_item)
    return df, user_item, user_sim

if __name__ == "__main__":
    df, user_item, user_sim = build_recommender()
    recs = recommend_for_user(df, user_item, user_sim, cliente_id=10, top_n=5)
    print("✅ Recomendações para cliente 10:")
    print(recs)
