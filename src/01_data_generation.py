import os
import numpy as np
import pandas as pd

# Configura aleatoriedade para reprodutibilidade
np.random.seed(42)

# Garante pasta data
os.makedirs("data", exist_ok=True)

# Parâmetros
n_clientes = 300
n_restaurantes = 120
categorias = ["pizza", "burguer", "japonesa", "brasileira", "saudavel", "doces", "mexicana", "italiana"]

# Função para gerar nome fictício de restaurante
def gerar_nome_restaurante(categoria, id):
    prefixos = {
        "pizza": ["Pizza", "La Massa", "Rodízio"],
        "burguer": ["Burger", "Smash", "Hambúrguer"],
        "japonesa": ["Sushi", "Temaki", "Japão"],
        "brasileira": ["Sabor", "Feijão", "Brasil"],
        "saudavel": ["Verde", "Natural", "Leve"],
        "doces": ["Açúcar", "Doçura", "Confeitaria"],
        "mexicana": ["Tacos", "México", "Nacho"],
        "italiana": ["Pasta", "Cantina", "Itália"]
    }
    sufixos = ["da Vila", "360", "Express", "Gourmet", "Zen", "do Chef", "no Prato"]
    nome = f"{np.random.choice(prefixos[categoria])} {np.random.choice(sufixos)}"
    return nome + f" #{id}"

# Gera restaurantes com categorias, popularidade e nomes
restaurantes = pd.DataFrame({
    "restaurante_id": range(1, n_restaurantes + 1),
    "categoria": np.random.choice(categorias, size=n_restaurantes)
})
restaurantes["popularidade"] = np.clip(np.random.beta(2, 5, size=n_restaurantes), 0.05, 0.95)
restaurantes["nome"] = restaurantes.apply(lambda row: gerar_nome_restaurante(row["categoria"], row["restaurante_id"]), axis=1)

# Gera preferências por cliente (peso por categoria)
clientes = pd.DataFrame({
    "cliente_id": range(1, n_clientes + 1)
})
afinidades = pd.DataFrame({
    "cliente_id": np.repeat(clientes["cliente_id"], len(categorias)),
    "categoria": categorias * n_clientes,
    "afinidade": np.clip(np.random.beta(3, 3, size=n_clientes * len(categorias)), 0.01, 0.99)
})

# Simula histórico de pedidos/avaliações
registros = []
for cid in clientes["cliente_id"]:
    n_inter = np.random.randint(20, 81)
    chosen = restaurantes.sample(n_inter, replace=True, weights=restaurantes["popularidade"])
    for _, row in chosen.iterrows():
        cat = row["categoria"]
        nome = row["nome"]
        aff = afinidades[(afinidades["cliente_id"] == cid) & (afinidades["categoria"] == cat)]["afinidade"].values[0]
        base = 0.4 * aff + 0.6 * row["popularidade"]
        rating = np.clip(np.random.normal(loc=2.5 + 2 * base, scale=0.7), 1, 5)
        registros.append({
            "cliente_id": cid,
            "restaurante_id": row["restaurante_id"],
            "nome": nome,
            "categoria": cat,
            "rating": round(float(rating), 2)
        })

historico = pd.DataFrame(registros)

# Salva
out_path = os.path.join("data", "historico_pedidos.csv")
historico.to_csv(out_path, index=False)
print(f"✅ Dados gerados em {out_path} com shape {historico.shape}")
