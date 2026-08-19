"""
Análise da Copa do Mundo 2026
==============================
Este script:
  1. Importa o CSV de partidas da Copa do Mundo;
  2. Faz a limpeza e o tratamento dos dados;
  3. Constrói métricas de desempenho por seleção jogando em casa (mandante)
     e como visitante;
  4. Aplica clusterização (K-Means) para agrupar seleções com perfis
     ofensivo/defensivo, casa x fora, semelhantes;
  5. Gera um dashboard HTML interativo (Plotly) com a correlação entre o
     desempenho como mandante e como visitante, além dos clusters.

Uso:
    python analise_copa_do_mundo.py caminho/para/world-cup-matches.csv

Saída:
    dashboard_copa_do_mundo.html  -> dashboard interativo
    times_clusterizados.csv       -> tabela com métricas + cluster de cada time
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

import plotly.graph_objects as go
from plotly.subplots import make_subplots

warnings.filterwarnings("ignore")

RANDOM_STATE = 42


# ---------------------------------------------------------------------------
# 1. IMPORTAÇÃO
# ---------------------------------------------------------------------------
def carregar_dados(caminho_csv: str) -> pd.DataFrame:
    df = pd.read_csv(caminho_csv)
    print(f"[OK] {len(df)} partidas carregadas de '{caminho_csv}'.")
    return df


# ---------------------------------------------------------------------------
# 2. LIMPEZA E TRATAMENTO
# ---------------------------------------------------------------------------
def limpar_dados(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # --- Tipos e datas -----------------------------------------------------
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # --- Padroniza texto (remove espaços extras, ex.: "Mat ě j Kovar") -----
    colunas_texto = [
        "stage_name", "stadium_name", "city", "country",
        "home_team_name", "home_fifa_code", "away_team_name", "away_fifa_code",
        "status", "result_type", "home_goalkeeper", "away_goalkeeper",
        "player_of_the_match_name", "referee_name",
    ]
    for col in colunas_texto:
        if col in df.columns:
            df[col] = (
                df[col]
                .astype(str)
                .str.replace(r"\s+", " ", regex=True)
                .str.strip()
                .replace({"nan": np.nan})
            )

    # --- Placar / xG numéricos ---------------------------------------------
    for col in ["home_score", "away_score", "home_xg", "away_xg",
                "home_penalty_score", "away_penalty_score"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Pênaltis só existem quando houve disputa; preenchendo ausência com NaN
    # é o correto (não faz sentido usar 0), então mantemos como está.

    # --- Remove duplicatas e partidas não concluídas ------------------------
    antes = len(df)
    df = df.drop_duplicates(subset="match_id")
    df = df[df["status"].str.lower() == "completed"]
    df = df.dropna(subset=["home_score", "away_score"])
    depois = len(df)
    if antes != depois:
        print(f"[LIMPEZA] {antes - depois} linha(s) removida(s) (duplicadas/incompletas).")

    # --- Variáveis derivadas -------------------------------------------------
    df["gol_diff"] = df["home_score"] - df["away_score"]
    df["total_gols"] = df["home_score"] + df["away_score"]
    df["xg_diff"] = df["home_xg"] - df["away_xg"]

    def resultado_mandante(row):
        if row["home_score"] > row["away_score"]:
            return "Vitória Casa"
        elif row["home_score"] < row["away_score"]:
            return "Vitória Visitante"
        return "Empate"

    df["resultado"] = df.apply(resultado_mandante, axis=1)

    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# 3. MÉTRICAS POR SELEÇÃO (CASA x VISITANTE)
# ---------------------------------------------------------------------------
def construir_metricas_times(df: pd.DataFrame) -> pd.DataFrame:
    # Desempenho como mandante
    casa = df.groupby("home_team_name").agg(
        jogos_casa=("match_id", "count"),
        gols_marcados_casa=("home_score", "mean"),
        gols_sofridos_casa=("away_score", "mean"),
        xg_casa=("home_xg", "mean"),
        vitorias_casa=("resultado", lambda s: (s == "Vitória Casa").sum()),
    )

    # Desempenho como visitante
    fora = df.groupby("away_team_name").agg(
        jogos_fora=("match_id", "count"),
        gols_marcados_fora=("away_score", "mean"),
        gols_sofridos_fora=("home_score", "mean"),
        xg_fora=("away_xg", "mean"),
        vitorias_fora=("resultado", lambda s: (s == "Vitória Visitante").sum()),
    )

    times = casa.join(fora, how="outer").fillna(0)
    times.index.name = "time"
    times = times.reset_index()

    times["jogos_totais"] = times["jogos_casa"] + times["jogos_fora"]
    times["saldo_gols_casa"] = times["gols_marcados_casa"] - times["gols_sofridos_casa"]
    times["saldo_gols_fora"] = times["gols_marcados_fora"] - times["gols_sofridos_fora"]
    times["vantagem_casa"] = times["saldo_gols_casa"] - times["saldo_gols_fora"]

    # Só times que jogaram pelo menos uma vez em casa e uma fora entram na
    # correlação casa x visitante (para o gráfico específico), mas todos
    # entram na clusterização.
    return times


# ---------------------------------------------------------------------------
# 4. CLUSTERIZAÇÃO (K-MEANS)
# ---------------------------------------------------------------------------
def clusterizar_times(times: pd.DataFrame, n_clusters: int = 4) -> pd.DataFrame:
    features = [
        "gols_marcados_casa", "gols_sofridos_casa",
        "gols_marcados_fora", "gols_sofridos_fora",
        "xg_casa", "xg_fora",
    ]

    dados = times[features].fillna(0)

    scaler = StandardScaler()
    dados_norm = scaler.fit_transform(dados)

    k = min(n_clusters, max(2, times.shape[0] // 3))  # evita erro com poucos times
    kmeans = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
    times = times.copy()
    times["cluster"] = kmeans.fit_predict(dados_norm)

    # Nomeia os clusters de forma interpretável, com base no perfil médio
    perfil = times.groupby("cluster")[features].mean()
    perfil["poder_ofensivo"] = perfil["gols_marcados_casa"] + perfil["gols_marcados_fora"]
    perfil["poder_defensivo"] = -(perfil["gols_sofridos_casa"] + perfil["gols_sofridos_fora"])
    perfil["indice_geral"] = perfil["poder_ofensivo"] + perfil["poder_defensivo"]
    ordem = perfil["indice_geral"].sort_values(ascending=False).index.tolist()

    nomes = ["Elite (ataque e defesa fortes)", "Equilibrados", "Ofensivos irregulares", "Discretos/eliminados cedo"]
    mapa_nomes = {cluster_id: nomes[i] if i < len(nomes) else f"Grupo {i+1}"
                  for i, cluster_id in enumerate(ordem)}
    times["perfil_cluster"] = times["cluster"].map(mapa_nomes)

    print("\n[CLUSTERIZAÇÃO] Perfil médio por cluster:")
    print(perfil.round(2))

    return times


# ---------------------------------------------------------------------------
# 5. DASHBOARD (PLOTLY)
# ---------------------------------------------------------------------------
def gerar_dashboard(df: pd.DataFrame, times: pd.DataFrame, caminho_saida: str):
    cores_cluster = {
        "Elite (ataque e defesa fortes)": "#2ecc71",
        "Equilibrados": "#3498db",
        "Ofensivos irregulares": "#f39c12",
        "Discretos/eliminados cedo": "#e74c3c",
    }
    default_color = "#95a5a6"

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "Correlação: Gols marcados em casa x fora",
            "Correlação: Gols sofridos em casa x fora",
            "Clusters de times (Ataque em casa x fora)",
            "Matriz de correlação (métricas casa x visitante)",
        ),
        specs=[[{"type": "scatter"}, {"type": "scatter"}],
               [{"type": "scatter"}, {"type": "heatmap"}]],
        vertical_spacing=0.14,
        horizontal_spacing=0.10,
    )

    # --- (1,1) Gols marcados casa x fora, colorido por cluster --------------
    for perfil, cor in cores_cluster.items():
        sub = times[times["perfil_cluster"] == perfil]
        fig.add_trace(
            go.Scatter(
                x=sub["gols_marcados_casa"], y=sub["gols_marcados_fora"],
                mode="markers+text", text=sub["time"], textposition="top center",
                textfont=dict(size=8),
                marker=dict(size=10, color=cor, line=dict(width=1, color="white")),
                name=perfil, legendgroup=perfil,
                hovertemplate="<b>%{text}</b><br>Gols/jogo casa: %{x:.2f}<br>Gols/jogo fora: %{y:.2f}<extra></extra>",
            ),
            row=1, col=1,
        )

    corr_marcados = times["gols_marcados_casa"].corr(times["gols_marcados_fora"])
    fig.add_annotation(
        text=f"corr = {corr_marcados:.2f}", xref="x domain", yref="y domain",
        x=0.05, y=0.95, showarrow=False, font=dict(size=11, color="#555"),
    )

    # --- (1,2) Gols sofridos casa x fora -------------------------------------
    for perfil, cor in cores_cluster.items():
        sub = times[times["perfil_cluster"] == perfil]
        fig.add_trace(
            go.Scatter(
                x=sub["gols_sofridos_casa"], y=sub["gols_sofridos_fora"],
                mode="markers+text", text=sub["time"], textposition="top center",
                textfont=dict(size=8),
                marker=dict(size=10, color=cor, line=dict(width=1, color="white")),
                name=perfil, legendgroup=perfil, showlegend=False,
                hovertemplate="<b>%{text}</b><br>Gols sofridos/jogo casa: %{x:.2f}<br>Gols sofridos/jogo fora: %{y:.2f}<extra></extra>",
            ),
            row=1, col=2,
        )

    corr_sofridos = times["gols_sofridos_casa"].corr(times["gols_sofridos_fora"])
    fig.add_annotation(
        text=f"corr = {corr_sofridos:.2f}", xref="x2 domain", yref="y2 domain",
        x=0.05, y=0.95, showarrow=False, font=dict(size=11, color="#555"),
    )

    # --- (2,1) Clusters: saldo de gols casa x fora ---------------------------
    for perfil, cor in cores_cluster.items():
        sub = times[times["perfil_cluster"] == perfil]
        fig.add_trace(
            go.Scatter(
                x=sub["saldo_gols_casa"], y=sub["saldo_gols_fora"],
                mode="markers+text", text=sub["time"], textposition="top center",
                textfont=dict(size=8),
                marker=dict(size=12, color=cor, line=dict(width=1, color="white"), symbol="diamond"),
                name=perfil, legendgroup=perfil, showlegend=False,
                hovertemplate="<b>%{text}</b><br>Saldo casa: %{x:.2f}<br>Saldo fora: %{y:.2f}<extra></extra>",
            ),
            row=2, col=1,
        )

    # --- (2,2) Heatmap de correlação -----------------------------------------
    metricas_corr = [
        "gols_marcados_casa", "gols_sofridos_casa", "xg_casa",
        "gols_marcados_fora", "gols_sofridos_fora", "xg_fora",
    ]
    matriz = times[metricas_corr].corr().round(2)
    rotulos = ["Gols marc. (casa)", "Gols sofr. (casa)", "xG (casa)",
               "Gols marc. (fora)", "Gols sofr. (fora)", "xG (fora)"]

    fig.add_trace(
        go.Heatmap(
            z=matriz.values, x=rotulos, y=rotulos,
            colorscale="RdBu", zmid=0, zmin=-1, zmax=1,
            text=matriz.values, texttemplate="%{text}",
            colorbar=dict(title="corr", len=0.45, y=0.2),
        ),
        row=2, col=2,
    )

    fig.update_xaxes(title_text="Gols marcados/jogo em casa", row=1, col=1)
    fig.update_yaxes(title_text="Gols marcados/jogo fora", row=1, col=1)
    fig.update_xaxes(title_text="Gols sofridos/jogo em casa", row=1, col=2)
    fig.update_yaxes(title_text="Gols sofridos/jogo fora", row=1, col=2)
    fig.update_xaxes(title_text="Saldo de gols em casa", row=2, col=1)
    fig.update_yaxes(title_text="Saldo de gols fora", row=2, col=1)

    fig.update_layout(
        title=dict(
            text="Dashboard - Copa do Mundo: Desempenho Mandante x Visitante e Clusterização de Seleções",
            font=dict(size=18),
        ),
        height=950,
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="center", x=0.5),
        margin=dict(t=120),
    )

    fig.write_html(caminho_saida, include_plotlyjs="cdn")
    print(f"[OK] Dashboard salvo em '{caminho_saida}'.")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    caminho_csv = sys.argv[1] if len(sys.argv) > 1 else "world-cup-matches.csv"
    saida_dashboard = "dashboard_copa_do_mundo.html"
    saida_tabela = "times_clusterizados.csv"

    df = carregar_dados(caminho_csv)
    df_limpo = limpar_dados(df)
    times = construir_metricas_times(df_limpo)
    times = clusterizar_times(times)

    gerar_dashboard(df_limpo, times, saida_dashboard)

    times_ordenado = times.sort_values("vantagem_casa", ascending=False)
    times_ordenado.to_csv(saida_tabela, index=False)
    print(f"[OK] Tabela de times/clusters salva em '{saida_tabela}'.")

    print("\n[RESUMO] Times com maior vantagem em casa (saldo casa - saldo fora):")
    print(times_ordenado[["time", "saldo_gols_casa", "saldo_gols_fora", "vantagem_casa", "perfil_cluster"]].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
