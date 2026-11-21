#!/usr/bin/env python3
"""
Análise de processo — produção de biodiesel (dados fictícios).

Uso:  python analise_processo.py producao_biodiesel_2024_2025_ficticio.csv

Etapas: limpeza dos dados, comparação por reator / fornecedor / turno,
modelo de regressão linear múltipla para o rendimento, cenários de ganho
e geração do painel de gráficos (painel_processo.png).
Bibliotecas: pandas, numpy, scipy, matplotlib.
"""
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

PRECO_BIODIESEL = 5.20   # R$/kg  (premissa fictícia, ajuste para o caso real)
LIM_ESTER = 96.5         # especificação: teor de ésteres mínimo (%)
LIM_ACIDEZ = 0.50        # especificação: acidez máxima (mg KOH/g)


# ----------------------------------------------------------------------------
def limpar(caminho):
    df = pd.read_csv(caminho, encoding="utf-8-sig", parse_dates=["data_hora_inicio"])
    q = {"linhas no arquivo": len(df)}

    antes = len(df)
    df = df.drop_duplicates("lote_id", keep="first")
    q["lotes duplicados removidos"] = antes - len(df)

    novo = df["reator"].astype(str).map(lambda s: "R" + re.sub(r"\D", "", s))
    q["nomes de reator padronizados"] = int((novo != df["reator"]).sum())
    df["reator"] = novo

    falha = (df["temp_reacao_C"] < 30) | (df["temp_reacao_C"] > 90)
    q["temperatura com falha de sensor (0 ou 999)"] = int(falha.sum())
    df.loc[falha, "temp_reacao_C"] = np.nan

    imp = df["rendimento_pct"] > 100
    q["rendimento acima de 100% (erro de digitação)"] = int(imp.sum())
    df.loc[imp, "rendimento_pct"] = np.nan

    q["umidade do óleo em branco"] = int(df["umidade_oleo"].isna().sum())
    q["acidez do produto em branco"] = int(df["acidez_produto"].isna().sum())
    df["umidade_oleo"] = df["umidade_oleo"].fillna(df.groupby("fornecedor")["umidade_oleo"].transform("median"))
    df["nao_conforme"] = df["conforme"].eq("Não")
    q["lotes usados na análise"] = len(df)
    return df.reset_index(drop=True), q


# ----------------------------------------------------------------------------
def ols(X, y, nomes):
    X = np.column_stack([np.ones(len(X)), X])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    res = y - X @ beta
    n, k = X.shape
    s2 = res @ res / (n - k)
    se = np.sqrt(np.diag(s2 * np.linalg.inv(X.T @ X)))
    t = beta / se
    p = 2 * (1 - stats.t.cdf(np.abs(t), n - k))
    tc = stats.t.ppf(0.975, n - k)
    r2 = 1 - res @ res / ((y - y.mean()) @ (y - y.mean()))
    tabela = pd.DataFrame({"coef": beta, "ic95_inf": beta - tc * se, "ic95_sup": beta + tc * se, "p": p},
                          index=["constante"] + nomes)
    return beta, tabela, r2, np.sqrt(s2)


def matriz(d):
    """Variáveis do modelo (centradas para interpretar os coeficientes)."""
    X = pd.DataFrame({
        "temp (°C acima de 62)": d["temp_reacao_C"] - 62,
        "temp² ": (d["temp_reacao_C"] - 62) ** 2,
        "razão molar (metanol:óleo)": d["razao_molar"],
        "razão molar²": d["razao_molar"] ** 2,
        "catalisador (% acima de 1)": d["catalisador_pct"] - 1,
        "catalisador²": (d["catalisador_pct"] - 1) ** 2,
        "ln(tempo em min)": np.log(d["tempo_min"]),
        "agitação (por 100 rpm)": (d["agitacao_rpm"] - 400) / 100,
        "acidez do óleo (mg KOH/g)": d["acidez_oleo"],
        "umidade do óleo (por 0,1%)": d["umidade_oleo"] * 10,
        "reator R2 (vs R1)": (d["reator"] == "R2").astype(float),
        "reator R3 (vs R1)": (d["reator"] == "R3").astype(float),
        "dias desde limpeza, R1/R2 (por 30 d)": np.where(d["reator"] != "R3", d["dias_desde_limpeza"], 0) / 30,
        "dias desde limpeza, R3 (por 30 d)": np.where(d["reator"] == "R3", d["dias_desde_limpeza"], 0) / 30,
    })
    return X


# ----------------------------------------------------------------------------
def main():
    caminho = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("producao_biodiesel_2024_2025_ficticio.csv")
    if not caminho.exists():
        sys.exit(f"Erro: arquivo não encontrado: {caminho}")
    df, q = limpar(caminho)

    print("=== QUALIDADE DOS DADOS ===")
    for k, v in q.items():
        print(f"{k}: {v}")

    print("\n=== VISÃO GERAL ===")
    print(f"rendimento médio: {df.rendimento_pct.mean():.2f}% | lotes não conformes: {df.nao_conforme.mean()*100:.1f}%")

    for col in ["fornecedor", "reator", "turno"]:
        g = df.groupby(col).agg(lotes=("lote_id", "count"), rend=("rendimento_pct", "mean"),
                                nc=("nao_conforme", "mean"), sd_temp=("temp_reacao_C", "std"))
        g["nc"] = (g["nc"] * 100).round(1)
        print(f"\n--- por {col} ---")
        print(g.round(2).to_string())

    # R3: perda entre limpezas
    r3 = df[df.reator == "R3"]
    faixa = pd.cut(r3.dias_desde_limpeza, [-1, 60, 120, 180, 240, 330], labels=["0-60", "61-120", "121-180", "181-240", "241-330"])
    print("\n--- R3: rendimento por dias desde a limpeza ---")
    print(r3.groupby(faixa, observed=True).rendimento_pct.agg(["count", "mean"]).round(2).to_string())

    # ----- modelo
    d = df.dropna(subset=["rendimento_pct", "temp_reacao_C", "razao_molar", "catalisador_pct", "tempo_min",
                          "agitacao_rpm", "acidez_oleo", "umidade_oleo"]).copy()
    X = matriz(d)
    beta, tab, r2, rmse = ols(X.values, d["rendimento_pct"].values, list(X.columns))
    print(f"\n=== MODELO DE REGRESSÃO (n={len(d)}) ===")
    print(f"R² = {r2:.3f} | erro padrão residual = {rmse:.2f} p.p.")
    print(tab.round(3).to_string())
    b = dict(zip(["constante"] + list(X.columns), beta))
    t_otimo = 62 - b["temp (°C acima de 62)"] / (2 * b["temp² "])
    r_otimo = -b["razão molar (metanol:óleo)"] / (2 * b["razão molar²"])
    c_otimo = 1 - b["catalisador (% acima de 1)"] / (2 * b["catalisador²"])
    print(f"\nponto ótimo estimado: temperatura {t_otimo:.1f} °C | razão molar {r_otimo:.1f}:1 | catalisador {c_otimo:.2f}%")
    print(f"ganho de aumentar o tempo de 60 para 90 min: {b['ln(tempo em min)']*np.log(90/60):.2f} p.p.")

    # ----- cenários (usando o modelo)
    def prever(dd):
        return np.column_stack([np.ones(len(dd)), matriz(dd).values]) @ beta

    base = prever(d)
    massa = d["massa_oleo_kg"].mean()
    lotes_ano = len(d) / ((d.data_hora_inicio.max() - d.data_hora_inicio.min()).days / 365)
    cen = {}
    c1 = d.copy(); c1["temp_reacao_C"] = c1["temp_reacao_C"].clip(60.5, 63.5)
    cen["Temperatura sempre entre 60,5 e 63,5 °C"] = prever(c1) - base
    c2 = d.copy(); c2["dias_desde_limpeza"] = np.where(c2.reator == "R3", np.minimum(c2.dias_desde_limpeza, 150), c2.dias_desde_limpeza)
    cen["Limpeza do R3 a cada 150 dias"] = prever(c2) - base
    c3 = d.copy(); m = c3.fornecedor == "C"
    c3.loc[m, "acidez_oleo"] = 0.8; c3.loc[m, "umidade_oleo"] = 0.06
    cen["Pré-tratamento do óleo residual (acidez 0,8; umidade 0,06%)"] = prever(c3) - base
    print(f"\n=== CENÁRIOS (premissas: R$ {PRECO_BIODIESEL:.2f}/kg, {massa:.0f} kg de óleo/lote, {lotes_ano:.0f} lotes/ano) ===")
    for nome, delta in cen.items():
        pp = delta.mean()
        kg = pp / 100 * massa * lotes_ano
        print(f"{nome}: +{pp:.2f} p.p. | {kg:,.0f} kg/ano | R$ {kg*PRECO_BIODIESEL:,.0f}/ano")

    # ----- figuras
    cor = {"R1": "#0c7c86", "R2": "#7fc6cb", "R3": "#e0902a"}
    fig, ax = plt.subplots(2, 2, figsize=(13, 9))
    mes = d.assign(m=d.data_hora_inicio.dt.to_period("M").dt.to_timestamp()).groupby(["m", "reator"]).rendimento_pct.mean().unstack()
    for r in mes.columns:
        ax[0, 0].plot(mes.index, mes[r], label=r, color=cor[r], lw=2)
    ax[0, 0].set_title("Rendimento médio mensal por reator (%)"); ax[0, 0].legend(frameon=False)
    ax[0, 0].set_ylabel("Rendimento (%)")

    forn = ["A", "B", "C"]
    bp = ax[0, 1].boxplot([d[d.fornecedor == f].rendimento_pct for f in forn], tick_labels=["A (soja)", "B (soja)", "C (residual)"],
                          patch_artist=True, showfliers=False)
    for p, cc in zip(bp["boxes"], ["#0c7c86", "#0c7c86", "#e0902a"]):
        p.set_facecolor(cc); p.set_alpha(.7)
    ax[0, 1].set_title("Rendimento por fornecedor de óleo (%)")

    fx = d.assign(f=d.temp_reacao_C.round(0)).groupby("f").rendimento_pct.agg(["mean", "count"])
    fx = fx[fx["count"] >= 30]
    ax[1, 0].plot(fx.index, fx["mean"], marker="o", color="#0c7c86", lw=2)
    ax[1, 0].axvline(t_otimo, color="#e0902a", ls="--"); ax[1, 0].text(t_otimo + .2, fx["mean"].min(), f"ótimo ≈ {t_otimo:.1f} °C", color="#e0902a")
    ax[1, 0].set_title("Rendimento médio por temperatura de reação"); ax[1, 0].set_xlabel("Temperatura (°C)")

    tn = df.groupby("turno").nao_conforme.mean().reindex(["Manhã", "Tarde", "Noite"]) * 100
    ax[1, 1].bar(tn.index, tn.values, color=["#7fc6cb", "#7fc6cb", "#e0902a"])
    for i, v in enumerate(tn.values):
        ax[1, 1].text(i, v + .2, f"{v:.1f}%", ha="center")
    ax[1, 1].set_title("Lotes fora da especificação por turno (%)")
    for a in ax.flat:
        a.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Química Horizonte (fictícia) — painel de processo, 2024–2025", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig("painel_processo.png", dpi=140)
    print("\nPainel salvo em painel_processo.png")


if __name__ == "__main__":
    main()
