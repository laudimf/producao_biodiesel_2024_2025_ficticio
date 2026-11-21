"""Painel de processo — produção de biodiesel, Química Horizonte (dados fictícios)."""
import re
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Painel de processo — Química Horizonte", page_icon="⚗️", layout="wide")

CSV = Path(__file__).parent / "producao_biodiesel_2024_2025_ficticio.csv"
COR_REATOR = {"R1": "#0c7c86", "R2": "#7fc6cb", "R3": "#e0902a"}
COR_FORN = {"A": "#0c7c86", "B": "#7fc6cb", "C": "#e0902a"}
TURNOS = ["Manhã", "Tarde", "Noite"]


def pct(v):
    return f"{v:.1f}".replace(".", ",") + "%"


def num(v, casas=0):
    return f"{v:,.{casas}f}".replace(",", "X").replace(".", ",").replace("X", ".")


def escala(mapa):
    return alt.Scale(domain=list(mapa.keys()), range=list(mapa.values()))


@st.cache_data
def carregar():
    df = pd.read_csv(CSV, encoding="utf-8-sig", parse_dates=["data_hora_inicio"])
    q = {"Linhas no arquivo": len(df)}
    antes = len(df)
    df = df.drop_duplicates("lote_id", keep="first")
    q["Lotes duplicados removidos"] = antes - len(df)
    novo = df["reator"].astype(str).map(lambda s: "R" + re.sub(r"\D", "", s))
    q["Nomes de reator padronizados"] = int((novo != df["reator"]).sum())
    df["reator"] = novo
    falha = (df["temp_reacao_C"] < 30) | (df["temp_reacao_C"] > 90)
    q["Temperaturas com falha de sensor (0 ou 999 °C)"] = int(falha.sum())
    df.loc[falha, "temp_reacao_C"] = np.nan
    imp = df["rendimento_pct"] > 100
    q["Rendimentos acima de 100% (erro de digitação)"] = int(imp.sum())
    df.loc[imp, "rendimento_pct"] = np.nan
    q["Umidade do óleo em branco (preenchida pela mediana do fornecedor)"] = int(df["umidade_oleo"].isna().sum())
    df["umidade_oleo"] = df["umidade_oleo"].fillna(df.groupby("fornecedor")["umidade_oleo"].transform("median"))
    df["fora_espec"] = (df["conforme"] == "Não") * 100.0
    df["mes"] = df["data_hora_inicio"].dt.to_period("M").dt.to_timestamp()
    q["Lotes na análise"] = len(df)
    return df.reset_index(drop=True), q


def resumo(d, col):
    g = d.groupby(col).agg(lotes=("lote_id", "count"), rendimento=("rendimento_pct", "mean"),
                           fora=("fora_espec", "mean"), dp_temp=("temp_reacao_C", "std"),
                           acidez=("acidez_oleo", "mean"), umidade=("umidade_oleo", "mean")).reset_index()
    return g


def caixa(d, col, cores):
    g = d.groupby(col)["rendimento_pct"]
    s = g.quantile([0.25, 0.5, 0.75]).unstack()
    s.columns = ["q1", "med", "q3"]
    s["lo"], s["hi"] = g.min(), g.max()
    s = s.reset_index()
    base = alt.Chart(s).encode(x=alt.X(f"{col}:N", title=None))
    y = alt.Y("lo:Q", title="Rendimento (%)", scale=alt.Scale(zero=False))
    rule = base.mark_rule(color="#5d6f76").encode(y=y, y2="hi:Q")
    bar = base.mark_bar(size=46, opacity=0.85).encode(
        y="q1:Q", y2="q3:Q", color=alt.Color(f"{col}:N", scale=escala(cores), legend=None))
    med = base.mark_tick(color="white", size=46, thickness=2).encode(
        y="med:Q", tooltip=[col, alt.Tooltip("med:Q", title="Mediana", format=".2f")])
    return alt.layer(rule, bar, med).properties(width="container", height=300)


# --- modelo de risco do lote (regressão logística com termos quadráticos, só com numpy)
NUM = ["massa_oleo_kg", "acidez_oleo", "umidade_oleo", "temp_reacao_C", "razao_molar",
       "catalisador_pct", "tempo_min", "agitacao_rpm", "dias_desde_limpeza"]
COLUNAS_MODELO = NUM + ["reator_R2", "reator_R3", "turno_Noite", "turno_Tarde",
                        "fornecedor_B", "fornecedor_C", "temp_c2", "razao_c2", "catal_c2"]
GRUPOS = {"massa_oleo_kg": "Massa de óleo", "acidez_oleo": "Acidez do óleo", "umidade_oleo": "Umidade do óleo",
          "temp_reacao_C": "Temperatura", "temp_c2": "Temperatura", "razao_molar": "Razão molar",
          "razao_c2": "Razão molar", "catalisador_pct": "Catalisador", "catal_c2": "Catalisador",
          "tempo_min": "Tempo de reação", "agitacao_rpm": "Agitação", "dias_desde_limpeza": "Dias desde a limpeza",
          "reator_R2": "Reator", "reator_R3": "Reator", "turno_Noite": "Turno", "turno_Tarde": "Turno",
          "fornecedor_B": "Fornecedor do óleo", "fornecedor_C": "Fornecedor do óleo"}


def caracteristicas(d):
    X = d[NUM].astype(float).copy()
    for col, val in [("reator", "R2"), ("reator", "R3"), ("turno", "Noite"), ("turno", "Tarde"),
                     ("fornecedor", "B"), ("fornecedor", "C")]:
        X[f"{col}_{val}"] = (d[col] == val).astype(float)
    X["temp_c2"] = (d["temp_reacao_C"] - 62) ** 2
    X["razao_c2"] = (d["razao_molar"] - 7.8) ** 2
    X["catal_c2"] = (d["catalisador_pct"] - 1.0) ** 2
    return X[COLUNAS_MODELO]


def ajustar_logistica(X, y, l2=1.0, iteracoes=50):
    """Regressão logística com penalidade L2, ajustada pelo método de Newton."""
    mu, sd = X.mean(axis=0), X.std(axis=0)
    sd[sd == 0] = 1.0
    Z = np.column_stack([np.ones(len(X)), (X - mu) / sd])
    pen = np.eye(Z.shape[1]) * l2
    pen[0, 0] = 0.0
    w = np.zeros(Z.shape[1])
    for _ in range(iteracoes):
        p = 1 / (1 + np.exp(-(Z @ w)))
        grad = Z.T @ (p - y) + pen @ w
        hess = Z.T @ (Z * (p * (1 - p))[:, None]) + pen
        passo = np.linalg.solve(hess, grad)
        w -= passo
        if np.abs(passo).max() < 1e-8:
            break
    return {"mu": mu, "sd": sd, "w": w}


@st.cache_resource
def treinar_modelo(_dados):
    d = _dados.dropna(subset=NUM)
    return ajustar_logistica(caracteristicas(d).values, (d["fora_espec"] > 0).astype(float).values)


def risco(modelo, d):
    Z = (caracteristicas(d).values - modelo["mu"]) / modelo["sd"]
    eta = modelo["w"][0] + Z @ modelo["w"][1:]
    return 1 / (1 + np.exp(-eta)), Z


df, qualidade = carregar()
modelo = treinar_modelo(df)

# --- filtros
st.sidebar.header("Filtros")
dmin, dmax = df["data_hora_inicio"].dt.date.min(), df["data_hora_inicio"].dt.date.max()
periodo = st.sidebar.date_input("Período", value=(dmin, dmax), min_value=dmin, max_value=dmax)
if not isinstance(periodo, (tuple, list)) or len(periodo) != 2:
    st.info("Escolha a data inicial e a final no filtro de período.")
    st.stop()
reatores = st.sidebar.multiselect("Reator", sorted(df["reator"].unique()), default=sorted(df["reator"].unique()))
fornecedores = st.sidebar.multiselect("Fornecedor de óleo", sorted(df["fornecedor"].unique()),
                                      default=sorted(df["fornecedor"].unique()))
turnos = st.sidebar.multiselect("Turno", TURNOS, default=TURNOS)

dia = df["data_hora_inicio"].dt.date
d = df[(dia >= periodo[0]) & (dia <= periodo[1]) & df["reator"].isin(reatores)
       & df["fornecedor"].isin(fornecedores) & df["turno"].isin(turnos)]

st.title("Painel de processo — Química Horizonte")
st.caption("Produção de biodiesel em batelada, 2024–2025 · empresa e dados fictícios criados para portfólio")

if d.empty:
    st.warning("Nenhum lote para esta combinação de filtros.")
    st.stop()

# --- indicadores
k = st.columns(5)
k[0].metric("Lotes", num(len(d)))
k[1].metric("Rendimento médio", pct(d["rendimento_pct"].mean()))
k[2].metric("Fora da especificação", pct(d["fora_espec"].mean()))
k[3].metric("Biodiesel produzido", num(d["massa_biodiesel_kg"].sum() / 1000, 0) + " t")
k[4].metric("Variação da temperatura", num(d["temp_reacao_C"].std(), 1) + " °C")

por_reator = resumo(d, "reator")
por_forn = resumo(d, "fornecedor")
if len(por_reator) > 1 and len(por_forn) > 1:
    r_pior = por_reator.loc[por_reator["rendimento"].idxmin()]
    f_pior = por_forn.loc[por_forn["fora"].idxmax()]
    st.info(f"Menor rendimento: reator {r_pior['reator']} ({pct(r_pior['rendimento'])}). "
            f"Mais lotes fora da especificação: fornecedor {f_pior['fornecedor']} ({pct(f_pior['fora'])}).")

aba1, aba2, aba3, aba4, aba5, aba6 = st.tabs(["Visão geral", "Reatores e limpeza", "Matéria-prima",
                                                "Temperatura e turno", "Risco do lote", "Dados"])

with aba1:
    st.subheader("Rendimento médio mensal por reator")
    m = d.groupby(["mes", "reator"])["rendimento_pct"].mean().reset_index()
    st.altair_chart(alt.Chart(m).mark_line(point=True, strokeWidth=2.5).encode(
        x=alt.X("mes:T", title=None, axis=alt.Axis(format="%m/%y")),
        y=alt.Y("rendimento_pct:Q", title="Rendimento médio (%)", scale=alt.Scale(zero=False)),
        color=alt.Color("reator:N", scale=escala(COR_REATOR), legend=alt.Legend(title="Reator")),
        tooltip=[alt.Tooltip("mes:T", format="%m/%Y"), "reator",
                 alt.Tooltip("rendimento_pct:Q", title="Rendimento (%)", format=".2f")]
    ).properties(width="container", height=320))
    st.subheader("Lotes fora da especificação por mês")
    n = d.groupby("mes")["fora_espec"].mean().reset_index()
    st.altair_chart(alt.Chart(n).mark_bar(color="#e0902a").encode(
        x=alt.X("mes:T", title=None, axis=alt.Axis(format="%m/%y")),
        y=alt.Y("fora_espec:Q", title="Fora da especificação (%)"),
        tooltip=[alt.Tooltip("mes:T", format="%m/%Y"), alt.Tooltip("fora_espec:Q", title="%", format=".1f")]
    ).properties(width="container", height=240))

with aba2:
    st.subheader("Rendimento conforme o tempo desde a última limpeza")
    st.caption("O R3 perde rendimento de forma contínua e se recupera após cada limpeza; R1 e R2 quase não mudam.")
    f = d.assign(faixa=(d["dias_desde_limpeza"] // 30) * 30)
    g = f.groupby(["faixa", "reator"]).agg(rendimento=("rendimento_pct", "mean"), lotes=("lote_id", "count")).reset_index()
    g = g[g["lotes"] >= 20]
    st.altair_chart(alt.Chart(g).mark_line(point=True, strokeWidth=2.5).encode(
        x=alt.X("faixa:Q", title="Dias desde a limpeza (faixas de 30 dias)"),
        y=alt.Y("rendimento:Q", title="Rendimento médio (%)", scale=alt.Scale(zero=False)),
        color=alt.Color("reator:N", scale=escala(COR_REATOR), legend=alt.Legend(title="Reator")),
        tooltip=["reator", "faixa", alt.Tooltip("rendimento:Q", format=".2f"), "lotes"]
    ).properties(width="container", height=320))
    t = por_reator.rename(columns={"reator": "Reator", "lotes": "Lotes", "rendimento": "Rendimento médio (%)",
                                   "fora": "Fora da especificação (%)"})[["Reator", "Lotes", "Rendimento médio (%)", "Fora da especificação (%)"]]
    st.dataframe(t, hide_index=True, column_config={
        "Rendimento médio (%)": st.column_config.NumberColumn(format="%.2f"),
        "Fora da especificação (%)": st.column_config.NumberColumn(format="%.1f")})

with aba3:
    st.subheader("Rendimento por fornecedor de óleo")
    st.caption("Caixa: do 1º ao 3º quartil. Traço branco: mediana. Linha: mínimo a máximo.")
    st.altair_chart(caixa(d, "fornecedor", COR_FORN))
    tf = por_forn.rename(columns={"fornecedor": "Fornecedor", "lotes": "Lotes", "rendimento": "Rendimento médio (%)",
                                  "fora": "Fora da especificação (%)", "acidez": "Acidez média do óleo (mg KOH/g)",
                                  "umidade": "Umidade média do óleo (%)"})
    st.dataframe(tf.drop(columns="dp_temp"), hide_index=True, column_config={
        "Rendimento médio (%)": st.column_config.NumberColumn(format="%.2f"),
        "Fora da especificação (%)": st.column_config.NumberColumn(format="%.1f"),
        "Acidez média do óleo (mg KOH/g)": st.column_config.NumberColumn(format="%.2f"),
        "Umidade média do óleo (%)": st.column_config.NumberColumn(format="%.3f")})

with aba4:
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Rendimento por temperatura")
        st.caption("Pico próximo de 62 °C: abaixo falta velocidade de reação, acima há perda de metanol.")
        tt = d.assign(temp=d["temp_reacao_C"].round(0)).groupby("temp").agg(
            rendimento=("rendimento_pct", "mean"), lotes=("lote_id", "count")).reset_index()
        tt = tt[tt["lotes"] >= 20]
        linha = alt.Chart(tt).mark_line(point=True, color="#0c7c86", strokeWidth=2.5).encode(
            x=alt.X("temp:Q", title="Temperatura (°C)", scale=alt.Scale(zero=False)),
            y=alt.Y("rendimento:Q", title="Rendimento médio (%)", scale=alt.Scale(zero=False)),
            tooltip=["temp", alt.Tooltip("rendimento:Q", format=".2f"), "lotes"])
        regra = alt.Chart(pd.DataFrame({"x": [62]})).mark_rule(color="#e0902a", strokeDash=[5, 4]).encode(x="x:Q")
        st.altair_chart(alt.layer(linha, regra).properties(width="container", height=300))
    with c2:
        st.subheader("Fora da especificação por turno")
        tu = resumo(d, "turno")
        st.altair_chart(alt.Chart(tu).mark_bar().encode(
            x=alt.X("turno:N", sort=TURNOS, title=None),
            y=alt.Y("fora:Q", title="Fora da especificação (%)"),
            color=alt.condition(alt.datum.turno == "Noite", alt.value("#e0902a"), alt.value("#7fc6cb")),
            tooltip=["turno", alt.Tooltip("fora:Q", format=".1f"), "lotes"]
        ).properties(width="container", height=300))
    tu = tu.set_index("turno").reindex([t for t in TURNOS if t in set(tu["turno"])]).reset_index()
    tu = tu.rename(columns={"turno": "Turno", "lotes": "Lotes", "rendimento": "Rendimento médio (%)",
                            "fora": "Fora da especificação (%)", "dp_temp": "Variação da temperatura (°C)"})
    st.dataframe(tu[["Turno", "Lotes", "Rendimento médio (%)", "Fora da especificação (%)", "Variação da temperatura (°C)"]],
                 hide_index=True, column_config={
        "Rendimento médio (%)": st.column_config.NumberColumn(format="%.2f"),
        "Fora da especificação (%)": st.column_config.NumberColumn(format="%.1f"),
        "Variação da temperatura (°C)": st.column_config.NumberColumn(format="%.2f")})
    st.caption("O turno da noite tem poucos lotes; interprete a diferença com cautela.")

with aba5:
    st.subheader("Risco de o lote sair fora da especificação")
    st.caption("Informe as condições previstas para o lote e o modelo estima o risco antes do fim da batelada. "
               "Esta aba não usa os filtros da barra lateral.")
    c1, c2, c3 = st.columns(3)
    reator_in = c1.selectbox("Reator", ["R1", "R2", "R3"])
    turno_in = c1.selectbox("Turno", TURNOS)
    forn_in = c1.selectbox("Fornecedor do óleo", ["A", "B", "C"])
    dias_in = c1.slider("Dias desde a última limpeza", 0, 330, 60, 5)
    acid_in = c2.slider("Acidez do óleo (mg KOH/g)", 0.1, 5.5, 0.4, 0.1)
    umid_in = c2.slider("Umidade do óleo (%)", 0.01, 0.30, 0.05, 0.01)
    massa_in = c2.slider("Massa de óleo (kg)", 850, 1150, 1000, 10)
    temp_in = c3.slider("Temperatura de reação (°C)", 55.0, 70.0, 62.0, 0.5)
    razao_in = c3.slider("Razão molar metanol:óleo", 4.0, 9.0, 6.0, 0.1)
    cat_in = c3.slider("Catalisador (% da massa de óleo)", 0.5, 1.6, 1.0, 0.05)
    tempo_in = c3.slider("Tempo de reação (min)", 40, 120, 75, 5)
    rpm_in = c3.slider("Agitação (rpm)", 300, 560, 420, 10)

    linha_in = pd.DataFrame([{
        "massa_oleo_kg": massa_in, "acidez_oleo": acid_in, "umidade_oleo": umid_in, "temp_reacao_C": temp_in,
        "razao_molar": razao_in, "catalisador_pct": cat_in, "tempo_min": tempo_in, "agitacao_rpm": rpm_in,
        "dias_desde_limpeza": dias_in, "reator": reator_in, "turno": turno_in, "fornecedor": forn_in}])
    prob, Z = risco(modelo, linha_in)
    p = float(prob[0])
    m1, m2 = st.columns([1, 2])
    m1.metric("Risco de ficar fora da especificação", pct(p * 100))
    if p < 0.10:
        m2.success("Risco baixo: condições parecidas com as dos lotes que saem dentro da especificação.")
    elif p < 0.30:
        m2.warning("Risco moderado: veja abaixo os fatores que mais pesam neste lote.")
    else:
        m2.error("Risco alto: considere ajustar as condições ou o fornecedor antes de iniciar o lote.")
    st.caption(f"Para comparação, {pct((df['fora_espec'] > 0).mean() * 100)} dos lotes da base saíram fora da especificação.")

    g1, g2 = st.columns(2)
    with g1:
        st.subheader("O que pesa neste lote")
        st.caption("Em relação a um lote médio. Barras laranja aumentam o risco; azuis reduzem.")
        efeito = pd.Series(Z[0] * modelo["w"][1:], index=COLUNAS_MODELO).groupby(GRUPOS).sum()
        efeito = efeito.sort_values(ascending=False).reset_index()
        efeito.columns = ["fator", "efeito"]
        st.altair_chart(alt.Chart(efeito).mark_bar().encode(
            x=alt.X("efeito:Q", title="Efeito no risco (escala do modelo)"),
            y=alt.Y("fator:N", sort=list(efeito["fator"]), title=None),
            color=alt.condition(alt.datum.efeito > 0, alt.value("#e0902a"), alt.value("#7fc6cb")),
            tooltip=["fator", alt.Tooltip("efeito:Q", format=".2f")]
        ).properties(width="container", height=320))
    with g2:
        st.subheader("Risco conforme a temperatura")
        st.caption("Demais condições mantidas como informadas.")
        temps = np.arange(56.0, 68.01, 0.5)
        grade = pd.concat([linha_in] * len(temps), ignore_index=True)
        grade["temp_reacao_C"] = temps
        curva = pd.DataFrame({"temperatura": temps, "risco": risco(modelo, grade)[0] * 100})
        linha_c = alt.Chart(curva).mark_line(color="#0c7c86", strokeWidth=2.5).encode(
            x=alt.X("temperatura:Q", title="Temperatura (°C)", scale=alt.Scale(zero=False)),
            y=alt.Y("risco:Q", title="Risco (%)"),
            tooltip=[alt.Tooltip("temperatura:Q", format=".1f"), alt.Tooltip("risco:Q", format=".1f")])
        atual = alt.Chart(pd.DataFrame({"x": [temp_in]})).mark_rule(color="#e0902a", strokeDash=[5, 4]).encode(x="x:Q")
        st.altair_chart(alt.layer(linha_c, atual).properties(width="container", height=320))

    with st.expander("Sobre este modelo"):
        st.write("Regressão logística com termos quadráticos para temperatura, razão molar e catalisador, que têm "
                 "um ponto ótimo. Usa só variáveis conhecidas durante a operação, sem nenhum resultado do lote. "
                 "No teste (treino com 2024 e teste com 2025), o AUC foi de 0,94, e ao sinalizar os 20% de lotes de "
                 "maior risco o modelo pegou cerca de 82% dos lotes fora da especificação. A versão desta aba é "
                 "treinada com todos os lotes. Os dados são fictícios; numa planta real, o modelo deve ser "
                 "treinado e validado com os dados da própria empresa.")

with aba6:
    st.subheader("Lotes filtrados")
    mostrar = d.drop(columns=["mes", "fora_espec"]).sort_values("data_hora_inicio")
    st.dataframe(mostrar, hide_index=True)
    st.download_button("Baixar dados filtrados (CSV)", mostrar.to_csv(index=False).encode("utf-8-sig"),
                       "lotes_filtrados.csv", "text/csv")
    with st.expander("Como os dados foram tratados"):
        for chave, v in qualidade.items():
            st.write(f"- {chave}: {num(v)}")
        st.write("Especificação considerada: teor de ésteres de no mínimo 96,5%, acidez do produto de no máximo "
                 "0,50 mg KOH/g e viscosidade entre 3,0 e 6,0 cSt. Um lote fora de qualquer limite conta como não conforme.")

st.caption("Dados fictícios. Os padrões do painel foram criados para demonstrar o método e não representam uma planta real.")
