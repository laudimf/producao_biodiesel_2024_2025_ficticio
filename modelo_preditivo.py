#!/usr/bin/env python3
"""
Modelo preditivo — biodiesel (dados fictícios).

Pergunta 1: o lote vai sair fora da especificação?  (classificação)
Pergunta 2: qual será o rendimento do lote?          (regressão)

Compara: regressão simples, regressão com termos quadráticos,
PCA + regressão, PLS e um modelo não linear (gradient boosting) como referência.

Validação no tempo: treina com 2024 e testa em 2025.
Usa só variáveis conhecidas ANTES do fim do lote (nada do resultado do lote).

Uso:  python modelo_preditivo.py producao_biodiesel_2024_2025_ficticio.csv
Bibliotecas: pandas, numpy, scikit-learn.
"""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import average_precision_score, mean_squared_error, r2_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# Variáveis conhecidas durante a operação (entradas do processo).
# Ficam de fora: rendimento, teor de ésteres, acidez do produto, viscosidade e massa de biodiesel,
# que são RESULTADOS do lote e dariam a resposta ao modelo (vazamento de dados).
NUM = ["massa_oleo_kg", "acidez_oleo", "umidade_oleo", "temp_reacao_C", "razao_molar",
       "catalisador_pct", "tempo_min", "agitacao_rpm", "dias_desde_limpeza"]
CAT = ["reator", "turno", "fornecedor"]


class PLSDA(BaseEstimator, ClassifierMixin):
    """PLS para classificação: regride o alvo 0/1 e usa o resultado como pontuação de risco."""
    def __init__(self, n_components=2):
        self.n_components = n_components

    def fit(self, X, y):
        self.classes_ = np.array([0, 1])
        self.pls_ = PLSRegression(n_components=self.n_components, scale=False).fit(X, y)
        return self

    def decision_function(self, X):
        return self.pls_.predict(X).ravel()


def carregar(caminho):
    df = pd.read_csv(caminho, encoding="utf-8-sig", parse_dates=["data_hora_inicio"])
    df = df.drop_duplicates("lote_id")
    df["reator"] = df["reator"].astype(str).map(lambda s: "R" + re.sub(r"\D", "", s))
    df.loc[(df["temp_reacao_C"] < 30) | (df["temp_reacao_C"] > 90), "temp_reacao_C"] = np.nan
    df.loc[df["rendimento_pct"] > 100, "rendimento_pct"] = np.nan
    df["umidade_oleo"] = df["umidade_oleo"].fillna(df.groupby("fornecedor")["umidade_oleo"].transform("median"))
    df["fora_espec"] = (df["conforme"] == "Não").astype(int)
    return df.dropna(subset=NUM).sort_values("data_hora_inicio").reset_index(drop=True)


def montar(df, quadraticos):
    X = df[NUM].copy()
    X = X.join(pd.get_dummies(df[CAT], drop_first=True, dtype=float))
    if quadraticos:
        X["temp_c2"] = (df["temp_reacao_C"] - 62) ** 2
        X["razao_c2"] = (df["razao_molar"] - 7.8) ** 2
        X["catal_c2"] = (df["catalisador_pct"] - 1.0) ** 2
    return X


def topo20(y, score):
    """Precisão e cobertura se o cliente sinalizar os 20% de lotes de maior risco."""
    corte = np.quantile(score, 0.80)
    sinal = score >= corte
    return y[sinal].mean(), y[sinal].sum() / y.sum()


def main():
    caminho = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("producao_biodiesel_2024_2025_ficticio.csv")
    if not caminho.exists():
        sys.exit(f"Erro: arquivo não encontrado: {caminho}")
    df = carregar(caminho)
    treino = df["data_hora_inicio"] < "2025-01-01"
    teste = ~treino
    cv = TimeSeriesSplit(n_splits=4)
    print(f"Lotes: treino (2024) = {treino.sum()} | teste (2025) = {teste.sum()}")
    print(f"Fora da especificação: treino {df.loc[treino,'fora_espec'].mean()*100:.1f}% | teste {df.loc[teste,'fora_espec'].mean()*100:.1f}%")

    Xb, Xq = montar(df, False), montar(df, True)

    # ---------- Classificação
    y = df["fora_espec"].values
    modelos = {
        "1. Regressão logística (variáveis originais)": (Xb, make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))),
        "2. Regressão logística + termos quadráticos": (Xq, make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))),
        "3. PCA + regressão logística": (Xq, GridSearchCV(
            make_pipeline(StandardScaler(), PCA(), LogisticRegression(max_iter=2000)),
            {"pca__n_components": list(range(2, Xq.shape[1] + 1))}, scoring="roc_auc", cv=cv)),
        "4. PLS-DA": (Xq, GridSearchCV(make_pipeline(StandardScaler(), PLSDA()),
                                      {"plsda__n_components": list(range(1, 9))}, scoring="roc_auc", cv=cv)),
        "5. Gradient boosting (referência não linear)": (Xb, HistGradientBoostingClassifier(random_state=0)),
    }
    print("\n=== PREVER SE O LOTE SAI FORA DA ESPECIFICAÇÃO (2025) ===")
    print(f"{'modelo':<48}{'AUC':>7}{'AP':>7}{'prec.20%':>10}{'cobert.20%':>12}")
    for nome, (X, m) in modelos.items():
        m.fit(X[treino], y[treino])
        if hasattr(m, "predict_proba"):
            s = m.predict_proba(X[teste])[:, 1]
        else:
            s = m.decision_function(X[teste])
        pr, cb = topo20(y[teste], s)
        extra = ""
        if isinstance(m, GridSearchCV):
            extra = "  (" + ", ".join(f"{k.split('__')[1]}={v}" for k, v in m.best_params_.items()) + ")"
        print(f"{nome:<48}{roc_auc_score(y[teste], s):7.3f}{average_precision_score(y[teste], s):7.3f}{pr*100:9.1f}%{cb*100:11.1f}%{extra}")
    base = y[teste].mean()
    print(f"(referência: sem modelo, AUC = 0,500 e precisão = {base*100:.1f}%)")

    # Principais fatores no modelo 2 (coeficientes padronizados)
    m2 = modelos["2. Regressão logística + termos quadráticos"][1]
    coef = pd.Series(m2[-1].coef_[0], index=Xq.columns).sort_values(key=np.abs, ascending=False).head(6)
    print("\nPrincipais fatores no modelo 2 (coeficiente padronizado; positivo = mais risco):")
    print(coef.round(2).to_string())

    # ---------- Regressão (rendimento)
    ok = df["rendimento_pct"].notna().values
    yr = df["rendimento_pct"].values
    tr, te = (treino & ok).values, (teste & ok).values
    reg = {
        "1. Regressão linear (variáveis originais)": (Xb, make_pipeline(StandardScaler(), LinearRegression())),
        "2. Regressão linear + termos quadráticos": (Xq, make_pipeline(StandardScaler(), LinearRegression())),
        "3. PCA + regressão linear": (Xq, GridSearchCV(make_pipeline(StandardScaler(), PCA(), LinearRegression()),
                                                       {"pca__n_components": list(range(2, Xq.shape[1] + 1))},
                                                       scoring="neg_root_mean_squared_error", cv=cv)),
        "4. PLS": (Xq, GridSearchCV(make_pipeline(StandardScaler(), PLSRegression(scale=False)),
                                    {"plsregression__n_components": list(range(1, 9))},
                                    scoring="neg_root_mean_squared_error", cv=cv)),
        "5. Gradient boosting (referência não linear)": (Xb, HistGradientBoostingRegressor(random_state=0)),
    }
    print("\n=== PREVER O RENDIMENTO DO LOTE (2025) ===")
    print(f"{'modelo':<48}{'R²':>7}{'erro (p.p.)':>13}")
    for nome, (X, m) in reg.items():
        m.fit(X[tr], yr[tr])
        p = np.ravel(m.predict(X[te]))
        extra = ""
        if isinstance(m, GridSearchCV):
            extra = "  (" + ", ".join(f"{k.split('__')[1]}={v}" for k, v in m.best_params_.items()) + ")"
        print(f"{nome:<48}{r2_score(yr[te], p):7.3f}{np.sqrt(mean_squared_error(yr[te], p)):13.2f}{extra}")

    # Quanto o PCA realmente reduz neste caso
    pca = make_pipeline(StandardScaler(), PCA()).fit(Xq[treino])
    var = np.cumsum(pca[-1].explained_variance_ratio_)
    print(f"\nPCA: {Xq.shape[1]} variáveis; são necessários {int((var < 0.95).sum()) + 1} componentes para 95% da variância.")


if __name__ == "__main__":
    main()
