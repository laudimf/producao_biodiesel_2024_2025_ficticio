# Painel de processo — Química Horizonte

Projeto de portfólio em análise de dados para a indústria química. Painel interativo feito com Python e Streamlit, a partir de dados de lotes de uma planta fictícia de biodiesel.

**A empresa e todos os dados são fictícios.** Foram criados só para demonstração, e os padrões do painel não representam uma planta real.

## O que o painel mostra

- Indicadores: lotes, rendimento médio, lotes fora da especificação, biodiesel produzido e variação da temperatura
- Rendimento mensal por reator e lotes fora da especificação por mês
- Desgaste dos reatores em função do tempo desde a última limpeza
- Comparação entre fornecedores de óleo (rendimento, acidez e umidade)
- Efeito da temperatura de reação e do turno de operação
- Filtros por período, reator, fornecedor e turno, e download dos dados filtrados

## Como rodar no seu computador

```
pip install -r requirements.txt
streamlit run app_processo.py
```

## Tratamento dos dados

Remoção de lotes duplicados, padronização do nome dos reatores, exclusão de temperaturas com falha de sensor e de rendimentos acima de 100%, e preenchimento da umidade do óleo em branco pela mediana do fornecedor.

Especificação considerada: teor de ésteres de no mínimo 96,5%, acidez do produto de no máximo 0,50 mg KOH/g e viscosidade entre 3,0 e 6,0 cSt.

## Tecnologias

Python, pandas, Streamlit, Altair.
