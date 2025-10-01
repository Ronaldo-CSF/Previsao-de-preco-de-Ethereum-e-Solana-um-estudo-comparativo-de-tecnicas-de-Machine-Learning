# %% [markdown]
# -*- coding: utf-8 -*-  <br>
# MBA Data Science e Analytics USP/ESALQ <br>
# Código para o desenvolvimento do TCC - Funções Auxiliares <br>
# Autor: Ronaldo do Couto Silva Filho

import os
from typing import Union, Tuple, Dict, List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Função criada para ler o dataframe em formato parquet e salvar o titulo
def read_parquet_with_source(path, **kwargs):
    df = pd.read_parquet(path, **kwargs)
    temp_path = path.split("/")[2]
    df.attrs['source'] = temp_path    # usado para recuperar o titulo do df de origem
    return df


# Função criada para plotar gráficos de linha de acordo com as Normas 
# da USP ESALQ (2025)
Dado1D = Union[List[float], np.ndarray, pd.Series]

def formatar_grafico_mbausp(
    series: Union[
        Tuple[Dado1D, Dado1D],  # Para uma única série
        Dict[str, Tuple[Dado1D, Dado1D]]  # Para múltiplas séries com rótulo
    ],
    titulo_x: str = "Eixo X",
    titulo_y: str = "Eixo Y",
    titulo: Union[str, bool] = False,
    legenda: bool = False,
    salvar: bool = False,
    nome_arquivo: str = "grafico",
    formato: str = "jpg",
    cores: List[str] = ["#1e81b0", 'orange'],
    vertical: bool = False,
    linestyle: Union[str, Dict[str, str]] = 'solid'
) -> None:

    """
    Plotar séries temporais com formato padrão MBA USP e permite salvar em arquivo.

    Parâmetros:
    - series: 
        - tuple: (x, y) para uma única série
        - dict: {"Nome da Série": (x, y)} para múltiplas séries
    - titulo_x, titulo_y: str, rótulos dos eixos
    - titulo: str ou False, título do gráfico
    - legenda: bool, exibir legenda
    - salvar: bool, se True, salva o gráfico
    - nome_arquivo: str, nome do arquivo (sem extensão)
    - formato: 'jpg' ou 'pdf'
    - cores: List[str], string com a cor desejada
    - verticar: bool, se True, cria linha vertical para separar a partir de onde
    a série concatenada parte.
    linestyle: str, estilo da linha utilizado
    """

    # Criar a figura e o eixo
    fig, ax = plt.subplots(figsize=(8, 8))

    # Se for uma série só, transforma em dicionário
    if isinstance(series, tuple):
        series = {"Série Temporal": series}

    # Cores usadas para representar os diferentes gráficos presentes:
    cores = cores # Lista com as cores a serem usadas nos gráficos

    for i, (nome, (dados_x, dados_y)) in enumerate(series.items()):
        dados_x = pd.to_datetime(dados_x)
        if isinstance(dados_y, pd.Series):
            dados_y = dados_y.values

        # Define cor e estilo de linha
        cor = cores[i % len(cores)]
        estilo = linestyle[nome] if isinstance(linestyle, dict) else linestyle

        ax.plot(dados_x, dados_y, linewidth=1.5, label=nome, linestyle=estilo, color=cor)


    # Estilo MBA USP
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)
    ax.spines['left'].set_color('black')
    ax.spines['bottom'].set_color('black')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(False)

    ax.set_xlabel(titulo_x, fontsize=11/2.54, fontname="Arial", color="k")
    ax.set_ylabel(titulo_y, fontsize=11/2.54, fontname="Arial", color="k")

    # Linha vertical se houver duas séries (eixo x do segundo item)
    if len(series) > 1 and vertical == True:
        # Tenta encontrar o primeiro ponto do segundo conjunto
        lista_series = list(series.values())
        x2 = pd.to_datetime(lista_series[1][0])
        if len(x2) > 0:
            ax.axvline(x=x2[0], color='gray', linestyle='--', linewidth=1.2)


    if legenda:
        ax.legend()
    if titulo:
        plt.title(titulo)

    # plt.xticks(rotation=45) se necessário rotacionar o eixo X
    plt.tight_layout()

    # Salvar em arquivo
    if salvar:
        # Cria pasta se não existir
        os.makedirs("../graficos", exist_ok=True)
        caminho = os.path.join("../graficos", f"{nome_arquivo}.{formato}")
        plt.savefig(caminho, dpi=300, format=formato)
        print(f"Gráfico salvo como: {caminho}")

    plt.show()

