# %% [markdown]
# -*- coding: utf-8 -*-  <br>
# MBA Data Science e Analytics USP/ESALQ <br>
# Código para o desenvolvimento do TCC - Parte 01 <br>
# Autor: Ronaldo do Couto Silva Filho

# %% [markdown]
# # Previsão de preço de Ethereum e Solana um estudo comparativo de técnicas de 
# Machine Learning

# %% [markdown]
# ## Objetivo
# 
# O presente notebook tem como objetivo demonstrar a implementação de um 
# algoritmo de machine learnig que possa auxiliar a tomada de decisão na compra 
# de criptoativos, por meio da análise de dados históricos: preço, permitindo 
# identificar padrões e prever movimentações de mercado.
# 
# Para tal será realizada uma análise comparativa da precisão de previsões dos 
# preços das criptomoedas Ether e Sol, utilizando três técnicas distintas de 
# aprendizado de máquina: Random Forest [RF], LSTM e GRU. Pretende-se avaliar a 
# eficiência e acurácia de cada modelo nas previsões, por meio das métricas de 
# Erro Quadrático Médio [MSE], Raiz do Erro Quadrático Médio [RMSE], 
# Erro Médio Absoluto Escalado [MASE], proporcionando uma compreensão sobre o 
# desempenho de cada abordagem.
# 
# 
# ## Estrutura do Trabalho
# 
# 1. Coleta de Dados
# 2. Análise Explorativa de Dados
# 3. Pré-processamento dos Dados
# 4. Aplicação de Técnicas de Machine Learning
# 5. Validação e Teste dos Modelos
# 6. Interpretação dos Resultados

# %% [markdown]
# ### 1. Coleta de Dados
# Recolher dados históricos de preços de diferentes criptoativos, volume de 
# negociação, entre outras variáveis relevantes.

# %% [markdown]
# Os dados utilizados nesse trabalho foram extraidos do notebook presente no 
# Kaggle: 
# [The Most 50 Popular Crypto Data](https://www.kaggle.com/datasets/kaanxtr/btc-price-1m?resource=download).

# %% [markdown]
# 
# ### 2. Análise Explorativa de Dados:
# Realizar uma análise inicial dos dados para entender melhor os padrões 
# históricos e a correlação entre diferentes criptoativos.  
# Utilizar visualizações gráficas e estatísticas descritivas para identificar 
# tendências e anomalias.
# 

#%% In[0]: 
# Importação de Pacotes
import os
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.metrics import mean_squared_error
from utils import formatar_grafico_mbausp
from statsmodels.graphics.tsaplots import plot_acf

#%% In[1]: 
# Importar base de dados
eth_data = pd.read_csv("../databases_origin/ETHUSDT.csv")
sol_data = pd.read_csv("../databases_origin/SOLUSDT.csv")

#%%  In[2]:
# Visualização das variáveis presentes no dataset
print(eth_data.keys(),sol_data.keys(), sep="\n\n")

#%% In[3.1]: 
# Visualização das 6 primeiras observações
eth_data.head()

#%% In[3.2]:
# Visualização das estatisticas associadas em cada dataset
sol_data.describe() 

#%% In[4]: 
# Selecionar a Cripto a ser analisada
cripto_lista = {"eth":eth_data, "sol":sol_data}

# %% [markdown]
# Nesse momento do código é definido qual a criptomoeda a ser utilizada para 
# criar as bases de dados necessárias. O código foi escrito utilizando como 
# referência solana, cujo nome está sendo referido no dicionário cripto_lista, 
# como **"sol"**. O presente código se aplica a demais criptomoedas, no entanto,
# para utilizá-lo é necessário importar os dados seguindo a estrutura acima. <br>
# 
# **Procedimento**:  <br>
# Salve a base de dados a ser utilizada na pasta databases, no formato .csv <br>
# OBS: Se o seu dataset estiver em outro formato adapte o código tanto quanto 
# necessário.  <br>
#  
# Rode o código abaixo adaptando o NomeDaCriptomoeda pela referida cripto, 
# como por exemplo BTC <br>
#  
# `NomeDaCriptomoeda_data = pd.read_csv("databases/NomeDaCriptomoedaUSDT.csv")
# cripto_lista = {"NomeDaCriptomoeda": NomeDaCriptomoeda_data}`

#%% In[5]: 
# Selecionar a Criptomoeda e as variáveis que serão analisadas
cripto_name = "eth"

cripto = cripto_lista[cripto_name.lower()][['timestamp', 'close']]

cripto['timestamp'] = pd.to_datetime(cripto['timestamp'])

cripto.set_index('timestamp', inplace=True)

#%% In[6]: 
# Procurar por valores nulos:
null_values = cripto.isna().sum()
print(null_values)

#%% In[7]: 
# Criação de Objeto TimeSeries (mais leve e rápido)
cripto_ts = pd.Series(data=cripto['close'].values,index=cripto.index)

print(cripto_ts.head())  # Verifique os primeiros valores

cripto_ts.describe()

#%% In[8]:
titulo_y = f'Fechamento ({cripto_name.upper()}USDT)'
# Grafico da série temporal
formatar_grafico_mbausp((cripto_ts.index, cripto_ts),
                        titulo_x="Timestamp",
                        titulo_y=titulo_y)

# %% [markdown]
# > Com o objetivo de estudar intervalos mais espaçados de dados, foram criados
# resamples com intervalos de 30 a 30 minutos, hora em hora e dia a dia.

#%% In[9]: 
# Realizando o resamplying dos dados

# 30 min
cripto_30min = cripto.resample('30T').mean()
print(cripto_30min)

# 1 hora
cripto_1hr = cripto.resample('1H').mean()
print(cripto_1hr)

# 1 dia
cripto_1dia = cripto.resample('1D').mean()
print(cripto_1dia)

# %% [markdown]
#
# > Com o resample dos dados é possível que tenham sido criados intervalos com 
# valores nulos, e para corrigir tal equívoco (devido a falta de dados) como a 
# cotação/preço de um ativo é o último valor por ele, os valores nulos foram 
# preenchidos usando o método forwardfill.

#%% In[10]: 
# Lista de resamples
resample_cripto = [cripto, cripto_30min, cripto_1hr, cripto_1dia]

for df in resample_cripto:
    # Preenchendo os valores de fechamento com o método de forwarfill
    df["close"] = df["close"].fillna(method='ffill')


#%% In[11]: 
# Decomposição de Séries Temporais

# Decomposicao pelo modelo ADITIVO
cripto_decomp_add = seasonal_decompose(cripto["close"], model='aditive',
                                    period=60*24*30*12)

# Decomposicao pelo modelo MULTIPLICATIVO
cripto_decomp_mult = seasonal_decompose(cripto["close"], model='multiplicative',
                                    period=60*24*30*12)

#%% In[12]: 
# Analisar o resíduo para selecionar o melhor modelo
# Modelo Aditivo
reconstruida_add = cripto_decomp_add.trend + cripto_decomp_add.seasonal

# Modelo Multiplicativo
reconstruida_mult = cripto_decomp_mult.trend + cripto_decomp_mult.seasonal

# Remover NaNs causados pela decomposição
# Modelo Aditivo
validos_add = reconstruida_add.dropna()
original = cripto["close"].loc[validos_add.index]

# Modelo Multiplicativo
validos_mult = reconstruida_mult.dropna()
original = cripto["close"].loc[validos_mult.index]

# Calculo do RMSE
rmse_add = mean_squared_error(original, validos_add)
rmse_mult = mean_squared_error(original, validos_mult)

print(f'RMSE do modelo ADITIVO: {rmse_add:.2f}')
print(f'RMSE do modelo MULTIPLICATIVO: {rmse_mult:.2f}')

# %% [markdown]
#
# > Ambos os modelos apresentam um alto RMSE, não representando 
# significativamente a série temporal. O modelo ADITIVO, no entanto, apresenta 
# melhor resultado.

#%% In[12.1]:
# Avaliação do resíduos do modelo aditivo
residuos_add = cripto_decomp_add.resid.dropna()

print("Modelo Aditivo:")

plot_acf(residuos_add.tail(100_000), lags=50)
plt.show()

#%% In[12.2]:
# Avaliação do resíduos do modelo multiplicativo
residuos_mult = cripto_decomp_mult.resid.dropna()

print("Modelo Multiplicativo:")

plot_acf(residuos_mult.tail(100_000), lags=50)
plt.show()

# %% [markdown]
# > Por restrições computacionais foi optado por plotar a curva da função de 
# autocorrelação com o método `plot_acf`. É possível observar a alta correlação
# dos pontos residuais. Dessa forma, é evidente que a decomposição sazonal, por
# si só não é um bom modelo ajustado a cotações de criptoativos, e para tal, em 
# seguida usaremos técnicas mais sofisticadas para modelar seu comportamento.

# In[12]: Plotar a decomposicao
# Plotar decomposição aditiva da série temporal
# Reconstrução da série (sem o resíduo)
plt.figure(figsize=(10, 8))
plt.subplot(4, 1, 1)
plt.plot(cripto_decomp_add.trend)
plt.title('Tendencia')

plt.subplot(4, 1, 2)
plt.plot(cripto_decomp_add.seasonal)
plt.title('Componente Sazonal')

plt.subplot(4, 1, 3)
plt.plot(cripto_decomp_add.resid)
plt.title('Residuos')

plt.subplot(4, 1, 4)
plt.plot(cripto_1hr["close"], label='Original')
plt.plot(cripto_decomp_add.trend + cripto_decomp_add.seasonal, 
         label='Reconstruida')

plt.title('Original vs. Reconstruida')
plt.legend()

plt.tight_layout()
plt.show()

# %% [markdown]
# ### 3. Pré-processamento dos Dados
#
# Criar features e processar os dados necessários para dar início ao processo de
# análise dos dados

# %% In[13]: Criação de Features Adicionais:

# Criação de Feature adicionais
# Random Forest não lida diretamente com a dependência temporal, como outros 
# modelos # especializados em séries temporais (ARIMA, LSTM). Para contornar 
# isso, é necessário criar variáveis explicativas (X) a partir de transformações
# nos dados, como janelas deslizantes, indicadores técnicos ou agregados 
# temporais. Para esse trabalho foi escolhido criar indicadores técnicos e agre-
# gados temporais

for df in resample_cripto:

    # Retornos percentuais (% variação do preço).
    df['return_pct'] = df['close'].pct_change() * 100

    # Médias móveis (curta e longa duração).

    # Média móvel simples (janela de 10 períodos)
    df['SMA_10'] = df['close'].rolling(window=10).mean()  
    
    # Média móvel exponencial (janela de 10 períodos)
    df['EMA_10'] = df['close'].ewm(span=10, adjust=False).mean()  

    # Índice de Força Relativa (RSI - Relative Strength Index)
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()  #Ganhos médios
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean() #Perdas médias
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    df.dropna(inplace=True)

# %% In[14]: Criação das Bases de Dados

# Código para Criação das Bases de Dados, a ser alterado para cada criptomoeda
os.makedirs("../databases_parquet", exist_ok=True)

if cripto_name == "sol":
    cripto_30min.to_parquet("../databases_parquet/sol_30min")
    cripto_1hr.to_parquet("../databases_parquet/sol_1hr")
    cripto_1dia.to_parquet("../databases_parquet/sol_1dia")
    cripto.to_parquet("../databases_parquet/sol_1min")

elif cripto_name == "eth":
    cripto_30min.to_parquet("../databases_parquet/eth_30min")
    cripto_1hr.to_parquet("../databases_parquet/eth_1hr")
    cripto_1dia.to_parquet("../databases_parquet/eth_1dia")
    cripto.to_parquet("../databases_parquet/eth_1min")
