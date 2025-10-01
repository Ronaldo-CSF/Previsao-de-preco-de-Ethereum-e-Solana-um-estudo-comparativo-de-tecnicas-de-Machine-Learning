# %% [markdown]
# -*- coding: utf-8 -*-  <br>
# MBA Data Science e Analytics USP/ESALQ <br>
# Código para o desenvolvimento do TCC - Parte 02 <br>
# Autor: Ronaldo do Couto Silva Filho

# %% [markdown]
# 
# ### 4. Aplicação de Técnicas de Machine Learning
# Aplicar técnicas de Machine Learning para Analisar dados e fazer previsões, 
# validações e testes serão realizados na sequencia sem a devida separação em 
# tópicos

#%% In[0]: Importação de Pacotes
import os
import random
import pandas as pd  
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from time import time
from utils import formatar_grafico_mbausp, read_parquet_with_source
from pprint import pprint

# %% In[1]: Definir a seed para reprodutibilidade
seed = 0
np.random.seed(seed)
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)

# %% In[15]: Carregar a Base de Dados

# Nesse momento carregam-se as bases de dados a serem analisadas

# 30 min
sol_30min = read_parquet_with_source("../databases_parquet/sol_30min")
eth_30min = read_parquet_with_source("../databases_parquet/eth_30min")

# 1 hora
sol_1hr = read_parquet_with_source("../databases_parquet/sol_1hr")
eth_1hr = read_parquet_with_source("../databases_parquet/eth_1hr")

# 1 dia
sol_1dia = read_parquet_with_source("../databases_parquet/sol_1dia")
eth_1dia = read_parquet_with_source("../databases_parquet/eth_1dia")

# Salvar bases de dados em 1 dicionário
df_dict = {'sol_1dia': sol_1dia,
           'eth_1dia': eth_1dia,
           'sol_1hr': sol_1hr,
           'eth_1hr': eth_1hr,
           'sol_30min': sol_30min,
           'eth_30min': eth_30min}

# %% [markdown]
# 
# A partir desse ponto todo o código será desenvolvido para apenas uma cripto e
# um tamanho de base (resample), a alteração da base de dados em parquet se faz
# necessária para treinar os modelos nos respectivos casos.

# %% In[15.2]: Carregar a Base de Dados
# Selecionar qual base de dados será utilizada no momento
df = df_dict['sol_1dia']

# %% In[16]: 
# Segregar o dataset em dados de treino e teste

def segrega_data_set(df, target='close', prop_treino=0.8):
    # Proporções
    train_size = int(len(df) * prop_treino)

    # Divisão em treino e teste
    train = df[:train_size]
    test = df[train_size:]

    # Verificando as divisões
    print(f'Tamanho treino: {len(train)}, teste: {len(test)}')

    # Separa os atributos (X) e o alvo (y)
    X_train, y_train = train.drop(columns=[target]), train[target]
    X_test, y_test = test.drop(columns=[target]), test[target]

    return X_train, y_train, X_test, y_test, test

X_train, y_train, X_test, y_test, test = segrega_data_set(df)

# %% In[17]: 
# Aplicar a normalização dos dados de treino do modelo
def normaliza_dados(X_train, X_test):

    # Escalonar apenas o conjunto de treino
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # Usar o mesmo escalador nos outros conjuntos
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, scaler

X_train_scaled, X_test_scaled, scaler_X = normaliza_dados(X_train, X_test)

# ------------------------------------------------------------------------------
# %% In[18]: 
# Aplicar o Algoritmoo Naive, considerando para esse algoritmo o ultimo valor 
# presente no conjunto de validação

# Criando dicionário para salvar os resultados:
results = {}

def modela_Naive(df, test=test, target='close'):
    # Registrar tempo inicial: 
    start_time = time()

    # Criando a coluna de valores previstos
    test['predicted_naive']=test[target][0]

    # Copiando o primeiro valor real do conjunto de teste para valores previstos
    test['predicted_naive'][0]=test[target][0]

    # Calcular as nétricas de erro
    mae = mean_absolute_error(test[target].iloc[1:], 
                              test['predicted_naive'].iloc[1:])

    mse = mean_squared_error(test[target].iloc[1:], 
                                            test['predicted_naive'].iloc[1:])

    rmse = np.sqrt(mse)

    # Calculo de Tempo de Execução
    end_time = time()
    execution_time = end_time - start_time

    metricas_benchmark = [mae, mse, rmse, 'NA', execution_time]

    # Nome dinâmico da variável, ex: "sol_1dia_Naive"
    var_name = f"{df.attrs['source']}_Naive"

    # Cria um campo no dicionário results com nome de acordo com o df em estudo
    results[var_name] = metricas_benchmark

    print(f'MAE_test do Benchmark: {mae: .2f}')
    print(f'MSE_test do Benchmark: {mse: .2f}')
    print(f'RMSE_test do Benchmark: {rmse: .2f}')
    print(f'O tempo que o modelo levou para treinar foi de: {execution_time}')

    return metricas_benchmark

metricas_benchmark = modela_Naive(df)

# %% [markdown]

# ### Modelagem
# A partir desse momento serão apresentados os algoritmos a serem utilizados 
# nesse estudo. <br>
# 
# #### 1. Random Forest

# %% In[19]: 
# Aplicar o Algoritmo de Random Forest: 1º Regularização

def RF_simple_funcion(X_train_scaled = X_train_scaled, 
                      X_test_scaled = X_test_scaled, 
                      y_train = y_train, 
                      y_test = y_test,
                      mae_bench = metricas_benchmark[0]):
    """
    Hiperparâmetros utilizados para o Modelo de RF:
    Parametros = n_estimators=300, random_state=42, criterion='squared_error',
                 n_jobs=-1)
    
    rf_model = RandomForestRegressor(n_estimators=300, 
                                    random_state=42,
                                    criterion='squared_error',
                                    n_jobs=-1)
    """
    
    # Inicia a contagem de tempo para rodar o modelo:
    start_time = time()

    # Modelo Random Forest
    rf_model = RandomForestRegressor(n_estimators=300, 
                                    random_state=42,
                                    criterion='squared_error',
                                    n_jobs=-1)

    # Treinamento do modelo
    rf_model.fit(X_train_scaled, y_train)

    # Finaliza o Tempo de Execução
    end_time = time()
    execution_time = end_time - start_time

    # — Avaliação do modelo no conjunto de TREINO —
    y_train_pred = rf_model.predict(X_train_scaled)

    mae_rf_train = mean_absolute_error(y_train, y_train_pred)
    mse_rf_train = mean_squared_error(y_train, y_train_pred)
    rmse_rf_train= np.sqrt(mae_rf_train)
    mase_rf_train= mean_absolute_error(y_train, y_train_pred)/mae_bench
    execution_time = end_time - start_time

    results_RF_train = [mae_rf_train, mse_rf_train,
                        rmse_rf_train, mase_rf_train, execution_time]

    print(
        "Resultados das métricas para o modelo de RF com dados de treino são: ")
    pprint(results_RF_train)
    print()

    # Nome dinâmico da variável, ex: "sol_1dia_RF_train"
    var_name = f"{df.attrs['source']}_RF_train"

    # Cria um campo no dicionário results com nome de acordo com o df em estudo
    results[var_name] = results_RF_train

    # — Previsão do modelo no conjunto de TESTE —
    y_test_pred = rf_model.predict(X_test_scaled)

    # Avaliação no conjunto de teste
    mae_rf_test = mean_absolute_error(y_test, y_test_pred)
    mse_rf_test = mean_squared_error(y_test, y_test_pred)
    rmse_rf_test = np.sqrt(mse_rf_test)

    # Calculo do incicador MASE
    mase_rf_test = mae_rf_test/mae_bench

    results_RF_test = [mae_rf_test, mse_rf_test, rmse_rf_test, mase_rf_test,
                    execution_time]

    print("Resultados das métricas para o modelo de RF com dados de teste são: ")
    pprint(results_RF_test)

    print(f'O tempo que o modelo levou para treinar foi de: {execution_time}')

    # Salvando o resultado das métricas na respectiva variável
    var_name = f"{df.attrs['source']}_RF_teste"
    results[var_name] = results_RF_test

    return y_test_pred, rf_model

y_test_pred_RF, rf_model = RF_simple_funcion()


#%% In[20]: 
# Random Forest com GridSearch

def RF_GridSearch_function(
        X_train_scaled = X_train_scaled, 
        y_train = y_train):

    # Inicializar cronometro:
    start_time = time()

    param_grid = {
        "n_estimators": [100, 300, 500, 1000],
        "max_features": ["sqrt", None], # Como sqrt(4)=log2(4), não usado
        "max_depth": [10, 20, 100, None], 
        "min_samples_leaf":[5,10], 
        }

    rf_model = RandomForestRegressor(random_state=42)

    grid_search = GridSearchCV(estimator=rf_model, 
                            param_grid=param_grid, 
                            scoring='neg_root_mean_squared_error', 
                            cv=4, 
                            n_jobs=-1)

    grid_search.fit(X_train_scaled, y_train.values)
    
    # Finaliza o Tempo de Execução
    end_time = time()
    execution_time = end_time - start_time

    # Print the best parameters and the best score
    print(grid_search)
    print(grid_search.best_params_)
    print(grid_search.best_score_)

    print(f'O tempo que o modelo levou para treinar foi de: {execution_time}')

    print(grid_search.best_params_['max_depth'],
        grid_search.best_params_['max_features'], 
        grid_search.best_params_['min_samples_leaf'], 
        grid_search.best_params_['n_estimators'])
    
    return grid_search, execution_time

grid_search, time_GS = RF_GridSearch_function()

# Salvando em um dicionário os melhores parâmetros
best_RF_dict = {}
var_name = f"{df.attrs['source']}_best_RF_params"
best_RF_dict[var_name] = grid_search.best_params_

# Adicionando o tempo de execução aos dicionários criados com os melhores 
# parametros encontrados no GridSearchCV
best_RF_dict[var_name]["time"] = time_GS 

#%% In[20]: 
# Rodando o melhor modelo (best_RF)
def best_RF_function(X_train_scaled=X_train_scaled,
                     y_train=y_train,
                     mae_bench=metricas_benchmark[0],
                     grid_search=grid_search):

    # Inicializar cronometro:
    start_time = time()

    best_rf_model = RandomForestRegressor(
                n_estimators = grid_search.best_params_['n_estimators'],
                max_features = grid_search.best_params_['max_features'],
                max_depth = grid_search.best_params_['max_depth'],
                min_samples_leaf = grid_search.best_params_['min_samples_leaf'],
                random_state = 42,
                n_jobs = -1)

    best_rf_model.fit(X_train_scaled, y_train.values)

    # Finaliza o tempo de execução
    end_time = time()
    execution_time = end_time - start_time

    print(f'O tempo que o modelo levou para treinar foi de: {execution_time}')

    # Previsão no conjunto de treino
    y_train_pred = best_rf_model.predict(X_train_scaled)

    # Avaliação no conjunto de treino
    mae_rf_train = mean_absolute_error(y_train, y_train_pred)
    mse_rf_train = mean_squared_error(y_train, y_train_pred)
    rmse_rf_train = np.sqrt(mse_rf_train)
    mase_rf_train = mae_rf_train/mae_bench

    # Salvando valores de treino
    results_best_RF_train = [mae_rf_train, mse_rf_train, rmse_rf_train,
                         mase_rf_train, execution_time]

    var_name = f"{df.attrs['source']}_best_RF_train"
    results[var_name] = results_best_RF_train
    
    print("Resultados das métricas para o melhor modelo de RF com dados " \
        "de treino são: ")
    pprint(results_best_RF_train)
    print()
    
    # -------------------------------------------------------------------------

    # Previsão no conjunto de teste
    y_test_pred = best_rf_model.predict(X_test_scaled)

    # Avaliação no conjunto de teste
    mae_rf_test = mean_absolute_error(y_test, y_test_pred)
    mse_rf_test = mean_squared_error(y_test, y_test_pred)
    rmse_rf_test = np.sqrt(mse_rf_test)
    mase_rf_test = mae_rf_test/mae_bench

    # Salvando valores de treino
    results_best_RF_test = [mae_rf_test, mse_rf_test, rmse_rf_test,
                         mase_rf_test, execution_time]
    
    var_name = f"{df.attrs['source']}_best_RF_test"
    results[var_name] = results_best_RF_test
    
    print("Resultados das métricas para o melhor modelo de RF com dados " \
        "de teste são: ")
    pprint(results_best_RF_test)
    print()

    return y_test_pred, y_train_pred

y_test_pred_best_RF,  y_train_pred_best_RF= best_RF_function()

# %% [markdown]
# # 6. Interpretação dos Resultados:

# %% In[21]: Visualizando os resultados do modelo:

# 1. Vizualizar resultados de Naive

# Concatenando valores reais e previstos
y_real = pd.concat([y_train, y_test])
y_pred = pd.concat([
    pd.Series(test["predicted_naive"], index=test.index)
])

# Organizando as séries para o gráfico
series_plot = {
    "Valores Reais": (y_real.index, y_real),
    "Previsão Naive": (y_pred.index, y_pred)
}

# Chamando a função com estilos e salvando automaticamente
moeda = f"{df.attrs['source']}".split("_")[0].upper()
moeda = moeda + "USDT"
eixo_y = f"Fechamento ({moeda})"

nome_arquivo = f"{df.attrs['source']}_Naive_forecast"

formatar_grafico_mbausp(
    series = series_plot,
    # titulo="Série Temporal: Original vs Previsões (Random Forest)",
    titulo_x = "Timestamp",
    titulo_y = eixo_y,
    salvar = True,
    nome_arquivo = nome_arquivo,  
    formato = "jpg",
    cores = ["blue", "orange"],
    legenda = True,
    vertical = True
)

# 2. Vizualizar resultados de RF

# Concatenando valores reais e previstos
y_real = pd.concat([y_train, y_test])
y_pred = pd.concat([
    pd.Series(y_test_pred_RF, index=y_test.index)
])

# Organizando as séries para o gráfico
series_plot = {
    "Valores Reais": (y_real.index, y_real),
    "Previsão RF Simples": (y_pred.index, y_pred)
}

# Chamando a função com estilos e salvando automaticamente
moeda = f"{df.attrs['source']}".split("_")[0].upper()
moeda = moeda + "USDT"
eixo_y = f"Fechamento ({moeda})"

nome_arquivo = f"{df.attrs['source']}_RF_forecast"

formatar_grafico_mbausp(
    series = series_plot,
    # titulo="Série Temporal: Original vs Previsões (Random Forest)",
    titulo_x = "Timestamp",
    titulo_y = eixo_y,
    salvar = True,
    nome_arquivo = nome_arquivo,  
    formato = "jpg",
    cores = ["blue", "orange"],
    legenda = True,
    vertical = True
)

# 3. Vizualizar resultados de best RF
 
# Concatenando valores reais e previstos
y_real = pd.concat([y_train, y_test])
y_pred = pd.concat([
    pd.Series(y_test_pred_best_RF, index=y_test.index)
])

# Organizando as séries para o gráfico
series_plot = {
    "Valores Reais": (y_real.index, y_real),
    "Previsão RF Otimizado": (y_pred.index, y_pred)
}

# Chamando a função com estilos e salvando automaticamente
moeda = f"{df.attrs['source']}".split("_")[0].upper()
moeda = moeda + "USDT"
eixo_y = f"Fechamento ({moeda})"

nome_arquivo = f"{df.attrs['source']}_best_RF_forecast"

formatar_grafico_mbausp(
    series = series_plot,
    # titulo="Série Temporal: Original vs Previsões (Random Forest)",
    titulo_x = "Timestamp",
    titulo_y = eixo_y,
    salvar = True,
    nome_arquivo = nome_arquivo,  
    formato = "jpg",
    cores = ["blue", "orange"],
    legenda = True,
    vertical = True
)

# %% In[22]: Salvando os resultados obtidos:
os.makedirs("../Resultados_Naive_RF", exist_ok=True)
caminho_results = os.path.join("../Resultados_Naive_RF", f"Naive_RF-{df.attrs['source']}.csv")

arquivo = pd.DataFrame(results)
arquivo.to_csv(caminho_results, index = False)

caminho_param = os.path.join("../Resultados_Naive_RF", f"Best_param-{df.attrs['source']}.csv")
best_hiperpar = pd.DataFrame(best_RF_dict)
best_hiperpar.to_csv(caminho_param, index = False)