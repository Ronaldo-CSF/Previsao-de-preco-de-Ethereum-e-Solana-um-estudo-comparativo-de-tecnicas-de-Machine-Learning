# %% In[0]: Importação de Bibliotecas

# 1. Importação das Bibliotecas
import os
import random
from time import time
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, root_mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

# %% In[1]: Definir a seed para reprodutibilidade

seed = 0
np.random.seed(seed)
tf.random.set_seed(seed)
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)

initializer = tf.keras.initializers.GlorotUniform(seed=seed)

tf.config.experimental.enable_op_determinism()

# %% In[2]: Carga e Processamento dos Dados

# 2. Carregar e Pré-processar os Dados

# 🔹Para um DataFrame que se chama "df" e contém as colunas:
# timestamp, close, volume, number_of_trades, return_pct, SMA_10, EMA_10, RSI

df = pd.read_parquet("sol_1dia")

features = ['return_pct', 'SMA_10', 'EMA_10', 'RSI']  # Variáveis preditoras
target = ['close']  # Variável a ser prevista

# 🔹 Criar DataFrames separados
X = df[features]
y = df[target]


# 🔹 Separar os dados em 60% treino, 20% validação e 20% teste
train_size = int(len(df) * 0.6)
val_size = int(len(df) * 0.2)

X_train, X_val, X_test = X[:train_size], X[train_size:train_size + val_size], X[train_size + val_size:]
y_train, y_val, y_test = y[:train_size], y[train_size:train_size + val_size], y[train_size + val_size:]

# 🔹 Normalizar cada parte SEPARADAMENTE para evitar data leakage
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_train_scaled = scaler_X.fit_transform(X_train)
X_val_scaled = scaler_X.transform(X_val)
X_test_scaled = scaler_X.transform(X_test)

y_train_scaled = scaler_y.fit_transform(y_train)
y_val_scaled = scaler_y.transform(y_val)
y_test_scaled = scaler_y.transform(y_test)

# %% In[2.x]: Aplicar o Algoritmoo Naive, considerando para esse algoritmo
# o ultimo valor presente no conjunto de validação

# Tempo inicial
start_time = time()

# Criando a coluna de valores previstos
y_test['predicted_price_BM2']=y_test['close'][0]

# Copiando o primeiro valor real do conjunto de teste para valores previstos
y_test['predicted_price_BM2'][0]=y_test['close'][0]

# Calcular o erro (exemplo: MAE)
mae_benchmark_test = mean_absolute_error(y_test['close'].iloc[1:], y_test['predicted_price_BM2'].iloc[1:])

mse_benchmark_test = mean_squared_error(y_test['close'].iloc[1:], y_test['predicted_price_BM2'].iloc[1:])
rmse_benchmark_test = np.sqrt(mse_benchmark_test)

print(f'MAE_test do Benchmark: {mae_benchmark_test: .2f}')
print(f'MSE_test do Benchmark: {mse_benchmark_test: .2f}')
print(f'RMSE_test do Benchmark: {rmse_benchmark_test: .2f}')

# Calculo de Tempo de Execução
end_time = time()
execution_time = end_time - start_time

if df.equals(pd.read_parquet("sol_1dia")) or df.equals(pd.read_parquet("eth_1dia")):
    Naive_result_dia = [mae_benchmark_test, mse_benchmark_test, rmse_benchmark_test,'NA',
                execution_time]
elif df.equals(pd.read_parquet("sol_1hr")) or df.equals(pd.read_parquet("eth_1hr")):
    Naive_result_hr = [mae_benchmark_test, mse_benchmark_test, rmse_benchmark_test,'NA',
                execution_time]
elif df.equals(pd.read_parquet("sol_30min")) or df.equals(pd.read_parquet("eth_30min")):
    Naive_result_30min = [mae_benchmark_test, mse_benchmark_test, rmse_benchmark_test,'NA',
                execution_time]
elif df.equals(pd.read_parquet("sol_1min")) or df.equals(pd.read_parquet("eth_1min")):
    Naive_result_1min = [mae_benchmark_test, mse_benchmark_test, rmse_benchmark_test,'NA',
                execution_time]

print(f'O tempo que o modelo levou para treinar foi de: {execution_time}')

# %% In[3]: Criação de Janelas Deslizantes

# 3. Criar Janelas Deslizantes

def create_sequences(X_data, y_data, time_steps=15):
    """Cria janelas deslizantes para séries temporais."""
    X_seq, y_seq = [], []
    for i in range(len(X_data) - time_steps):
        X_seq.append(X_data[i:i + time_steps])  # 15 passos de entrada
        y_seq.append(y_data[i + time_steps])  # Próximo valor de "close"
    return np.array(X_seq), np.array(y_seq)

# 🔹 Criar janelas para treino, validação e teste
X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train_scaled)
X_val_seq, y_val_seq = create_sequences(X_val_scaled, y_val_scaled)
X_test_seq, y_test_seq = create_sequences(X_test_scaled, y_test_scaled)

# 🔹 Ajustar formato para redes neurais
X_train_seq = X_train_seq.reshape(X_train_seq.shape[0], X_train_seq.shape[1], X_train_seq.shape[2])
X_val_seq = X_val_seq.reshape(X_val_seq.shape[0], X_val_seq.shape[1], X_val_seq.shape[2])
X_test_seq = X_test_seq.reshape(X_test_seq.shape[0], X_test_seq.shape[1], X_test_seq.shape[2])

# %% In [aaa]:

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam

def build_lstm_model(units=50, learning_rate=0.001, input_shape=(15, 5)):  # ajuste input_shape se necessário
    model = Sequential()
    model.add(LSTM(units=units, activation='relu', return_sequences=True, input_shape=input_shape))
    model.add(LSTM(units=units, activation='relu'))
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mean_squared_error')
    return model

from scikeras.wrappers import KerasRegressor

input_shape = (X_train_seq.shape[1], X_train_seq.shape[2])

lstm_regressor = KerasRegressor(
    model=build_lstm_model,
    model__input_shape=input_shape,
    verbose=0
)
from sklearn.model_selection import GridSearchCV

param_grid = {
    "model__units": [32, 64],
    "model__learning_rate": [0.001, 0.0005],
    "batch_size": [32],
    "epochs": [100]
}

grid_lstm = GridSearchCV(
    estimator=lstm_regressor,
    param_grid=param_grid,
    cv=3,
    scoring='neg_mean_squared_error'
)

grid_lstm_result = grid_lstm.fit(X_train_seq, y_train_seq)




# %%
def build_lstm_model(units=100, learning_rate=0.001, **kwargs):
    model = Sequential()
    model.add(LSTM(units=units, activation='relu', return_sequences = True, 
                   input_shape= X_train )) #(X_train_seq.shape[1], X_train_seq.shape[2])))
    model.add(LSTM(units=units, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mean_squared_error')
    return model

def build_gru_model(units=50, learning_rate=0.001, **kwargs):
    model = Sequential()
    model.add(GRU(units=units, activation='relu', return_sequences = True, input_shape=(X_train_seq.shape[1], X_train_seq.shape[2])))
    model.add(GRU(units=units, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mean_squared_error')
    return model

# %%

from scikeras.wrappers import KerasRegressor


lstm_regressor = KerasRegressor(
    model=build_lstm_model,
    model__X_shape=(X_train_seq.shape[1], X_train_seq.shape[2]),
    verbose=0
)

# gru_regressor = KerasRegressor(model=build_gru_model, verbose=0)

param_grid = {
    "model__units": [32, 64],
    "model__learning_rate": [0.001, 0.0005],
    "batch_size": [32],
    "epochs": [100]
}

# %%


from sklearn.model_selection import GridSearchCV

# LSTM
grid_lstm = GridSearchCV(estimator=lstm_regressor, param_grid=param_grid, cv=4, scoring='neg_mean_squared_error')
grid_lstm_result = grid_lstm.fit(X_train_seq, y_train_seq.ravel())

# # GRU
# grid_gru = GridSearchCV(estimator=gru_regressor, param_grid=param_grid, cv=3, scoring='neg_mean_squared_error')
# grid_gru_result = grid_gru.fit(X_train_seq, y_train_seq.ravel())



# %%

print("🔹 Melhor LSTM:")
print(grid_lstm_result.best_params_)
print(f"Melhor MSE (negativo): {grid_lstm_result.best_score_}")

# print("\n🔸 Melhor GRU:")
# print(grid_gru_result.best_params_)
# print(f"Melhor MSE (negativo): {grid_gru_result.best_score_}")
