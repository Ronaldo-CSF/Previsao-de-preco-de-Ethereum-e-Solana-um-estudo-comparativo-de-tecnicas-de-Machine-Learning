# 📈 Previsão de Preço de Criptoativos — TCC

Este projeto faz parte do **Trabalho de Conclusão de Curso (TCC)** de **Data Science e Analytics**, desenvolvido por **Ronaldo do Couto**.  
O objetivo é **analisar e comparar o desempenho de diferentes modelos de machine learning** — incluindo **Random Forest**, **LSTM** e **GRU** — na **previsão de preços de criptoativos**, com foco em **Ethereum (ETH)** e **Solana (SOL)**.

---

## 🎯 Objetivo

Desenvolver e comparar modelos preditivos capazes de **antecipar o comportamento de mercado de criptoativos** com base em dados históricos.  
A proposta busca compreender avaliar **a eficiência preditiva** das técnicas utilizadas.

---

## 🧠 Metodologia

A metodologia adotada inclui as seguintes etapas:

1. **Coleta de Dados**  
   - Dados históricos de preços e volumes de negociação de ETH e SOL.  
   - Intervalo temporal: **1 minuto**,
   - Os dados de ETH iniciam às 4 horas do dia 17 de agosto de 2017 e se estendem até às 3 horas do dia 14 de dezembro de 2024. Já os dados de SOL iniciam-se às 6 horas do dia 11 de agosto de 2020 e se estendem até o mesmo período.  
   - Fonte: [The Most 50 Popular Crypto Data](https://www.kaggle.com/datasets/kaanxtr/btc-price-1m?resource=download).

2. **Pré-processamento**  
   - Normalização dos dados com `StandardScaler`.  
   - Criação de **janelas deslizantes** para capturar dependências temporais.  
   - Separação em **80% treino / 20% teste**, garantindo comparabilidade entre os modelos:
       - Para redes neurais foi usada a proporção **60% treino, 20% validação e 20% teste**

3. **Modelagem**  
   - **Random Forest (simplificado)**: modelo baseline tradicional de aprendizado supervisionado.
   - **Random Forest (otimizado)**: modelo desenvolvido definindo features a serem testados e avaliados, selecionando o modelo com menor erro associado.
   - **LSTM**: rede recorrente especializada em padrões sequenciais.  
   - **GRU**: variante simplificada e eficiente da LSTM.

4. **Avaliação**  
   - Métricas utilizadas:  
     - **MAE** (Erro Absoluto Médio)  
     - **MSE** (Erro Quadrático Médio)  
     - **RMSE** (Raiz do Erro Quadrático Médio)  
     - **MASE** (Erro Absoluto Médio Escalonado) — *métrica comum entre os modelos para comparação justa*.  

5. **Otimização de Hiperparâmetros**  
   - Utilização do `GridSearchCV` com `RandomForestRegressor` para RF.  
   - Ajuste de parâmetros como número de árvores, máximo de variáveis, profundidade da árvore e tamanho mínimo da folha.

---

## 📚 Autor

Ronaldo do Couto S. Filho
📍 Pós-Graduação em Data Science e Analytics
💻 Engenheiro Químico e entusiasta de Ciência de Dados
🔗 [adicione seu LinkedIn aqui](https://www.linkedin.com/in/ronaldo-do-couto/)


## 📜 Licença

Este projeto é de caráter acadêmico e foi desenvolvido para fins educacionais.
O código pode ser reutilizado e adaptado, desde que seja citada a fonte original.

---

## 🧩 Estrutura do Projeto

```bash
📁 crypto-forecast-tcc/
├── databases_origin/            # Dados brutos
├── databases_parquet/           # Dados pré-tratados, com resamples criados em formato parquet
├── notebooks/                   # Jupyter notebooks com análises e modelos
├── Resultados_LSTM_GRU/         # Métricas de desempenho
├── Resultados_Naive_RF/         # Métricas de desempenho
├── graficos/                    # Gráficos gerados durante as análises
├── requirements.txt      # Dependências do projeto
├── README.md             # Este arquivo
└── Scripts/              # Scripts de execução
```
