"""
App Streamlit — Projeção de Conta de Ar-Condicionado
Usa TensorFlow para estimar o consumo com base nas horas de uso diário.
"""

import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# ─────────────────────────────────────────────
# 1. CONFIGURAÇÃO DA PÁGINA
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="💨 Conta do AC",
    page_icon="❄️",
    layout="centered"
)

st.title("❄️ Projeção de Conta — Ar-Condicionado")
st.markdown("Informe quantas horas por dia você usa o ar-condicionado e veja a projeção do gasto mensal.")

# ─────────────────────────────────────────────
# 2. PARÂMETROS FÍSICOS (sidebar)
# ─────────────────────────────────────────────
st.sidebar.header("⚙️ Configurações")

potencia_kw = st.sidebar.number_input(
    "Potência do AC (kW)",
    min_value=0.5, max_value=5.0, value=1.5, step=0.1,
    help="Aparelhos domésticos comuns: 9000 BTU ≈ 0,9 kW | 12000 BTU ≈ 1,2 kW | 18000 BTU ≈ 1,8 kW"
)

tarifa_kwh = st.sidebar.number_input(
    "Tarifa de energia (R$/kWh)",
    min_value=0.50, max_value=2.00, value=0.75, step=0.01,
    help="Tarifa média residencial no Brasil em 2024: R$ 0,75/kWh"
)

dias_no_mes = st.sidebar.slider(
    "Dias no mês",
    min_value=28, max_value=31, value=30
)

# ─────────────────────────────────────────────
# 3. INPUT PRINCIPAL — horas de uso por dia
# ─────────────────────────────────────────────
horas_uso = st.slider(
    "🕐 Quantas horas por dia você usa o AC?",
    min_value=1, max_value=24, value=8
)

# ─────────────────────────────────────────────
# 4. TREINAMENTO DO MODELO TENSORFLOW
#    O modelo aprende a relação: horas → custo diário
#    Relação real: custo = horas × potência × tarifa
#    O TF vai aproximar essa função linear com dados sintéticos.
# ─────────────────────────────────────────────

@st.cache_resource  # treina apenas uma vez por sessão
def treinar_modelo(potencia, tarifa):
    """
    Treina uma rede neural simples para aprender:
    custo_diario = horas × potencia × tarifa
    """
    # Dados sintéticos de 1 a 24 horas
    X_treino = np.arange(1, 25, dtype=np.float32).reshape(-1, 1) / 24.0  # normaliza [0,1]
    y_treino = np.arange(1, 25, dtype=np.float32) * potencia * tarifa     # custo real

    modelo = keras.Sequential([
        layers.Dense(16, activation='relu', input_shape=(1,)),
        layers.Dense(8, activation='relu'),
        layers.Dense(1)  # saída linear → valor contínuo de custo
    ])

    modelo.compile(optimizer='adam', loss='mse')
    modelo.fit(X_treino, y_treino, epochs=300, verbose=0)

    return modelo

with st.spinner("🧠 Treinando modelo neural..."):
    modelo = treinar_modelo(potencia_kw, tarifa_kwh)

# ─────────────────────────────────────────────
# 5. PROJEÇÃO DIA A DIA AO LONGO DO MÊS
# ─────────────────────────────────────────────

# Para cada dia do mês, o custo acumulado cresce
# O modelo prevê o custo diário e somamos ao longo dos dias

horas_norm = np.array([[horas_uso / 24.0]], dtype=np.float32)
custo_diario_previsto = float(modelo.predict(horas_norm, verbose=0)[0][0])

# Gera evolução acumulada dia a dia
dias = np.arange(1, dias_no_mes + 1)
custo_acumulado = dias * custo_diario_previsto

# ─────────────────────────────────────────────
# 6. EXIBIÇÃO DOS RESULTADOS
# ─────────────────────────────────────────────

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("💡 Custo diário", f"R$ {custo_diario_previsto:.2f}")

with col2:
    custo_semanal = custo_diario_previsto * 7
    st.metric("📅 Custo semanal", f"R$ {custo_semanal:.2f}")

with col3:
    custo_mensal = custo_acumulado[-1]
    st.metric("🗓️ Custo mensal", f"R$ {custo_mensal:.2f}")

st.markdown("---")

# ─────────────────────────────────────────────
# 7. GRÁFICO DE LINHA — evolução do gasto
# ─────────────────────────────────────────────

st.subheader("📈 Evolução do gasto ao longo do mês")

df_grafico = pd.DataFrame({
    "Gasto acumulado (R$)": custo_acumulado
}, index=dias)

df_grafico.index.name = "Dia do mês"

st.line_chart(df_grafico)

# ─────────────────────────────────────────────
# 8. TABELA DETALHADA (opcional)
# ─────────────────────────────────────────────

with st.expander("🔍 Ver tabela detalhada dia a dia"):
    df_tabela = pd.DataFrame({
        "Dia": dias,
        "Custo diário (R$)": [round(custo_diario_previsto, 2)] * dias_no_mes,
        "Custo acumulado (R$)": [round(v, 2) for v in custo_acumulado]
    })
    st.dataframe(df_tabela, use_container_width=True)

# ─────────────────────────────────────────────
# 9. DICA FINAL
# ─────────────────────────────────────────────

st.info(
    f"💡 **Dica:** Reduzir 1 hora de uso por dia economizaria "
    f"**R$ {(custo_diario_previsto / horas_uso) * dias_no_mes:.2f}** por mês."
)

st.caption("Modelo: TensorFlow/Keras | Tarifa e potência configuráveis na barra lateral.")
