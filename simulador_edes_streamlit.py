# =============================================================================
# SIMULADOR DE EDEs - VERSIÓN STREAMLIT
# =============================================================================

import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
import streamlit as st

# =============================================================================
# MÉTODOS NUMÉRICOS
# =============================================================================

def euler_maruyama(T, X0, N, f, g, seed=42):
    np.random.seed(seed)
    dt = T / N
    dB = np.sqrt(dt) * np.random.randn(N)
    X = np.zeros(N + 1)
    X[0] = X0
    t = np.linspace(0, T, N + 1)
    for j in range(N):
        X[j+1] = X[j] + f(t[j], X[j]) * dt + g(t[j], X[j]) * dB[j]
    return t, X

def milstein(T, X0, N, f, g, dg, seed=42):
    np.random.seed(seed)
    dt = T / N
    dB = np.sqrt(dt) * np.random.randn(N)
    X = np.zeros(N + 1)
    X[0] = X0
    t = np.linspace(0, T, N + 1)
    for j in range(N):
        X[j+1] = (X[j] + f(t[j], X[j]) * dt +
                  g(t[j], X[j]) * dB[j] +
                  0.5 * g(t[j], X[j]) * dg(t[j], X[j]) * (dB[j]**2 - dt))
    return t, X

def heun_stochastic(T, X0, N, f, g, seed=42):
    np.random.seed(seed)
    dt = T / N
    dB = np.sqrt(dt) * np.random.randn(N)
    X = np.zeros(N + 1)
    X[0] = X0
    t = np.linspace(0, T, N + 1)
    for j in range(N):
        Binc = dB[j]
        X_tilde = X[j] + f(t[j], X[j]) * dt + g(t[j], X[j]) * Binc
        X[j+1] = (X[j] + 0.5 * (f(t[j], X[j]) + f(t[j+1], X_tilde)) * dt +
                  0.5 * (g(t[j], X[j]) + g(t[j+1], X_tilde)) * Binc)
    return t, X

def rk4_stochastic(T, X0, N, f, g, seed=42):
    np.random.seed(seed)
    dt = T / N
    dB = np.sqrt(dt) * np.random.randn(N)
    X = np.zeros(N + 1)
    X[0] = X0
    t = np.linspace(0, T, N + 1)
    for j in range(N):
        Binc = dB[j]
        F0 = f(t[j], X[j])
        G0 = g(t[j], X[j])
        X1 = X[j] + 0.5 * F0 * dt + 0.5 * G0 * Binc

        F1 = f(t[j] + 0.5 * dt, X1)
        G1 = g(t[j] + 0.5 * dt, X1)
        X2 = X[j] + 0.5 * F1 * dt + 0.5 * G1 * Binc

        F2 = f(t[j] + 0.5 * dt, X2)
        G2 = g(t[j] + 0.5 * dt, X2)
        X3 = X[j] + F2 * dt + G2 * Binc

        F3 = f(t[j], X3)
        G3 = g(t[j], X3)

        X[j+1] = (X[j] +
                  (dt / 6) * (F0 + 2*F1 + 2*F2 + F3) +
                  (1 / 6) * (G0 + 2*G1 + 2*G2 + G3) * Binc)
    return t, X

# =============================================================================
# INTERFAZ STREAMLIT
# =============================================================================

st.title("🧪 Simulador de EDEs (Itô / Stratonovich)")
st.markdown("Autor: **Dennis Quispe Sánchez**")

f_str = st.text_input("Función determinista f(t, x):", value="x")
g_str = st.text_input("Coef. estocástico g(t, x):", value="0.2*x")
X0 = st.number_input("Condición inicial X₀:", value=1.0)
T = st.number_input("Tiempo total T:", value=1.0)
N = st.number_input("Número de pasos N:", value=100, step=10)

interpretacion = st.selectbox("Interpretación", ["Itô", "Stratonovich"])
metodo = st.selectbox("Método numérico", ["Euler-Maruyama", "Milstein", "Heun", "Runge-Kutta 4", "Comparar todos"])

if st.button("📈 Simular"):
    try:
        t_sym, x_sym = sp.symbols('t x')
        f_expr = sp.sympify(f_str)
        g_expr = sp.sympify(g_str)
        dg_expr = sp.diff(g_expr, x_sym)

        f_ = lambda t_, x_: float(sp.lambdify((t_sym, x_sym), f_expr, 'numpy')(t_, x_))
        g_ = lambda t_, x_: float(sp.lambdify((t_sym, x_sym), g_expr, 'numpy')(t_, x_))
        dg_ = lambda t_, x_: float(sp.lambdify((t_sym, x_sym), dg_expr, 'numpy')(t_, x_)) if x_ > 0 else 0

        f1 = lambda t_, x_: f_(t_, x_) - 0.5 * g_(t_, x_) * dg_(t_, x_)
        f2 = lambda t_, x_: f_(t_, x_) + 0.5 * g_(t_, x_) * dg_(t_, x_)

        if interpretacion == "Itô":
            feuler = f_
            fmilstein = f_
            fheun = f1
            frk4 = f1
        else:
            feuler = f2
            fmilstein = f2
            fheun = f_
            frk4 = f_

        fig, ax = plt.subplots(figsize=(10, 5))
        if metodo == "Euler-Maruyama":
            t, X = euler_maruyama(T, X0, int(N), feuler, g_)
            ax.plot(t, X, label="Euler-Maruyama")

        elif metodo == "Milstein":
            t, X = milstein(T, X0, int(N), fmilstein, g_, dg_)
            ax.plot(t, X, label="Milstein")

        elif metodo == "Heun":
            t, X = heun_stochastic(T, X0, int(N), fheun, g_)
            ax.plot(t, X, label="Heun")

        elif metodo == "Runge-Kutta 4":
            t, X = rk4_stochastic(T, X0, int(N), frk4, g_)
            ax.plot(t, X, label="RK4")

        else:
            t1, X1 = euler_maruyama(T, X0, int(N), feuler, g_)
            t2, X2 = milstein(T, X0, int(N), fmilstein, g_, dg_)
            t3, X3 = heun_stochastic(T, X0, int(N), fheun, g_)
            t4, X4 = rk4_stochastic(T, X0, int(N), frk4, g_)
            ax.plot(t1, X1, '--', label="Euler-Maruyama")
            ax.plot(t2, X2, '--', label="Milstein")
            ax.plot(t3, X3, '--', label="Heun")
            ax.plot(t4, X4, '--', label="RK4")

        ax.set_title(f"Simulación EDE - {interpretacion}")
        ax.set_xlabel("t")
        ax.set_ylabel("X(t)")
        ax.grid(True)
        ax.legend()

        st.pyplot(fig)

    except Exception as e:
        st.error(f"Error: {str(e)}")