import numpy as np

def simulate_increments(delta_t, d):
    """
    Simula las variables aleatorias para los incrementos.
    - delta_t: Paso de tiempo.
    - d: Dimensión del sistema.

    Retorna:
    - Delta_beta: Incrementos \Delta \hat{\beta}_k^{(i)}.
    - Zeta: Variables \hat{\zeta}_k^{(i)}.
    - Delta_beta_ij: Incrementos \Delta \hat{\beta}_k^{(i,j)}.
    """
    sqrt_3dt = np.sqrt(3 * delta_t)
    sqrt_dt = np.sqrt(delta_t)

    # Simular \Delta \hat{\beta}_k^{(i)} con distribución en tres puntos
    Delta_beta = np.random.choice(
        [sqrt_3dt, 0, -sqrt_3dt],
        size=d,
        p=[1/6, 2/3, 1/6]
    )

    # Simular \hat{\zeta}_k^{(i)} con distribución en dos puntos
    Zeta = np.random.choice(
        [sqrt_dt, -sqrt_dt],
        size=d,
        p=[0.5, 0.5]
    )

    # Calcular \Delta \hat{\beta}_k^{(i,j)}
    Delta_beta_ij = np.zeros((d, d))
    for i in range(d):
        for j in range(d):
            if i < j:
                Delta_beta_ij[i, j] = 0.5 * (Delta_beta[i] * Delta_beta[j] - sqrt_dt * Zeta[i])
            elif i > j:
                Delta_beta_ij[i, j] = 0.5 * (Delta_beta[i] * Delta_beta[j] + sqrt_dt * Zeta[j])
            else:
                Delta_beta_ij[i, j] = 0.5 * (Delta_beta[i]**2 - delta_t)

    return Delta_beta, Zeta, Delta_beta_ij

# Ejemplo de uso
delta_t = 0.01
d = 3  # Dimensión del sistema

Delta_beta, Zeta, Delta_beta_ij = simulate_increments(delta_t, d)
print("Delta_beta:", Delta_beta)
print("Zeta:", Zeta)
print("Delta_beta_ij:\n", Delta_beta_ij)
