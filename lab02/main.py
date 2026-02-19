def compute(T_left: float,
            T_right: float,
            T_0: float,
            X_left: float,
            X_right: float,
            time: float,
            density: float, 
            specific_hit: float,
            conductivity_coeff: float,
            h: float,
            tau: float) -> dict():

    n_steps = int((X_left - X_right) / h)

    T_i = T_0

    beta = [...]
    alpha = [...]

    for idx in n_steps:
        A_i = C_i = conductivity_coeff / h ** 2
        B_i = 2 * conductivity_coeff / h ** 2 + density * specific_hit / tau
        F_i = - density * conductivity_coeff / tau * T_i

        beta[idx] = (C_i * beta_i - F_i) / (B_i - C_i * alpha_i)
        alpha[idx] = A_i / (B_i - C_i * alpha_i)
