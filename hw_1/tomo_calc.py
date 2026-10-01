from gates import RX, RY
import numpy as np
import matplotlib.pyplot as plt
import itertools
import cvxpy as cp



def get_freq_M(psi, N:int = 1000):
    """
    Функция для генерации вектора частот f и матрицы M для случайного состояния двух кубитов.
    """
    # Проекторы (без унитарных поворотов)
    base_projectors = [
        np.diag([1, 0, 0, 0]).astype(complex), # P_00
        np.diag([0, 1, 0, 0]).astype(complex), # P_01
        np.diag([0, 0, 1, 0]).astype(complex), # P_10
        np.diag([0, 0, 0, 1]).astype(complex)  # P_11
    ]


    # Все повороты для проекторов
    Rx = RX(np.pi/2).matrix
    Ry = RY(-np.pi/2).matrix
    I = np.eye(2, dtype=complex)
    gates = [I, Rx, Ry]

    M_cols = []
    f = []

    for u1, u2 in itertools.product(gates, repeat=2):
        U_alpha = np.kron(u1, u2)
        U_alpha_dag = U_alpha.conj().T
        
        # Состояние после применения унитарок
        psi_meas = U_alpha @ psi
        
        # Теоретические вероятности 
        p_theory = np.abs(psi_meas)**2
        
        # Мультиноминальное сэмплирование
        counts = np.random.multinomial(N, p_theory)
        
        # Вычисление частот
        frequencies = counts / N
        f.extend(frequencies)
        
        for P in base_projectors:
            P_rotated = U_alpha_dag @ P @ U_alpha
            M_cols.append(P_rotated.flatten('F'))

    M_dagger = np.column_stack(M_cols).conj().T
    f = np.array(f).reshape(-1, 1)

    return M_dagger, f

def pseudo_inverse_method(M_dag, f, num_qubits = 2):
    """
    Метод наименьших квадратов для решения системы M * \rho = f.
    Возвращает оценку плотности \hat{\rho}.
    """
    # Размерность
    dim = 2**num_qubits

    # # SVD разложение
    U, S, Vh = np.linalg.svd(M_dag)

    q = (U.conj().T @ f)[:dim**2]
    v = []
    for i in range(dim**2):
        if S[i] <= 0: 
            print(f'index {i} violates IC criterion: sig = {S[i]}')
        v.append(q[i] / S[i])
    v = np.array(v, dtype=complex)
    rho_estimated = (Vh.conj().T @ v).reshape((dim, dim), order='F')
    return rho_estimated

def lsq_improvement(rho_estimated):
    # Спектральное разложение
    eigvals, eigvecs = np.linalg.eigh(rho_estimated)

    # Раворачиваем нумерацию (она изначально по возрастанию)
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:,idx]

    dim = len(eigvals)
    delta = 0.0

    for nu in range(dim, 0, -1):
        D_nu = np.sum(eigvals[:nu])
        current_delta = (D_nu - 1.0)/nu

        if eigvals[nu-1] - current_delta >=0:
            delta = current_delta
            break

    new_eigvals = np.maximum(eigvals - delta, 0.0)

    new_rho = eigvecs @ np.diag(new_eigvals) @ eigvecs.conj().T
    return new_rho


def cvx_method(M_dag, f, num_qubits=2):
    dim = 2**num_qubits
    rho = cp.Variable((dim, dim), complex=True, hermitian=True)

    expected_probs = []
    for i in range(len(f)):
        P_i = (M_dag[i, :].conj()).reshape((dim, dim), order='F')
        expected_probs.append(cp.real(cp.trace(P_i @ rho)))
        
    residual = cp.hstack(expected_probs) - f.flatten()
    objective = cp.Minimize(cp.sum_squares(residual))
    constraints = [cp.trace(rho) == 1, rho >> 0]
    
    prob = cp.Problem(objective, constraints)
    try:
        prob.solve(solver=cp.SCS, eps=1e-5)
    except Exception:
        prob.solve(solver=cp.CLARABEL)
        
    return rho.value

def mle_method(M_dag, f, rho_0, num_qubits = 2,  mu = 0.5, eps=1e-6, max_iter=2000):
    dim = 2**num_qubits
    f_flat = f.flatten()
    num_bases = len(f_flat) // dim # количество базисов (та самая девятка)

    projectors = []
    for i in range(len(f)):
        P_i = (M_dag[i, :].conj()).reshape((dim, dim), order='F')
        projectors.append(P_i)
    
    _, eigvecs = np.linalg.eigh(rho_0)
    psi_0 = eigvecs[:, -1] # СВ с максимальным СЗ

    def J(psi):
        p = np.real(np.einsum('j, mjk, k -> m', np.conj(psi), projectors, psi))
        weights = (f_flat/ num_bases) / np.maximum(p, 1e-15)
        return np.einsum('m, mjk -> jk', weights, projectors)

    psi_1 = mu * J(psi_0) @ psi_0 + (1-mu) * psi_0
    psi_1 /= np.linalg.norm(psi_1)
    infid = 1 - np.abs(np.conj(psi_1) @ psi_0)**2 
    iteration = 1
    while infid >= eps and iteration <= max_iter:
        psi_0 = psi_1
        psi_1 = mu * J(psi_0) @ psi_0 + (1-mu) * psi_0
        psi_1 /= np.linalg.norm(psi_1)
        infid = 1 - np.abs(np.conj(psi_1) @ psi_0)**2 
        iteration += 1
    print(f"MLE сошелся за {iteration} итераций")

    return np.outer(psi_1, np.conj(psi_1))

def plot_density_matrix(rho, num_qubits=2, title="Density Matrix"):
    """
    Визуализирует матрицу плотности в виде двух тепловых карт 
    (действительная и мнимая части).
    """
    dim = 2**num_qubits
    
    labels = [f"{i:0{num_qubits}b}" for i in range(dim)]
    
    rho_real = np.real(rho)
    rho_imag = np.imag(rho)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(title, fontsize=16)
    
    
    vmax_real = max(np.max(np.abs(rho_real)), 1.0)
    vmax_imag = max(np.max(np.abs(rho_imag)), 0.5) 
    
    im1 = axes[0].imshow(rho_real, cmap="RdBu_r", vmin=-vmax_real, vmax=vmax_real)
    axes[0].set_title("Real Part")
    fig.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)
    
    im2 = axes[1].imshow(rho_imag, cmap="RdBu_r", vmin=-vmax_imag, vmax=vmax_imag)
    axes[1].set_title("Imaginary Part")
    fig.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)
    
    for ax, data in zip(axes, [rho_real, rho_imag]):
        ax.set_xticks(np.arange(dim))
        ax.set_yticks(np.arange(dim))
        ax.set_xticklabels(labels)
        ax.set_yticklabels(labels)
        
    
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        

        for i in range(dim):
            for j in range(dim):
                text_color = "white" if np.abs(data[i, j]) > np.max(np.abs(data))/2 else "black"
                ax.text(j, i, f"{data[i, j]:.2f}",
                        ha="center", va="center", color=text_color, fontsize=9)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Инициализируем случайный вектор состояния для двух кубитов
    dim = 2**2
    psi = np.random.randn(dim) + 1j * np.random.randn(dim)
    psi /= np.linalg.norm(psi)
    true_rho = np.outer(psi, np.conj(psi))

    # матрицы M и f
    M, f = get_freq_M(psi, N=1000)
    cond_number = np.linalg.cond(M)
    print(f'condition number of matrix M is {cond_number}')

    # псевдо-инверсия
    rho_estimated = pseudo_inverse_method(M, f)
    print(f'Trace = {np.trace(rho_estimated)}')
    print(f'Eigs: {np.linalg.eigvals(rho_estimated)}')
    print(f'Is it hermitian? - {np.allclose(rho_estimated, rho_estimated.conj().T)}', '\n')

    # псевдо-инверсия + lsq улучшение
    rho_improved = lsq_improvement(rho_estimated)
    print(f'Trace = {np.trace(rho_estimated)}')
    print(f'Is it hermitian? - {np.allclose(rho_improved, rho_improved.conj().T)}')
    print(f'Eigs: {np.linalg.eigvals(rho_improved)}', '\n')

    # выпуклая оптимизация
    rho_opt = cvx_method(M, f)
    print(f'Trace = {np.trace(rho_opt)}')
    print(f'Is it hermitian? - {np.allclose(rho_opt, rho_opt.conj().T)}')
    print(f'Eigs: {np.linalg.eigvals(rho_opt)}')
    print(f'Infid: {1 - np.trace(rho_opt @ true_rho)}\n')

    # MLE
    rho_mle = mle_method(M, f, rho_opt)
    print(f'Trace = {np.trace(rho_mle)}')
    print(f'Is it hermitian? - {np.allclose(rho_mle, rho_mle.conj().T)}')
    print(f'Eigs: {np.linalg.eigvals(rho_mle)}')
    print(f'Infid: {1 - np.trace(rho_mle @ true_rho)}')

