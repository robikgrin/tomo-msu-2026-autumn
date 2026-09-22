import numpy as np


def haar_measure(n):
    z = (np.random.randn(n,n) + 1j*np.random.randn(n,n))/np.sqrt(2.0)
    q,r = np.linalg.qr(z)
    d = np.diagonal(r)
    ph = d/np.absolute(d)
    q = np.multiply(q,ph,q)
    return q

class Gate:
    """Базовый класс для квантовых гейтов"""
    def __init__(self, matrix, name):
        self._matrix = np.array(matrix)
        self._name = name

    @property
    def matrix(self):
        """Возвращает матрицу гейта"""
        return self._matrix

    @property
    def name(self):
        """Возвращает имя гейта"""
        return self._name

    def __repr__(self):
        return f"Gate: {self._name}"
    
class I(Gate):
    def __init__(self):
        super().__init__([[1, 0], [0, 1]], "I")

class X(Gate):
    def __init__(self):
        super().__init__([[0, 1], [1, 0]], "X")

class H(Gate):
    def __init__(self):
        super().__init__(
            [[1, 1], [1, -1]] / np.sqrt(2), "H"
        )

class RX(Gate):
    def __init__(self, theta):
        matr = np.array([
            [np.cos(theta / 2), -1j * np.sin(theta / 2)],
            [-1j * np.sin(theta / 2), np.cos(theta / 2)]
        ], dtype=complex)
        super().__init__(matr, f"RX({np.round(theta, 2)})")

class RZ(Gate):
    def __init__(self, theta):
        matr = np.array([
            [np.exp(-1j*theta/2), 0],
            [0, np.exp(1j*theta/2)]
        ], dtype=complex)
        super().__init__(matr, f"RZ({np.round(theta, 2)})")

class RY(Gate):
    def __init__(self, theta):
        matr = np.array([
            [np.cos(theta / 2), -np.sin(theta / 2)],
            [np.sin(theta / 2), np.cos(theta / 2)]
        ], dtype=complex)
        super().__init__(matr, f"RY({np.round(theta, 2)})")

class U_random_1q(Gate):
    def __init__(self):
        super().__init__(haar_measure(2), "U_random")