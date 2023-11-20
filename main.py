import time
import numpy as np

from scipy.stats import multivariate_normal

print("1 задача")

with open("matrix.txt") as file:
    matrix = np.matrix(
        [list(map(int, row.split(","))) for row in file.readlines()])
    print(matrix)
    print(f"Сумма элементов матрицы {np.sum(matrix)}")
    print(f"Минимальный {np.min(matrix)}")
    print(f"Максимальный {np.max(matrix)}")

print("2 задача")


def rle(inp):
    elem = np.array(list(), dtype=int)
    ctn = np.array(list(), dtype=int)
    i = 0
    while i < len(inp):
        t = 1
        elem = np.append(elem, inp[i])
        while i < len(inp)-1 and inp[i] == inp[i + 1]:
            t += 1
            i += 1
        ctn = np.append(ctn, t)
        i += 1
    return elem, ctn


inp = np.array([2, 2, 2, 3, 3, 3, 5])
print(rle(inp))

print("3 задача")

x = np.random.normal(size=(10, 4))
print(f"""
min: {np.min(x)}
max: {np.max(x)}
mean: {np.mean(x)}
std: {np.std(x)}
{x[0:5]}
""")


print("4 задача")

x = np.array([6, 2, 0, 3, 0, 0, 5, 7, 0])

zero = x == 0
print(x[1:][zero[:-1]].max())


print("5 задача")


def logarifm(X, m, C):
    detCov = np.linalg.det(C)
    invCov = np.linalg.inv(C)
    X_centered = X - m
    exponent = -0.5 * np.sum(X_centered @ invCov * X_centered, axis=1)
    normalization = -0.5 * D * np.log(2 * np.pi) - 0.5 * np.log(detCov)
    log = exponent + normalization
    return log


N = 10
D = 7
X = np.random.randn(N, D)
m = np.random.randn(D)
C = np.random.randn(D, D)
C = np.dot(C, C.T)
start_time = time.time()
result_custom = logarifm(X, m, C)
custom_duration = time.time() - start_time
start_time = time.time()
result_scipy = multivariate_normal(m, C).logpdf(X)
scipy_duration = time.time() - start_time

print(f"\nMy result: \n{result_custom}\n")
print(f"Scipy result:\n{result_scipy}\n")

print(f"\nMy time: {custom_duration} seconds\n")
print(f"Scipy time: {scipy_duration} seconds\n")

print("6 задача")
a = np.arange(16).reshape(4, 4)
print(a)
a[[1, 3]] = a[[3, 1]]
print(a)

print("7 задача")
u = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
iris = np.genfromtxt(u, delimiter=',', dtype='object')
with_out = iris[:, :-1]
uni = np.unique(with_out)
print(uni, len(uni), sep="\n")

print("7 задача")
n = np.array([0, 1, 2, 0, 0, 4, 0, 6, 9])
i = np.where(n != 0)
print(i)
