import itertools
from math import sqrt

# La matriz asociada para un n es la matriz que representa qué valores son
# afectadas para un dado i que es presionado. Es calculada revisando qué celdas
# son adyacentes a la celda actual, y generand la matriz.
def associated_matrix_for_size(n: int):
    res: list[list[int]] = []
    for i in range(0, n**2):
        up_idx = i - n if i >= n else -1
        down_idx = i + n if i < n**2 - n else -1
        right_idx = i + 1 if i % n != n - 1 else -1
        left_idx = i - 1 if i % n != 0 else -1
        row = [0] * (n**2)
        row[i] = 1
        for idx in [up_idx, down_idx, right_idx, left_idx]:
            if idx != -1:
                row[idx] = 1
        res.append(row)

    return res

def associated_matrix(mat: list[list[int]]):
    return associated_matrix_for_size(len(mat))

# Esta función implementa el algoritmo de eliminación de Gauss. Su
# implementación está basada en este código
# (https://en.wikipedia.org/wiki/Gaussian_elimination#Pseudocode), pero las
# operaciones sobre los valores de la matriz asumen que estamos trabajando en
# Z_2.
def gauss(mat: list[list[int]], b: list[int]):
    def swap_mat_rows(mat: list[list[int]], row1: int, row2: int):
        n = len(mat)
        for i in range(0, n):
            (mat[row1][i], mat[row2][i]) = (mat[row2][i], mat[row1][i])

    n = len(mat)
    assert len(b) == n
    for l in mat:
        for i in l:
            i = i % 2

    res = b.copy()
    h = 0
    k = 0

    # Algoritmo de eliminación de gauss. Tomamos a la matriz aumentada [A|b],
    # y realizamos escalonización en A, replicando las operaciones elementales
    # de fila en los valores de b
    while h < n and k < n:
        i_max = h
        # 1: Necesito encontrar la primera fila que tenga un 1 en la columna k
        for i in range(h + 1, n):
            if mat[i][k] > mat[i_max][k]: i_max = i

        # 1.1: Si no tengo un 1 en la columna k, quiere decir que no tiene
        # pivote. Puedo pasar a la próxima columna.
        if mat[i_max][k] == 0:
            k += 1
            continue

        # 2: Si hay una fila con valor 1, la intercambio con la fila h. Por
        # construcción, la fila h-1 tiene su entrada principal en una columna
        # anterior a k, y ahora la fila h tiene su entrada principal en la
        # columna k.
        swap_mat_rows(mat, h, i_max)
        (res[h], res[i_max]) = (res[i_max],res[h])

        # 3: Reduzco a 0 el resto de los elementos de esta columna que están
        # debajo de la fila h. Para hacerlo, realizo la operación elemental
        # F_i <- F_i + F_h. Después de esto, en las filas posteriores a h
        # la entrada principal va a estar en una columna posterior a k.
        for i in range(h + 1, n):
            f = mat[i][k] // mat[h][k]
            mat[i][k] = 0
            for j in range(k + 1, n):
                mat[i][j] = (mat[i][j] + mat[h][j] * f) % 2
            res[i] = (res[i] + res[h] * f) % 2
        h += 1
        k += 1

    # Después de escalonizar la matriz, puede pasarla a la forma reducida. Esto
    # permite que el b final represente el valor exacto de la solución, en caso
    # de que todas las columnas tengan pivote.
    for i in reversed(range(0, n)):
        if mat[i][i] == 0: continue

        # F_j <- F_j + F_i
        for j in range(0, i):
            if mat[j][i] == 0: continue

            for k in range(0, n):
                mat[j][k] = (mat[j][k] + mat[i][k]) % 2
            res[j] = (res[j] + res[i]) % 2

    return res

# Para resolver el tablero de juego, se necesita resolver el sistema matricial
# [A|b], donde A es la matriz asociada al valor n, y b es la representación
# vectorial del tablero.
#
# Si la matriz asociada tiene pivote en todas las columnas, existe una solución.
# En caso contrario, hay que verificar que el sistema sea compatible, y luego
# resolver en términos de la variable libre. En el caso de que sea compatible,
# esta función fija a los valores libres en 0 y genera la solución resultante.
def solve(mat: list[list[int]]):
    mat1 = associated_matrix(mat)
    b=list(itertools.chain.from_iterable(mat))
    res = gauss(mat1, b)

    indet_idx = -1
    for i in reversed(range(len(mat1))):
        # Como la matriz es cuadrada, y la matriz ahora está en forma
        # escalonada reducida, si el valor en la diagonal es 0 quiere decir
        # que la fila es nula también.
        if mat1[i][i] == 0:
            indet_idx = i
            if res[i] == 1:
                return None

    # Si es un sistema indeterminado (hay filas nulas), fijamos los valores de
    # las variables libres en 0. Como la matriz está en forma escalonada reducida,
    # todas las filas reducirían a la ecuación x_h + x_(i_1) + x_(i_2) + ... + x_(i_m) = b_h,
    # donde h es la fila y i_j es alguna secuencia de las variables libres. Como todas
    # las variables libres son 0, entonces la ecuación siempre se reduce x_h = b_h.


    return res

# Desaplanar la representación vectorial de la solución a una matriz
# para visualización. Simplemente involucra ingresar las filas de la matriz
# como las entradas de h*n a (h+1)*n, para h de 0 a n, ingresando las n filas.
def sol_to_mat(sol: list[int]):
    n = int(sqrt(len(sol)))
    return [sol[(i * n):((i + 1)*n)] for i in range(0, n)]

# https://stackoverflow.com/a/66192895, pero modificado asumiendo
# que los números en la matriz son de Z_2
def det(mat: list[list[int]]):
    matrix = [row[:] for row in mat]
    N = len(matrix)
    for i in range(N-1):
        if matrix[i][i] == 0:
            swapto = next((j for j in range(i+1,N) if matrix[j][i] != 0), None)
            if swapto is None:
                return 0
            matrix[i], matrix[swapto] = matrix[swapto], matrix[i]

        for j in range(i+1,N):
            for k in range(i+1,N):
                t = (matrix[j][k] * matrix[i][i] + matrix[j][i] * matrix[i][k]) % 2
                matrix[j][k] = t

    return matrix[-1][-1]

def ask_user():
    while True:
        try:
            n = int(input('Ingrese el tamaño del tablero de juego: '))
            break
        except ValueError:
            print('Debe ingresar un número')

    mat: list[list[int]] = []
    while len(mat) < n:
        l = input(f'Ingrese la fila {len(mat) + 1} del tablero (separada por espacios): ')
        try:
            row: list[int] = []
            for v in l.split(' '):
                i = int(v)
                if i not in [0, 1]: raise ValueError
                row.append(i)
            if len(row) != n:
                print(f'Debe ingresar {n} números, ingresó {len(row)}')
            mat.append(row)
        except ValueError:
            print('Debe ingresar números de 0 a 1')
    print()

    print('Ingresó el siguiente tablero')
    for l in mat:
        print(l)
    print()

    return mat

if __name__ == '__main__':
    while True:
        print('Ingrese una opción:')
        opts = ['Estadísticas para tablas', 'Resolvedor de tablero']
        for i, opt in enumerate(opts):
            print(f'{i + 1}: {opt}')

        try:
            opt = int(input())
            break
        except ValueError:
            print(f'Debe ingresar un número válido')

    if opt == 1:
        for i in range(1, 31):
            m = associated_matrix_for_size(i)
            d = det(m)
            print(f'{i}: {d}')
    elif opt == 2:
        # mat = [
        #     [1, 1, 1, 1, 1],
        #     [1, 1, 1, 1, 1],
        #     [1, 1, 1, 1, 1],
        #     [1, 1, 1, 1, 1],
        #     [1, 1, 1, 1, 1],
        # ]

        mat = ask_user()

        res = solve(mat)
        if res is None:
            print('No hay solución')
        else:
            print('La solución es')
            for l in sol_to_mat(res):
                print(l)
    else:
        print(f'No se reconoce la opción {opt}')
