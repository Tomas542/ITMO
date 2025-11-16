# https://edabit.com/challenge/MGALfBAXhXqqdFyqo

from typing import Union


def id_mtrx(n: int) -> Union[str, list]:
    if not isinstance(n, int):
        return "Error"
    if n == 0:
        return []

    zeros = [[0 for _ in range(abs(n))] for _ in range(abs(n))]
    for i in range(abs(n)):
        if n > 0:
            for j in range(abs(n)):
                if i == j:
                    zeros[i][j] = 1
        else:
            for j in range(-1, n - 1, -1):
                if i == (-j - 1):
                    zeros[i][j] = 1
    return zeros
