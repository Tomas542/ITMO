# https://edabit.com/challenge/YqLBEZJR9ySndYQpH

def staircase(n: int) -> str:
    if n > 0:
        return "\n".join(["_" * (n - i) + "#" * i for i in range(1, n + 1)])
    n = -n
    return "\n".join(["_" * (n - i) + "#" * i for i in range(n, 0, -1)])
