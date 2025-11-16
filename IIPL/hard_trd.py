# https://edabit.com/challenge/Y5Ji2HDnQTX7MxeHt

def snakefill(n: int) -> int:
    limit = n * n
    num = 0

    while 2**num <= limit:
        num += 1
    return num - 1
