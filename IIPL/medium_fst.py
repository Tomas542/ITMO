# https://edabit.com/challenge/3DAkZHv2LZjgqWbvW

def is_adjacent(matrix: list, node1: int, node2: int):
    return matrix[node1][node2] and matrix[node2][node1]
