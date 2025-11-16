# https://edabit.com/challenge/AJGqpNL2yAyhbdpvB

def V_DAC(value: int) -> float:
    return round(value / 1023 * 5, 2)
