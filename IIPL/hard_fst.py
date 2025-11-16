# https://edabit.com/challenge/3A3mHS5B3NNZddQL2

def interview(questions: list[int], time: int) -> str:
    time_limit = [x for i in range(4) for x in 2 * [5 * (i + 1)]]
    if (
        time > 120
        or not all(q <= t for q, t in zip(questions, time_limit))
        or len(questions) < len(time_limit)
    ):
        return "disqualified"
    return "qualified"
