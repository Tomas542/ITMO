# https://edabit.com/challenge/MGALfBAXhXqqdFyqo

def atbash(txt: str) -> str:
    txt = list(txt)
    for i in range(len(txt)):
        if 65 <= ord(txt[i]) <= 90:
            txt[i] = chr(ord(txt[i]) + int(2 * (77.5 - ord(txt[i]))))

        elif 97 <= ord(txt[i]) <= 122:
            txt[i] = chr(ord(txt[i]) + int(2 * (109.5 - ord(txt[i]))))

    return "".join(txt)
