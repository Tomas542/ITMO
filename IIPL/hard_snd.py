# https://edabit.com/challenge/MtktG9Dz7z9vBCFYM

import socket


def get_domain(ip_address: str) -> str:
    return socket.gethostbyaddr(ip_address)[0]
