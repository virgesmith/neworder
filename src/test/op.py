# https://docs.python.org/3/extending/embedding.html
def add(a,b):
    print("[python]: ", a, "+", b)
    return a + b

def sub(a,b):
    print("[python]: ", a, "-", b)
    return a - b

def mul(a,b):
    print("[python]: ", a, "*", b)
    return a * b

def div(a,b):
    print("[python]: ", a, "/", b)
    return a / b


notafunc = 3

import neworder
print("[python] callback:", neworder.name())