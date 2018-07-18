# https://docs.python.org/3/extending/embedding.html
def multiply(a,b):
    print("[python]: ", a, "*", b)
    c = 0
    for i in range(0, a):
        c = c + b
    return c

notafunc = 3