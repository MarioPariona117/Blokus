# Print the header row
SIZE = 5
K = 10
def div(a, b):
    return a / b if b != 0 else 100

def sign(x):
    return (x > 0) - (x < 0)

def func(a, b, k=K):
    return (div(a, b) + div(b, a)) * (a - b)
    # return (div(a, b) - div(b, a)) + (b - a) * k

def render(k=K):
    print(f"{'':^{SIZE}}", end=" ")  # Empty top-left cell
    for j in range(0, 16):
        print(f"{j:^{SIZE}}", end=" ")
    print()  # Newline after the header

    # Print the table rows
    for i in range(0, 16):
        print(f"{i:^{SIZE}}", end=" ")  # Row label on the left
        for j in range(0, 16):
            formatted_value = f"{func(i, j, k):.3g}"  # Format with 3 significant digits
            print(f"{formatted_value:^{SIZE}}", end=" ")  # Center the value within SIZE spaces
        print()  # Newline after each row

import random

class Generator:
    def __init__(self):
        self.i = 1
        self.b = False
        self.l = -0.25
        self.r = -0.15
    def get_k(self):
        if not self.b:
            self.b = True
            return -0.22
        
        while True:
            k = random.random()
            if (-0.00001 + self.l) / (self.r - self.l) < k < (0.00001 + self.r) / (self.r - self.l):
                continue
            break
        k = random.random() * (self.r - self.l) + self.l
        k = round(k, 3)
        return k

    # def get_k():

def dist(i, j, i1, j1, k, new_k):
    for_k = func(i, j, k) - func(i1, j1, k)
    for_new_k = func(i, j, new_k) - func(i1, j1, new_k)
    if sign(for_k) == sign(for_new_k):
        return 0
    return abs(for_k + for_new_k)
    
generator = Generator()
k = generator.get_k()
render(k)
# print("k = ", k)
while True:
    s = input(" Random k :)) ")
    if s == "-":
        print("Exiting...")
        print(k)
        break
    
    new_k = generator.get_k()
    # print("new_k = ", new_k)
    while True:
        max_dist = 0
        i, j, i1, j1, = 0, 0, 0, 0
        for i_ in range(1, 16):
            for i1_ in range(1, 16):
                for j_ in range(1, i_ + 1):
                    for j1_ in range(1, i1_ + 1):
                        current_dist = dist(i_, j_, i1_, j1_, k, new_k)
                        if max_dist < current_dist:
                            max_dist = current_dist
                            i, j, i1, j1 = i_, j_, i1_, j1_
                j_, j1_ = 0, 0
                current_dist = dist(i_, j_, i1_, j1_, k, new_k)
                if max_dist < current_dist:
                    max_dist = current_dist
                    i, j, i1, j1 = i_, j_, i1_, j1_
                        
        # print(f"Comparing {i, j} and {i1, j1}")
        if sign(func(i, j, k) - func(i1, j1, k)) != sign(func(i, j, new_k) - func(i1, j1, new_k)):
            while True:
                breakout = True
                print(f"k = {k}, new_k = {new_k}")
                whichone = input(f"Which one should be higher (better for you) A/B? {i, j} or {i1, j1}?")
                if whichone == ">":
                    if func(i, j, k) <= func(i1, j1, k):
                        k = new_k
                elif whichone == "<":
                    if func(i, j, k) >= func(i1, j1, k):
                        k = new_k
                elif whichone == "=":
                    k = round((k + new_k) / 2, 3)
                elif whichone == "see":
                    breakout = False
                    print(f"func({i}, {j}, {k}) = {func(i, j, k)}")
                    print(f"func({i1}, {j1}, {k}) = {func(i1, j1, k)}")
                    print(f"func({i}, {j}, {new_k}) = {func(i, j, new_k)}")
                    print(f"func({i1}, {j1}, {new_k}) = {func(i1, j1, new_k)}")
                elif whichone == "render":
                    breakout = False
                    render(k)
                else:
                    breakout = False
                    print("Don't be dummy, Mario!")
                if breakout:
                    break
            break