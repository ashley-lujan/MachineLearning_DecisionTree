import math


def entropy(p1, p2):
    if p1 == 1 or p2 == 1:
        return 0
    # if p1 == 0:
    #     return -p2 * math.log(p2, 10)
    # elif p2 == 0:
    #     return -p1 * math.log(p1, 10)
    return round(-p1 * math.log(p1, 2) - p2 * math.log(p2, 2), 2)

def something():
    while True:
        results = input("number = 0, number = 0 & y = 0, number = 1 & y = 0, sample size?\n")
        param =  results.strip().split(',')
        # firstp = p[0].split('/')
        # p1 = (float)(firstp[0]) / (float)(firstp[1])
        #
        # secondp = p[1].split('/')
        # p2 = (float)(secondp[0]) / (float)(secondp[1])
        # print(round(entropy(p1, p2), 3))

        num0 = int(param[0])
        num00 = int(param[1])
        num10 = int(param[2])
        numS = int(param[3])

        p0 = num0 / numS
        p1 = 1 - p0

        if num0 == 0:
            entropy00 = num00 / num0
            entropy01 = 1 - entropy00
        else:
            entropy00 = 1
            entropy01 = 0

        if numS - num0 != 0:
            entropy10 = num10 / (numS - num0)
            entropy11 = 1 - entropy10
        else:
            entropy10 = 1
            entropy11 = 0

        num = p0 * entropy(entropy00, entropy01) + p1 * entropy(entropy10, entropy11)
        print(num)

def question1():
    calcs = [
        #question 3d
        (4/6) * ent(3 / 4, 1 / 4) + (2/6)* ent(1 / 2, 1 / 2),
        (2/6) * ent(1 / 2, 1 / 2) + (4/6) * ent(1 / 4, 3 / 4)
        # (2/6) * ent(2 / 2, 0 / 2) + (3/6) * ent(1 / 3, 2 / 3) + (2/6) * ent(0 / 1, 1 / 1),
        # (3/6) * ent(3 / 3, 0 / 3) + (3/6) * ent(0 / 3, 3 / 3),
        # (2/6) * ent(1 / 2, 1 / 2) + (4/6) * ent(2 / 4, 2 / 4)
        # (3 / 3) * entropy(2 / 3, 1 / 3),
        # (2 / 3) * entropy(1 / 2, 1 / 2) + (1 / 3) * entropy(1 / 1, 0 / 1),
        # (2 / 3) * entropy(2 / 2, 0 / 2) + (1 / 3) * entropy(0 / 1, 1 / 1)
        # (2 / 4) * entropy(2 / 2, 0 / 2) + (2 / 4) * entropy(1 / 2, 1 / 2),
        # (1 / 4) * entropy(0 / 1, 1 / 1) + (3 / 4) * entropy(3 / 3, 0 / 3),
        # (2 / 4) * entropy(2 / 2, 0 / 2) +(2 / 4) * entropy(1 / 2, 1 / 2)

    # (5/7) * entropy(4/6, 1/6) + (2/7) * entropy(1/2, 1/2),
        # (3 / 7) * entropy(1 / 3, 2 / 3) + (4 / 7) * entropy(4 / 4, 0 / 4),
        # (4 / 7) * entropy(3 / 4, 1 / 4) + (3 / 7) * entropy(2 / 3, 1 / 3),
        # (4 / 7) * entropy(4 / 4, 0 / 4) + (3 / 7) * entropy(1 / 3, 2 / 3)
    ]
    for calc in calcs:
        print(calc, " & ", 1 - calc)

    # print(entropy(1/3, 2/3))

def makeRow():
    # l = []
    # while True:
    #     inp = input("hey?\n")
    #     if inp == "STOP":
    #         break
        # l.append(inp)
        #print(inp.replace(" ", " & "), "\\\\ \\hline")

    inp = input("hey\n")
    l = inp.split("\\hline")
    for i in range(0, len(l)):
        print((i+1), " & ", l)

def GI(p1, p2):
    return 1 - (p1 * p1) - (p2 * p2)

def doingGI():
    l = [
        (3/5) * GI(1 / 3, 2 / 3) + (2/5) * GI(1 / 2, 1 / 2),
        (2/5) * GI(1 / 2, 1 / 2) + (3/5) * GI(2 / 3, 1 / 3),
        (3/5) * GI(3 / 3, 0 / 3) + (2/5) * GI(0 / 2, 2 / 2)
        # (2/5) * GI(0 / 2, 2 / 2) + (2/5) * GI(1 / 2, 1 / 2) + (1/5) * GI(0 / 1, 1 / 1),
        # (3/5) * GI(0 / 3, 3 / 3) + (2/5) * GI(0 / 2, 2 / 2),
        # (3/5) * GI(2 / 3, 1 / 3) + (2/5) * GI(1 / 2, 1 / 2)
    ]
    for s in l:
        print(0.48 - round(s, 3))
    # print(round(GI(3/5, 2/5), 3))

if __name__ == '__main__':
    def ent(p1, p2):
        return entropy(p1, p2)
    # print(entropy(5/15, 10/15))

    l = [
        ((5+5/14)/15) * ent(3 / (5 + 5 / 14), (2 + 5 / 14) / (5 + 5 / 14)) +
        ((4+4/14)/15) * ent(0, (4 + (4 / 14)) / (4 + 4 / 14)) +
        ((5+5/14)/15) * ent((2 / (5 + 5 / 14)), ((3 + 5 / 14) / (5 + 5 / 14)))
        # ((5 + 5/14)/15) * ent(3 / 5, 2 / 5 + 5 / 14) + ((4 + (4/14))/15)* ent(0,(4/4) + (4 / 14)) + ((5 + (5/14))/15)* ent(2 / 5, 3 / 5 + (5 / 14))
       # (5/15) * ent(3 / 5, 2 / 5) + (5/15) *  ent(0 / 5, 5 / 5) + (5/15) * ent(2 / 5, 3 / 5),
       # (6/15) * ent(3 / 6, 3 / 6) + (4/15) *  ent(0 / 4, 4 / 4) + (5/15) * ent(2 / 5, 3 / 5),
       #  (4/15) * ent(2 / 4, 2 / 4) + (7/15) * ent(2 / 7, 5 / 7) + (4/15) * ent(1 / 4, 3 / 4),
       #  (7/15) * ent(4 / 7, 3 / 7) + (8/15) * ent(1 / 8, 7 / 8),
       #  (6/15) * ent(3 / 6, 3 / 6) + (9/15) * ent(2 / 9, 7 / 9)
    ]

    # for s in l:
    #     print( round(s, 3))

    # print(ent(2/6, 4/6))

    question1()