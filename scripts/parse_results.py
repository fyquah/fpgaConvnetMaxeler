import factors

logic_utilization = [
    14512,
    12621,
    13616,
    12222,
    15726,
    13193,
    12972,
    12864,
    12299,
    12961,
]

multipliers = [
    14,
    5 ,
    50,
    23,
    10,
    60,
    10,
    20,
    32,
    40,
]

max_multipliers = 3926
max_logic = 262400

for a, b in factors.factors:
    f = open("%s_%s.out" % (a, b), "r")
    time_taken = float(f.read())
    throughput.append(100 / (time_taken * 0.000001))
    f.close()



