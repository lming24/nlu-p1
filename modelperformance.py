import statistics

with open("group10.perplexityC") as f:
    numbers = [float(a.strip()) for a in f]
    print(statistics.median(numbers))


