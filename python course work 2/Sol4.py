def chocolateDistribution(arr, m):
    arr.sort()

    min_diff = float("inf")

    for i in range(len(arr) - m + 1):
        current_diff = arr[i + m - 1] - arr[i]

        if current_diff < min_diff:
            min_diff = current_diff

    return min_diff


arr = list(map(int, input().split(",")))
m = int(input())

print("Minimum Difference is", chocolateDistribution(arr, m))