def arrange(nums: list) -> list:

    for i in range(len(nums) - 1):

        for j in range(len(nums) - 1 - i):

            if nums[j] > nums[j + 1]:
                temp = nums[j]
                nums[j] = nums[j + 1]
                nums[j + 1] = temp

    return nums


nums = list(map(int, input().split(',')))

k = arrange(nums)

print(*k)