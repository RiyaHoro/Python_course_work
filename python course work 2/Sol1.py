def threeSum(nums:list) -> list:
    triplets = []
    nums.sort()
    for i in range(len(nums)):
        for j in range(i+1,len(nums)):
            for k in range(j+1,len(nums)):
                if i!=j and i!=k and j!=k:
                    if nums[i] + nums[j] + nums[k] == 0:
                    
                        res = [nums[i],nums[j],nums[k]]
                        if res not in triplets:
                            triplets.append(res)
    triplets.sort()
    return triplets
nums = list(map(int, input().split(',')))
result = threeSum(nums)
print(result)