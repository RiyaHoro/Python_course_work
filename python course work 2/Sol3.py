def twoSum(nums:list,k:int)->int:
    result = 0
    
    for i in range(len(nums)-1):
        for j in range(i+1,len(nums)):
            if k - nums[i] == nums[j]:
                result += 1
    return result

nums = list(map(int,input().split(",")))
k = int(input())
print(twoSum(nums,k))