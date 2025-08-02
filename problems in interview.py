'''class Solution(object):
    def twoSum(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: List[int]
        """
        d={}   #S.C:-O(N)
        n=len(nums)
        for a in range(0,n):   #T.C:-O(N)
            b=target-nums[a]
            if (b in d):
                return [a,d[b]]
            else:   
                d[nums[a]]=a''' #twosum (most common question in interviews) (better solution)

'''def twoSum(nums,target): 
    n=len(nums)
    low=0
    high=n-1
    while(low<high):
        Sum=nums[low]+nums[high]
        if(Sum==target):
            return "YES"
        elif(Sum>target):
            high-=1
        elif(Sum<target):
            low+=1
    return "NO"
nums=list(map(int,input().split()))
target=int(input())
print(twoSum(nums,target))''' #twosum for sorted array (when array is sorted and only for yes or no conditions)(optimal solution)

'''class Solution(object):
    def threeSum(self, nums):
        triplet_sum=set()
        n=len(nums)
        for i in range(0,n-1):
            for j in range(i+1,n-1):
                for k in range(j+1,n):
                    if(nums[i]+nums[j]+nums[k]==0):
                        temp=[nums[i],nums[j],nums[k]]
                        temp.sort()
                        triplet_sum.add(tuple(temp))
        ans=[]
        for triplet in triplet_sum:
            ans.append(list(triplet))
        return ans'''# T.C:-O(N**3) # 3sum

'''class Solution(object):
    def threeSum(self, nums):
        triplet_sum=set()
        n=len(nums)
        for i in range(0,n-1):
            hashmap=[]
            for j in range(i+1,n):
                k=-(nums[i]+nums[j])
                if(k in hashmap):
                    temp=[nums[i],nums[j],k]
                    temp.sort()
                    triplet_sum.add(tuple(temp))
                hashmap.append(nums[j])
        ans=[]
        for triplet in triplet_sum:
            ans.append(list(triplet))
        return ans'''   #3sum(better solution)

'''class Solution(object):
    def threeSum(self, nums):
        nums.sort()
        n=len(nums)
        ans=[]
        for i in range(0,n):
            if(i>0 and nums[i]==nums[i-1]):
                continue
            j=i+1
            k=n-1
            while(j<k):
                Sum=nums[i]+nums[j]+nums[k]
                if(Sum<0):
                    j+=1
                elif(Sum>0):
                    k-=1
                else:
                    temp=[nums[i],nums[j],nums[k]]
                    ans.append(temp)
                    j+=1
                    k-=1
                    while(j<k and nums[j-1]==nums[j]):
                        j+=1
                    while(j<k and nums[k+1]==nums[k]):
                        k-=1
        return ans'''                                  #3sum (optimal solution)

'''class Solution(object):
    def majorityElement(self, nums):
        n=len(nums)
        for i in nums:
            if(nums.count(i)>n//2):
                return i''' #majority elements (T.C:-O(N**2))

'''class Solution(object):
    def majorityElement(self, nums):
        d = {}
        n=len(nums)
        for i in nums:
            if(i in d):
                 d[i]=d[i]+1
            else:
                d[i]=1
        for key,values in d.items():
            if(values>n//2):
                return key''' #majority elements (better soluton)(169in leetcode)

'''class Solution(object):
    def majorityElement(self, nums):
        d = {}
        n=len(nums)
        for i in nums:
            if(i in d):
                d[i]=d[i]+1
            else:
                d[i]=1
        ans=[]
        for key,value in d.items():
            if(value>n//3):n
                ans.append(key)
        return ans'''           #majority elements-2 (229 in leetcode)

'''class Solution(object):
    def fourSum(self, nums):
        triplet_sum=set()
        n=len(nums)
        for i in range(0,n-1):
            for j in range(i+1,n-1):
                for k in range(j+1,n-1):
                    for l in range(k+1,n):
                        if(nums[i]+nums[j]+nums[k]+nums[l]==target):
                            temp=[nums[i],nums[j],nums[k],nums[l]]
                            temp.sort()
                            triplet_sum.add(tuple(temp))
        ans=[]
        for triplet in triplet_sum:
            ans.append(list(triplet))
        return ans''' #4sum
#powersum for negitive numbeers(1/x)
'''if(n<0):
    x=1/x
    n=abs(n)
    ans=1
    for i in range(n):
        ans=ans*x
    return ans'''

'''class Solution(object):
    def power(self,x,n):
        if(n==0):
            return 1
        if(n%2==1):
            return x*self.power(x,n-1)
        return self.power(x*x,n//2)
    def myPow(self, x, n):
        
        if(n<0):
            x=1/x
        n=abs(n)
        return self.power(x,n)
        ans=1
        for i in range(n):
            ans=ans*x
        return ans'''
        
        

#subsets
'''class Solution:
    def generate(self,ind,curr,ans,nums):
        if(ind==len(nums)):
            ans.append(curr.copy())
            return 
        curr.append(nums[ind])
        self.generate(ind+1,curr,ans,nums)
        curr.pop()
        self.generate(ind+1,curr,ans,nums)
    def subsets(self,nums: list[int]) -> list[list[ind]]:
        ind=0
        curr=[]
        ans=[]
        self.generate(ind,curr,ans,nums)
        return ans'''
    
#combination sum(39)
'''class Solution:
    def generate(self,ind,curr,ans,candidates,target):
        if(target==0):
            ans.append(curr.copy())
            return 
        if(target<0):
            return
        if(ind==len(candidates)):
            return
        curr.append(candidates[ind])
        self.generate(ind,curr,ans,candidates,target-candidates[ind])
        curr.pop()
        self.generate(ind+1,curr,ans,candidates,target)
    def combinationSum(self, candidates: List[int], target: int) -> List[List[int]]:
        ind=0
        curr=[]
        ans=[]
        self.generate(ind,curr,ans,candidates,target)
        return ans'''

#combination sum2(40)
'''class Solution:
    def generate(self,ind,curr,ans,candidates,target):
        if(target==0):
            ans.append(curr.copy())
            return
        if(target<0):
            return
        if(ind==len(candidates)):
            return
        curr.append(candidates[ind])
        self.generate(ind+1,curr,ans,candidates,target-candidates[ind])
        curr.pop()
        for i in range(ind+1,len(candidates)):
            if(candidates[ind]!=candidates[i]):
                self.generate(i,curr,ans,candidates,target)
                break
    def combinationSum2(self, candidates: List[int], target: int) -> List[List[int]]:
        candidates.sort()
        ind=0
        curr=[]
        ans=[]
        self.generate(ind,curr,ans,candidates,target)
        return ans   '''     


'''class Solution:
    def generate(self,ind,curr_str,ans,O,C,n):
        if(O==C and ind==2*n):
            ans.append(curr_str)
            return
        if(O>n):
            return
        self.generate(ind+1,curr_str+"(",ans,O+1,C,n)
        if(O>C):
            
         self.generate(ind+1,curr_str+")",ans,O,C+1,n)
    def generateParenthesis(self, n: int) -> List[str]:
        ind=0
        curr_str=""
        ans=[]
        O=0
        C=0
        self.generate(ind,curr_str,ans,O,C,n)
        return ans'''#"()"
 #19/05/25       
# merge sort(when list is sorted)
'''nums1=list(map(int,input().split()))
nums2=list(map(int,input().split()))
i=0
j=0
result=[]
while(i<len(nums1) and j<len(nums2)):
    if(nums1[i]<=nums2[j]):
        result.append(nums1[i])
        i+=1
    else:
        result.append(nums2[j])
        j+=1
while(i<len(nums1)):
    result.append(nums1[i])
    i+=1
while(j<len(nums2)):
    result.append(nums2[j])
    j+=1
print(result)'''

'''def mergeSort(arr,n):
    def mS(arr,low,high):
        if(low==high):
            return
        mid=(low+high)//2
        mS(arr,low,mid)
        mS(arr,mid+1,high)
        Sort(arr,low,mid,high)
    def Sort(arr,low,mid,high):
        i=low
        j=mid+1
        k=[]
        while(i<=mid and j<=high):
            if(arr[i]<=arr[j]):
                k.append(arr[i])
                i+=1
            else:
                k.append(arr[j])
                j+=1
        while(i<mid):
            k.append(arr[i])
            i+=1
        while(j<=high):
            k.append(arr[j])
            j+=1
        for ind,val in enumerate(k):
            arr[ind+low]=val
    low=0
    high=n-1
    mS(arr,low,high)
    return arr 
arr=list(map(int,input().split()))
n=len(arr)
print(mergeSort(arr,n))''' #given signle arrary is divided into 2 arrays and sorted

# Quick Sort
'''def quickSort(arr):
    def qs(arr,low,high):
        if(low<high):
            pIndex=partition(arr,low,high)
            qs(arr,low,pIndex-1)
            qs(arr,pIndex+1,high)
    def partition(arr,low,high):
        i=low
        j=high
        pivot=arr[low]
        while(i<j):
            while(arr[i]<=pivot and i<high):
                i+=1
            while(arr[j]>=pivot and j>low):
                j-=1
            if(i<j):
                arr[i],arr[j]=arr[j],arr[i]
        arr[low],arr[j]=arr[j],arr[low]
        return j
    n=len(arr)
    low=0
    high=n-1
    qs(arr,low,high)
    return arr
arr=list(map(int,input().split()))
print(quickSort(arr))'''


#selection sort
'''for i in range(0,n):
    min=i
    for j in range(i+1,n):
        if(arr[j]<arr[i]):
            min=j
    arr[i],arr[min]=arr[min],arr[i]

arr=list(map(int,input().split()))
n=len(arr)
return arr'''

#bubbleSort(greek for greek)
'''n=len(arr)
for i in range(n-1,-1,-1):
    for j in range(i):
        if(arr[j]>arr[j+1]):
            arr[j],arr[j+1]=arr[j+1],arr[j]
return arr'''   

#insertion sort(grrek for greek)
'''n=len(arr)
for i in range(0,n):
    while(i>0 and arr[i-1]>arr[i]):
        arr[i-1],arr[i]=arr[i],arr[i-1]
        i-=1
return arr'''

'''s="hey hii hello"
s=s.split(" ")
s=s[::-1]
print(*s)
print(" ".join(s))'''#  word reverse
            
# power of 2
'''for i in range(31):#31 is constrain )
    if(2**i==n):
        return True
return False  '''

'''return sorted(s)==sorted(t)'''
#20/05/25
#Binary search
'''def binarysearch(arr,k):
    n=len(arr)
    low=0
    high=n-1
    while(low<=high):
        mid=(low+high)//2
        if(arr[mid]==k):
            return mid
        elif(k>arr[mid]):
            low=mid+1
        elif(k<arr[mid]):
            high=mid-1
    return -1'''    #when duplicates are not allowed and lowerbound is not used

'''def lowerBound(arr,target):
    n=len(arr)
    low=0
    high=n-1
    ans=n
    while(low<=high):
        mid=(low+high)//2
        if(arr[mid]>=target):
            ans=mid
            high=mid-1
        else:
            low=mid+1
    return ans'''        # binary search using lowerbound 

#upper bound
'''def upperBound(self, arr, target):
        #code here
        n=len(arr)
        low=0
        high=n-1
        ans=n
        while(low<=high):  
            mid=(low+high)//2
            if(arr[mid]>target):#upper bound
                ans=mid
                high=mid-1
            else:
                low=mid+1
        return ans'''

'''def searchInsert(self, nums: List[int], target: int) -> int:
        n=len(nums)
        low=0
        high=n-1
        ans=n
        while(low<=high):
            mid=(low+high)//2
            if(nums[mid]>=target):
                ans=mid
                high=mid-1
            else: 
                low=mid+1
        return ans'''     #leetcode prblm(35)
            

'''def getFloorAndCeil(a, n, x):
    # Write your code here
    def getFloor(a,n,x):
        low=0
        high=n-1
        ans=-1
        while(low<=high):
            mid=(low+high)//2
            if(a[mid]<=x):
                ans=a[mid]
                low=mid+1
            else:
                high=mid-1
        return ans
    def getCeil(a,n,x):
        low=0
        high=n-1
        ans=-1
        while(low<=high):
            mid=(low+high)//2
            if(a[mid]>=x):
                ans=a[mid]
                high=mid-1
            else:
                low=mid+1
        return ans
    floor=getFloor(a,n,x)
    ceil=getCeil(a,n,x)
    return [floor,ceil]'''   #Floor and Ceil prblm (code ninjas)

'''class Solution:
    def searchRange(self, nums: List[int], target: int) -> List[int]:
        def getLowerBound(nums,target):
            n = len(nums)
            low = 0
            high = n-1
            ans = -1
            while(low<=high):
                mid=(low+high)//2
                if(nums[mid]>=target):
                    ans = mid
                    high = mid-1
                else:
                    low = mid+1
            return ans
        def getUpperBound(nums,target):
            n=len(nums)
            low=0
            high=n-1
            ans=n
            while(low<=high):
                mid=(low+high)//2
                if(nums[mid]>target):
                    ans=mid
                    high=mid-1
                else:
                    low=mid+1
            return ans
        lb=getLowerBound(nums,target)
        if(lb==-1 or nums[lb]!=target):
            return [-1, -1]
        ub=getUpperBound(nums,target)-1
        return [lb,ub]'''   #leetcode (34)

# Search in rotated sorted array
'''class Solution:
    def search(self,arr,key):
        # Complete this function
        n=len(arr)
        low=0
        high=n-1
        while(low<=high):
            mid=(low+high)//2
            if(arr[mid]==key):
                return mid
            # left half is sorted 
            elif(arr[low]<=arr[mid]):
                if(arr[low]<=key<=arr[mid]):
                    high=mid-1
                else:
                    low=mid+1
                # right half is sorted
            elif(arr[mid]<=arr[high]):
                if(arr[mid]<=key<=arr[high]):
                    low=mid+1
                else:
                    high=mid-1
        return -1'''   # T.C:-O(logn)

# Search in rotated sorted array2
'''class Solution:
    def Search(self, arr, key):
        # code here
        n=len(arr)
        low=0
        high=n-1
        while(low<=high):
            mid=(low+high)//2
            if(arr[mid]==key):
                return True
            if(arr[low]==arr[mid]==arr[high]):
                low+=1
                high-=1
          elif(arr[low]<=arr[mid]):
                if(arr[low]<=key<=arr[high]):
                    high=mid-1
                else:
                    low=mid+1
            elif(arr[mid]<=arr[high]):
                if(arr[mid]<=key<=arr[high]):
                    low=mid+1
                else:
                    high=mid-1
        return False'''

#sorted and rotated minimum
'''class Solution:
    def findMin(self, arr):
        #complete the function here
        n=len(arr)
        low=0
        high=n-1
        Min=float("inf")
        while(low<=high):
            mid=(low+high)//2
            if(arr[low]<=arr[mid]):
                if(arr[low]<Min):
                    Min=arr[low]
                low=mid+1
            if(arr[mid]<=arr[high]):
                if(arr[mid]<Min):
                    Min=arr[mid]
                high=mid-1
        return Min'''

#find Kth rotation
'''class Solution:
    def findKRotation(self, arr):
        # code here
        n=len(arr)
        low=0
        high=n-1
        Min=float("inf")
        mIndex=-1
        while(low<=high):
            mid=(low+high)//2
            if(arr[low]<=arr[mid]):
                if(arr[low]<Min):
                    Min=arr[low]
                    mIndex=low
                low=mid+1
            elif(arr[mid]<=arr[high]):
                if(arr[mid]<Min):
                    Min=arr[mid]
                    mIndex=mid
                high=mid-1
        return mIndex'''
#21/05/25               
 #single among doubles in a sorted
'''class Solution:
    def findOnce(self,arr):
        # Complete this function
        d={}
        for i in arr:
            if(i in d):
                d[i]=d[i]+1
            else:
                d[i]=1
        for key,val in d.items():
            if(val==1):
                return key'''     #T.C:-O(2N), S.C:-O(N)
'''class Solution:
    def findOnce(self,arr):
        # Complete this function
        n=len(arr)
        if(n==1):
            return arr[0]
        elif(arr[0]!=arr[1]):
            return arr[0]
        elif(arr[n-1]!=arr[n-2]):
            return arr[n-1]
        low=1
        high=n-2
        while(low<=high):
            mid=(low+high)//2
            if(arr[mid]!=arr[mid-1] and arr[mid]!=arr[mid+1]):
                return arr[mid]
            #you are on the left half
            elif(mid%2==1 and arr[mid]==arr[mid-1]) or (mid%2==0 and arr[mid]==arr[mid+1]):
                low=mid+1
            # you are on the right half 
            elif(mid%2==0 and arr[mid]==arr[mid-1]) or (mid%2==1 and arr[mid]==arr[mid+1]):
                high=mid-1   '''        #same prblm with T.C:-O(logn) and S.C:-O(1)

#square root
'''class Solution:
    def floorSqrt(self, n):
        ans=0
        for i in range(1,n+1):
            if(i*i<=n):
                ans=i
            else:
                break
        return ans''' #T.C:-O(N)

'''class Solution:
    def floorSqrt(self, n):
        ans=0
        low=1
        high=n
        while(low<=high):
            mid=(low+high)//2
            if(mid*mid<=n):
                ans=mid
                low=mid+1
            elif(mid*mid>n):
                high=mid-1
        return ans'''  #same prblm with T.C:-O(logn) 
#find the nth root of m
'''class Solution:
	def nthRoot(self, n: int, m: int) -> int:
		# Code here
		ans=-1
		for i in range(1,m+1):
		    if(i**n==m):
		        ans=i
		        break
		    elif(i**n>m):
		        break
		return ans'''   #T.C:-O(N)

'''class Solution:
	def nthRoot(self, n: int, m: int) -> int:
		# Code here
		low=1
		high=m
		while(low<=high):
		    mid=(low+high)//2
		    if(mid**n==m):
		        return mid
		    elif(mid**n>m):
		        high=mid-1
		    else:
		        low=mid+1
		return -1'''        #T.C:-O(logn)

#samller divisor
'''from math import *
class Solution:
    def smallestDivisor(self, arr, k):
        # Code here
        for div in range(1,max(arr)+1):
            Sum=0
            for num in arr:
                Sum+=ceil(num/div)
            if(Sum<=k):
                return div'''    #T.C:-O(N**2)

'''from math import *
class Solution:
    def smallestDivisor(self, arr, k):
        # Code here
        low=1
        high=max(arr)
        while(low<=high):
            div=(low+high)//2
            Sum=0
            for num in arr:
                Sum+=ceil(num/div)
            if(Sum<=k):
                high=div-1
            else:
                low=div+1
        return low'''         #T.C:-O(logn)
                	        
# koko eating banana
'''from math import *
class Solution:
    def kokoEat(self,arr,k):
        # Code here
        for noOfBanana in range(1,max(arr)+1):
            time=0
            for num in arr:
                time+=ceil(num/noOfBanana)
            if(time<=k):
                return noOfBanana'''         #T.C:-O(N**2)

'''from math import *
class Solution:
    def kokoEat(self,arr,k):
        # Code here
        low=1
        high=max(arr)
        while(low<=high):
            noOfBanana=(low+high)//2
            time=0
            for num in arr:
                time+=ceil(num/noOfBanana)
            if(time<=k):
                high=noOfBanana-1
            else:
                low=noOfBanana+1
        return low  '''         #T.C:-O(logn)

# minimum days tpo make a  M bouquets
#22/05/25l
#countprime
'''def isPrime(num):
    c=0
    for i in range(1,num+1):
        if(num%i==0):
            c+=1
    return c==2
n=int(input())
count=0
for num in range(2,n):
    if(isPrime(num)):
        count+=1
print(count)'''    #T.C:-O(N**2)

'''n=int(input())
prime=[1]*n
for i in range(2,int(n**0.5)+1):
    if(prime[i==1]):
        for j in range(i*i,n,i):
            prime[j]=0
count=0
for i in range(2,n):
    if(prime[i]==1):
        count+=1
print(count)'''      #  T.C:-O(nlog(logn))
    

#search  a 2D matrix
'''def searchMatrix(self, matrix: List[List[int]], target: int) -> bool:
        n=len(matrix)
        m=len(matrix[0])
        row=0
        col=m-1
        while(row<n and col>=0):
            if(matrix[row][col]==target):
                return True
            elif(target>matrix[row][col]):
                row+=1
            elif(target<matrix[row][col]):
                col-=1
        return False'''

#rotate image
''' def rotate(self, matrix: List[List[int]]) -> None:
        """
        Do not return anything, modify matrix in-place instead.
        """
        n=len(matrix)
        m=len(matrix[0])
        dupMat=[]
        for i in range(n):
            temp=[0]*n
            dupMat.append(temp)
        for i in range(0,n):
            for j in range(0,n):
                dupMat[j][n-i-1]=matrix[i][j]
        for i in range(0,n):
            for j in range(0,n):
                matrix[i][j]=dupMat[i][j]'''      #T.C AND S.C:-O(N**2)

'''def rotate(self, matrix: List[List[int]]) -> None:
        """
        Do not return anything, modify matrix in-place instead.
        """
        n=len(matrix)
        for i in range(n-1):  
            for j in range(i+1,n):
                matrix[i][j],matrix[j][i]=matrix[j][i],matrix[i][j]
        for i in range(n):
            matrix[i]=matrix[i][::-1] # for reversing the rows'''        #T.C:-O(N**2) no space complexity

#24/05/25
'''class Solution:

    def aggressiveCows(self, stalls, k):
        # your code here
        def canWePlace(minDist,stalls,k):
            cow=stalls[0]
            placedCow=1
            for stall in range(1,len(stalls)):
                if(stalls[stall]-cow>=minDist):
                    cow=stalls[stall]
                    placedCows+=1
                if(placedCows==k):
                    return True
                return False
        stalls.sort()
        Min=min(stalls)
        Max=max(stalls)
        if(k==2):
            return Max-Min
        for minDist in range(1,Max-Min+1):
            if(canWePlace(minDist,stalls,k)):
                continue
            else:
                return minDist-1'''
                    
'''class Solution:

    def aggressiveCows(self, stalls, k):
        # your code here
        def canweplace(minDist,stalls,k):
            cow=stalls[0]
            placedCows=1
            for stall in range(1,len(stalls)):
                if(stalls[stall]-cow>=minDist):
                    cow=stalls[stall]
                    placedCows+=1
                if(placedCows==k):
                    return True
                return False
        stalls.sort()
        Min=min(stalls)
        Max=max(stalls)
        if(k==2):
            return Max-Min
        low=1
        high=max(stalls)
        while(low<=high):
            minDist=(low+high)//2
            if(canweplace(minDist,stalls,k)):
                low=minDist+1
            else:
                high=minDist-1'''#(optimal)


#allocate books               
'''def findPages(arr: [int], n: int, m: int) -> int:
    def canWeAllocate(maxPages,arr):
        no_of_pages_allocated=arr[0]
        students=1
        for pages in range(1,len(arr)):
            if(arr[pages]+no_of_pages_allocated<=maxPages):
                no_of_pages_allocated+=arr[pages]
            else:
                no_of_pages_allocated=arr[pages]
                students+=1
        return students
    if(m>len(arr)):
        return -1
    Min=max(arr)
    Max=sum(arr)
    for maxPages in range(Min,Max+1):
        if(canWeAllocate(maxPages,arr)<=m):
            return maxPages'''#(problem in code 360)

'''def findPages(arr: [int], n: int, m: int) -> int:
    def canWeAllocate(maxPages,arr):
        no_of_pages_allocated=arr[0]
        students=1
        for pages in range(1,len(arr)):
            if(arr[pages]+no_of_pages_allocated<=maxPages):
                no_of_pages_allocated+=arr[pages]
            else:
                no_of_pages_allocated=arr[pages]
                students+=1

        return students
    if(m>len(arr)):
        return -1
    low=max(arr)
    high=sum(arr)
    while(low<=high):
        maxPages=(low+high)//2
        if(canWeAllocate(maxPages,arr)<=m):
            high=maxPages-1
        else:
            low=maxPages+1
    return low'''         #optimal


#valild paranthesis
'''class Solution:
    def isValid(self, s: str) -> bool:
        stack=[] 
        for ele in s:
            if(ele in "([{"):
                stack.append(ele)
            else:
                if(len(stack)==0):
                    return False
                x=stack.pop()
                if(x=="(" and ele==")" or x=="[" and ele=="]" or x=="{" and ele=="}"):
                    continue
                else:
                    return False
        return len(stack)==0'''

# Trapping rain water
'''class Solution:
    def trap(self, height: List[int]) -> int:
        n=len(height)
        leftMax=[-1]*n
        leftMax[0]=height[0]
        for i in range(1,n):
            leftMax[i]=max(height[i],leftMax[i-1])
        rightMax=[-1]*n
        rightMax[n-1]=height[n-1]
        for i in range(n-2,-1,-1):
            rightMax[i]=max(height[i],rightMax[i+1])
        MinArray=[]
        for i in range(0,n):
            MinArray.append(min(rightMax[i],leftMax[i]))
        result=0
        for i in range(0,n):
            result+=MinArray[i]-height[i]
        return result'''
                
#26/05/25
#creation of linkedlist
'''class Node:
	   def __init__(self, data):   # data -> value stored in node
	        self.data = data
            self.next = None
	        
class Solution:
    def constructLL(self, arr):
        # code here
        head=None
        for data in arr:
            if(head==None):
                head=Node(data)
                temp=head
            else:
                temp.next=Node(data)
                temp=temp.next
        printLinkedList(head)
def printLinkedList(head):
    temp=head
    while(temp):
        print(str(temp.val)+"-->"+str(temp.next),end=" ")
        temp=temp.next
arr=list(map(int,input().split()))
Createlinkedlist(arr)
print(printLinkedList(head))'''

# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
'''class Solution:
    def middleNode(self, head: Optional[ListNode]) -> Optional[ListNode]:
        slow=head
        fast=head
        while(fast and fast.next):
            slow=slow.next
            fast=fast.next.next
        return slow   '''          #T.C:-O(N)
        
'''class Node:
	   def __init__(self, data):   # data -> value stored in node
	        self.data = data
            self.next = None
	        
class Solution:
    def constructLL(self, arr):
        # code here
        head=None
        for data in arr:
            if(head==None):
                head=Node(data)
                temp=head
            else:
                temp.next=Node(data)
                temp=temp.next
        #printLinkedList(head)
        middleNode=getMiddle(head)
        print(middleNode)
def getMiddle(head):
    slow=head
    fast=head
    while(fast and fast.next):
        slow=slow.next
        fast=fast.next
    return slow.val
def printLinkedList(head):
    temp=head
    while(temp):
        print(str(temp.val)+"-->"+str(temp.next),end=" ")
        temp=temp.next
arr=list(map(int,input().split()))
Createlinkedlist(arr)
print(printLinkedList(head)'''
'''#delete the middle term in linkedlist            
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def deleteMiddle(self, head: Optional[ListNode]) -> Optional[ListNode]:
        if(head==None or head.next==None):
            return None
        prev=None
        slow=head
        fast=head
        while(fast and fast.next):
            prev=slow
            slow=slow.next
            fast=fast.next.next
        prev.next=slow.next
        slow.next=None
        return head'''

#delete head in the linkedlist
'''
class Node:
    def __init__(self, data):
        self.data = data
        self.next = None

def deleteHead(head): 
    front=head.next
    head.next=None
    return front'''
#delete at last
'''class Node:
    def __init__(self, data):
        self.data = data
        self.next = None
    def createlinkedlist
    



arr=list(map(int,input().split()))'''

#insert at beginning
'''class Node:
    def __init__ (self, data):
        self.data=data
        self.next=None
        
class Solution:
    # Function to insert a node at the beginning of the linked list
    def insertAtBegining(self, head, x):
        newNode=Node(x)
        newNode.next=head
        head=newNode
        return head'''

#reverse linkedlist
'''#Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def reverseList(self, head: Optional[ListNode]) -> Optional[ListNode]:
        temp=head
        arr=[]
        while(temp):
            arr.append(temp.val)
            temp=temp.next
        arr=arr[::-1]
        i=0
        temp=head
        while(temp):
            temp.val=arr[i]
            i+=1
            temp=temp.next
        return head      ''' #T.C:-O(2N) AND S.C:-O(N)
            
'''class Solution:
    def reverseList(self, head: Optional[ListNode]) -> Optional[ListNode]:
        prev=None
        temp=head
        while(temp):
            front=temp.next
            temp.next=prev
            prev=temp
            temp=front
        return prev   '''

#linkedlist cycle
'''# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution:
    def hasCycle(self, head: Optional[ListNode]) -> bool:
        slow=head
        fast=head
        while(fast and fast.next):
            slow=slow.next
            fast=fast.next.next
            if(slow==fast):
                return True
        return False  '''      #T.C:-O(N) 

#27/05/25       
#linked list cycle-2
'''#  Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution:
    def detectCycle(self, head: Optional[ListNode]) -> Optional[ListNode]:
        slow=head
        fast=head
        while(fast and fast.next):
            slow=slow.next
            fast=fast.next.next
            if(slow==fast):
                slow=head
                while(slow!=fast):
                    slow=slow.next
                    fast=fast.next
                return slow
        return None'''       #T.C:-O(N) 

#find length of loop/cycle
'''# Node Class
class Node:
    def __init__(self, data):   # data -> value stored in node
        self.data = data
        self.next = None
class Solution:
    # Function to find the length of a loop in the linked list.
    def countNodesInLoop(self, head):
        #code here
        slow=head
        fast=head
        while(fast and fast.next):
            slow=slow.next
            fast=fast.next.next
            if(slow==fast):
                slow=head
                while(slow!=fast):
                    slow=slow.next
                    fast=fast.next
                count=1
                slow=slow.next
                while(slow!=fast):
                    slow=slow.next
                    count+=1
                return count
        return 0  '''

# DFS(in.pre and post orders)
'''def inorder(root):
    if(root==None):
        return
    inorder(root.left)
    print(root.val)
    inorder(root.right)

def preorder(root):
    if(root==None):
        return
    print(root.val)
    preorder(root.left)    
    preorder(root.right)

def postorder(root):
    if(root==None):
        return
    postorder(root.left)
    postorder(root.right)
    print(root.val)  '''       


#BFS(level order traversal) most comon quest in interviews
'''class Solution:
    def levelOrder(self, root: Optional[TreeNode]) -> List[List[int]]:
        if(root==None):
            return []
        q=[root]
        ans=[]
        while(q):
            level=[]
            for i in range(len(q)):
                node=q.pop(0)
                if(node.left):
                    q.append(node.left)
                if(node.right):
                    q.append(node.right)
                level.append(node.val)
            ans.append(level)
        return ans  '''

#height of the tree
'''def findheight(root):
    if(root==None):
        return 0
    lh=findheight(root.left)
    rh=findheight(root.right)
    return 1+max(lh,rh)  
return findheight(root)'''

# Binary Tree Zigzag Level Order Traversal

'''class Solution:
    def zigzagLevelOrder(self, root: Optional[TreeNode]) -> List[List[int]]:
        if(root==None):
            return []
        q=[root]
        ans=[]
        while(q):
            level=[]
            for i in range(len(q)):
                node=q.pop(0)
                if(node.left):
                    q.append(node.left)
                if(node.right):
                    q.append(node.right)
                level.append(node.val)
            ans.append(level)
        for i in range(len(ans)):
            if(i%2==1):
                ans[i]=ans[i][::-1]
        return ans    '''

#234. Palindrome Linked List
'''class Solution:
    def isPalindrome(self, head: Optional[ListNode]) -> bool:
        arr=[]
        temp=head
        while(temp):
            arr.append(temp.val)
            temp=temp.next
        return arr==arr[::-1]  '''


#148. Sort List
'''# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def sortList(self, head: Optional[ListNode]) -> Optional[ListNode]:
        arr=[]
        temp=head
        while(temp):
            arr.append(temp.val)
            temp=temp.next
        arr.sort()
        ind=0
        temp=head
        while(temp):
            temp.val=arr[ind]
            ind+=1
            temp=temp.next
        return head      '''

#coversion of arr to binary search tree
'''class Node:
    def __init__ (self,data):
        self.left=None
        self.val=data
        self.right=None
def createBST(arr):
    root=None
    for data in arr:
        if(root==None):
            root=Node(data)
        else:
            temp=root
            while(True):
                #smaller elements
                if(data<temp.val):
                    if(temp.left==None):
                        temp.left=Node(data)
                        break
                    else:
                        temp=temp.left
                #larger elements
                if(data>temp.val):
                    if(temp.right==None):
                        temp.right=Node(data)
                        break
                    else:
                        temp=temp.right
    print(root.left.right.val)
arr=list(map(int,input().split()))
createBST(arr)   '''

# Doublelinkedlist
'''class Node:
    def __init__ (self,data):
        self.prev=None
        self.val=data
        self.next=None
def createDoublyLinkedList(arr):
        head=None
        for data in arr:
            if(head==None):
                head=Node(data)
                temp=head
            else:
                newNode=Node(data)
                newNode.prev=temp
                temp.next=newNode
                temp=temp.next
arr=list(map(int,input().split()))
createDoublyLinkedList(arr)  '''
                
#coversion of arr to binary search tree using inorder
'''class Node:
    def __init__ (self,data):
        self.left=None
        self.val=data
        self.right=None
def createBST(arr):
    root=None
    for data in arr:
        if(root==None):
            root=Node(data)
        else:
            temp=root
            while(True):
                #smaller elements
                if(data<temp.val):
                    if(temp.left==None):
                        temp.left=Node(data)
                        break
                    else:
                        temp=temp.left
                #larger elements
                if(data>temp.val):
                    if(temp.right==None):
                        temp.right=Node(data)
                        break
                    else:
                        temp=temp.right
    inorder(root)
def inorder(root):
    if(root==None):
        return 
    inorder(root.left)
    print(root.val,end=" ")
    inorder(root.right)


arr=list(map(int,input().split()))
createBST(arr)   '''
        
#28/05/25
#Graph
'''from typing import List
class Solution:
    def printGraph(self, V : int, edges : List[List[int]]) -> List[List[int]]:
        # code here
        adjList=[]
        for i in range(V):
            adjList.append([])
        for n,m in edges:
            adjList[n].append(m)
            adjList[m].append(n)
        for lst in adjList:
            lst.sort()
        return adjList   ''' #gfg problem
        
#BFS of graph (connected graph)
'''class Solution:
    # Function to return Breadth First Search Traversal of given graph.
    def bfs(self, adj):
        # code here
        V=len(adj)
        visited=[0]*V
        startedNode=0
        ans=[]
        q=[]
        if(visited[startedNode]==0):
            visited[startedNode]=1
            q=[startedNode]
            while(q):
                node=q.pop(0)
                ans.append(node)
                for i in adj[node]:
                    if(visited[i]==0):
                        visited[i]=1
                        q.append(i)
            return ans      '''   #gfg problem

#BFS of graph (unconnected graph)
'''class Solution:
    # Function to return Breadth First Search Traversal of given graph.
    def bfs(self, adj):
        # code here
        V=len(adj)
        visited=[0]*V
        startedNode=0
        ans=[]
        q=[]
        for startNode in range(0,V):
            if(visited[startedNode]==0):
                visited[startedNode]=1
                q=[startedNode]
                while(q):
                    node=q.pop(0)
                    ans.append(node)
                    for i in adj[node]:
                        if(visited[i]==0):
                            visited[i]=1
                            q.append(i)
        return ans  '''

#DFS in graphs
'''class Solution:
    def depthFirstSearch(self,Node,visited,adj,ans):
        visited[Node]=1
        ans.append(Node)
        for i in adj[Node]:
            if(visited[i]==0):
                self.depthFirstSearch(i,visited,adj,ans)
    #Function to return a list containing the DFS traversal of the graph.
    def dfs(self, adj):
        # code here
        V=len(adj)
        visited=[0]*V
        ans=[]
        Node=0
        if(visited[Node]==0):
            self.depthFirstSearch(Node,visited,adj,ans)
        return ans   '''

#number of islands
'''class Solution:
    def dfs(self,i,j,grid,visited,n,m):
        visited[i][j]=1
        for row,col in [[-1,0],[1,0],[0,-1],[0,1]]:
            delRow=i+row
            delCol=j+col
            if(delRow>=0 and delRow<n and delCol>=0 
            and delCol<m and grid[delRow][delCol]=="1" 
            and visited[delRow][delCol]==0):
                self.dfs(delRow,delCol,grid,visited,n,m)
    def numIslands(self, grid: List[List[str]]) -> int:
        n=len(grid) # rows 
        m =len(grid[0]) # cols 
        visited=[]
        for i in range(n):
            temp=[0]*m
            visited.append(temp)
        count=0
        for i in range(n):
            for j in range(m):
                if(grid[i][j]=="1" and visited[i][j]==0):
                    self.dfs(i,j,grid,visited,n,m)
                    count+=1
        return count  '''

#no.of bouquets
#User function Template for python3
'''class Solution:
    def minDaysBloom(self, m, k, arr):
        # Code here
        if(m>len(arr)):
            return -1 
        Min=min(arr) 
        Max=max(arr) 
        for bloomday in range(Min,Max+1):
            count=0 
            noOfB=0 
            for flower in arr:
                if(flower<=bloomday):
                    count+=1 
                else:
                    noOfB+=count//k 
                    count=0 
            noOfB+=count//k 
            if(noOfB>=m):
                return bloomday 
        return -1 
#User function Template for python3

----------------------------------------------------------
class Solution:
    def minDaysBloom(self, m, k, arr):
        # Code here
        if(m>len(arr)):
            return -1 
        low=min(arr) 
        high=max(arr) 
        ans = -1 
        while(low<=high):
            bloomday=(low+high)//2 
            count=0 
            noOfB=0 
            for flower in arr:
                if(flower<=bloomday):
                    count+=1 
                else:
                    noOfB+=count//k 
                    count=0 
            noOfB+=count//k 
            if(noOfB<m):
                low=bloomday+1 
            else:
                ans=bloomday 
                high=bloomday-1 
        return ans   '''

#create subarrays from array
'''arr=list(map(int,input().split()))
n=len(arr)
for i in range(0,n):
    for j in range(i,n):
        print(arr[i:j+1])  '''

#constant window(maxSum)
'''arr=list(map(int,input().split()))
k=int(input())
n=len(arr)
maxSum=0
for i in range(0,n):
    for j in range(i,n):
        if(len(arr[i:j+1])==k):
            maxSum=max(maxSum,sum(arr[i:j+1])) 
print(maxSum)  ''' # T.C:-O(N**2)

'''arr=list(map(int,input().split()))
k=int(input())
n=len(arr)
left=0
right=k-1
Sum=sum(arr[left:right+1])
maxSum=Sum
while(right<n-1):
    Sum-=arr[left]
    left+=1
    right+=1
    Sum+=arr[right]
    maxSum=max(maxSum,Sum)
print(maxSum) '''      #optimal solution

# for maxlength
'''arr=list(map(int,input().split()))
n=len(arr)
k=int(input())
maxLen=0
for i in range(0,n):
    for j in range(i,n):
        if(sum(arr[i:j+1])<=k):
            maxLen=max(maxLen,j-i+1)
print(maxLen) ''' #T.C:-0(N**2)

'''arr=list(map(int,input().split()))
n=len(arr)
k=int(input())
maxLen=0
left=0
right=0
Sum=0
while(right<n):
    Sum+=arr[right]  #expand
    while(Sum>k):
        Sum-=arr[left]  #shrink
        left+=1
    maxLen=max(maxLen,right-left+1)
    right+=1
print(maxLen)  ''' #T.C:-O(N)

#longest length or smallest length
'''arr=list(map(int,input().split()))
n=len(arr)
k=int(input())
maxLen=0
left=0
right=0
Sum=0
while(right<n):
    Sum+=arr[right]  #expand
    if(Sum>k):
        Sum-=arr[left]  #shrink
        left+=1
    maxLen=max(maxLen,right-left+1)
    right+=1
print(maxLen)  '''  #T.C:-O(N)

#example for constant window pattern lc:1423
'''class Solution:
    def maxScore(self, cardPoints: List[int], k: int) -> int:
        n=len(cardPoints)
        left=0
        right=k-1
        leftSum=sum(cardPoints[left:right+1])
        rightSum=0
        maxSum=leftSum
        rightIndex=n-1
        for i in range(k-1,-1,-1):
            leftSum-=cardPoints[i]
            rightSum+=cardPoints[rightIndex]
            maxSum=max(maxSum,leftSum+rightSum)
            rightIndex-=1
        return maxSum ''' #T.C:-O(N)
#pattern-2(longest substring without repeating the characters)
'''class Solution(object):
    def lengthOfLongestSubstring(self, s):
        n=len(s)
        maxLen=0
        for i in range(0,n):
            checker=[0]*256
            for j in range(i,n):
                if(checker[ord(s[j])]==1):
                    break
                checker[ord(s[j])]=1
                maxLen=max(maxLen,j-i+1)
        return maxLen'''  #lc-3 T.C:-O(N**2) AND S.C:-O(1)

'''class Solution(object):
    def lengthOfLongestSubstring(self, s):
        n=len(s)
        maxLen=0
        left=0
        right=0
        d={}
        while(right<n):
            if(s[right] in d):
                if(d[s[right]]>=left):
                    left=d[s[right]]+1
            d[s[right]]=right
            maxLen=max(maxLen,right-left+1)
            right+=1
        return maxLen '''  #optimal solution T.C:-O(N)
#1004 max consecutive ones 111
'''class Solution:
    def longestOnes(self, nums: List[int], k: int) -> int:
        n=len(nums)
        maxLen=0
        for i in range(0,n):
            zero_count=0
            for j in range(i,n):
                if(nums[j]==0):
                    zero_count+=1
                if(zero_count>k):
                    break
                maxLen=max(maxLen,j-i+1)
        return maxLen '''

'''class Solution:
    def longestOnes(self, nums: List[int], k: int) -> int:
        n=len(nums)
        left=0
        right=0
        maxLen=0
        count_zeros=0
        while(right<n):
            if(nums[right]==0):
                count_zeros+=1
            if(count_zeros>k):   #if we remove this if condition then also it generate correct output
                while(count_zeros>k):
                    if(nums[left]==0):
                        count_zeros-=1
                    left+=1
            maxLen=max(maxLen,right-left+1)
            right+=1
        return maxLen ''' #optimal code T.C:-O(N)
       
#904 fruits into baskets
'''class Solution:
    def totalFruit(self, fruits: List[int]) -> int:
        n=len(fruits)
        maxLen=0
        for i in range(0,n):
            s=set()
            for j in range(i,n):
                s.add(fruits[j])
                if(len(s)>2):
                    break
                maxLen=max(maxLen,j-i+1)
        return maxLen  ''' #T.C:-O(N**2),S.C:-O(N)

'''class Solution:
    def totalFruit(self, fruits: List[int]) -> int:
        n=len(fruits)
        left=0
        right=0
        maxLen=0
        d={}
        while(right<n):
            if(fruits[right] in d):
                d[fruits[right]]+=1
            else:
                d[fruits[right]]=1 
            if(len(d)>2):
                while(len(d)>2):
                    d[fruits[left]]-=1
                    if(d[fruits[left]]==0):
                        del d[fruits[left]]
                    left+=1
            maxLen=max(maxLen,right-left+1)
            right+=1
        return maxLen  ''' #optimal

#longest substring with k uniques             
'''class Solution:

    def longestKSubstr(self, s, k):
        # code here
        n=len(s)
        if(n==1):
            return 1+
        dup=sorted(s)
        if(dup[0]==dup[n-1]):
            return -1
        if(len(set(dup))<k):
            return -1
        maxLen=0
        for i in range(0,n):
            hash_set=set()
            for j in range(i,n):
                hash_set.add(s[j])
                if(len(hash_set)>k):
                    break
                maxLen=max(maxLen,j-i+1)
        return maxLen  ''' #T.C:-O(N**2) gfg
            

'''class Solution:

    def longestKSubstr(self, s, k):
        # code here
        n=len(s)
        if(n==1):
            return 1
        dup=sorted(s)
        if(dup[0]==dup[n-1]):
            return -1
        if(len(set(dup))<k):
            return -1
        maxLen=0
        left=0
        right=0
        
        d={}
        while(right<n):
            if(s[right] in d):
                d[s[right]]+=1
            else:
                d[s[right]]=1 
            if(len(d)>k):
                while(len(d)>k):
                    d[s[left]]-=1
                    if(d[s[left]]==0):
                        del d[s[left]]
                    left+=1
            maxLen=max(maxLen,right-left+1)
            right+=1
        return maxLen  ''' #optimal
              
 #maximum subarray(53)  
'''class Solution:
    def maxSubArray(self, nums: List[int]) -> int:
        n=len(nums)
        maxSum=float("-inf")
        for i in range(0,n):
            for j in range(i,n):
                Sum=sum(nums[i:j+1])
                maxSum=max(maxSum,Sum)
        return maxSum  '''

#maximum subarray using kadanes algorithim
'''class Solution:
    def maxSubArray(self, nums: List[int]) -> int:
        n=len(nums)
        maxSum=float("-inf")
        currentSum=0
        for  i in nums:
            currentSum+=i
            maxSum=max(maxSum,currentSum)
            if(currentSum<0):
                currentSum=0
        return maxSum    ''' #optimal

#longest subarray with sum k
'''class Solution:
    def longestSubarray(self, arr, k):  
        # code here
        n=len(arr)
        d={}
        Sum=0
        maxLen=0
        for i in range(0,n):
            Sum+=arr[i]
            if(Sum==k):
                maxLen=i+1
            rem = Sum-k
            if(rem in d):
                maxLen = max(maxLen,i-d[rem])
            if(Sum not in d):
                d[Sum]=i
        return maxLen '''

