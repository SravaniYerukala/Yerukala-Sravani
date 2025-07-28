class TreeNode:
        def __init__(self,val):   
          self.data = val
          self.left = None
          self.right = None

def createbt(arr,i,n):
        if i>n:
            return None
        root=TreeNode(arr[i-1])
        root.left=createbt(arr,2*i,n)
        root.right=createbt(arr,2*i+1,n)
        return root
def inorder(root):
        if root is None:
            return
        print(root.data,end=" ")
        inorder(root.left)
        inorder(root.right)

arr=[8,3,10,1,4,7]
binary_Tree=createbt(arr,1,len(arr))
print(binary_Tree)
inorder(binary_Tree)        

        