'''n=int(input())
for i in range(1,n+1):
    for j in range(n-i):
        print(" ",end=" ")
    for j in range(2*i-1):
        print("*",end=" ")
    print()   ''' 
     
       # * 
      #* * * 
    #* * * * * 
  #* * * * * * * 
#* * * * * * * * * 

'''n=int(input())
for i in range(1,n+1):
    for j in range(n-i):
        print(" ",end=" ")
    for j in range(2*1-1):
        print("*",end=" ")
    print() '''

        #*
      #*
    #*
  #*
#*

'''n=int(input())
for i in range(1,n+1):
    for j in range(n-i):
        print(" ",end=" ")
    for j in range(2*i-i):
        print("*",end=" ")
    print() '''



        #*
      #* *
    #* * *
  #* * * *
#* * * * *

'''n=int(input())
for i in range(1,n+1):
    for j in range(1,i+1):
        print("*",end=" ")
    print()  '''
               
#*
#* *
#* * *
#* * * *

'''n=int(input())
for i in range(1,n+1):
    for j in range(i,n+1):
        print("*",end=" ")
    print() '''

#* * * *
#* * *
#* *
#*                
             
'''n=int(input())
for i in range(1,n+1):
    for j in range(1,n+1):
        print("*",end=" ")
    print()  '''

#* * * *
#* * * *
#* * * *
#* * * *