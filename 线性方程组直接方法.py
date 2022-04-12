import numpy as ny
def GAUSS(A,b):
    A,b=ny.array(A,dtype=float),ny.array(b,dtype=float)
    h,l=A.shape
    n=b.shape[0]
    b=b.reshape(n,1)
    if h!=l or l!=n:
        return "你输入的不对"
    else:
        A=ny.c_[A,b]
        for k in range(h):
            a=A[k:n,k]#13-19选主元
            a=ny.abs(a)
            dt=ny.max(a,axis=0)
            t=ny.argmax(a,axis=0)
            if dt>a[0]:
                A[(k,t+k),:]=A[(t+k,k),:]
            for t in range(k,h-1):
                A[(t+1),:]=A[(t+1),:]-A[k,:]*A[(t+1),k]/A[k,k]
        print(A) #输出对角阵
        x=ny.zeros(h)
        s=ny.zeros(h+1)
        for k in range(h-1,-1,-1):
            x[k]=(A[k,h]-s[k+1])/A[k,k]
            if ny.abs(x[k])<1.0e-10:#极小化为0
                x[k]=0
            if k>=1:
                for i in range(k,h):
                    s[k]=s[k]+A[k-1,i]*x[i]
    return x.reshape(n,1)
def Crout(a,c,d,b):
    a,b,c,d=ny.array(a,dtype=float),ny.array(b,dtype=float),ny.array(c,dtype=float),ny.array(d,dtype=float)
    n=ny.alen(a)
    aalpha=ny.zeros(n)
    bbeta=ny.zeros(n-1)
    dd=ny.zeros(n)
    for q in range(1,n):
        dd[q]=d[q-1]
    aalpha[0]=a[0]
    for i in range(0,n-1):
        bbeta[i]=c[i]/aalpha[i]
        aalpha[i+1]=a[i+1]-dd[i+1]*bbeta[i]
    y=ny.zeros(n)
    y[0]=b[0]/aalpha[0]
    for j in range(1,n):
        y[j]=(b[j]-dd[j]*y[j-1])/aalpha[j]
    x=ny.zeros(n)
    x[n-1]=y[n-1]
    for k in range(n-2,-1,-1):
        x[k]=y[k]-bbeta[k]*x[k+1]
    x=x.reshape(n,1)
    for p in range(n):
        if ny.abs(x[p])<1.0e-12:
            x[p]=0
    return x
def Doolittle(A,b):
    A,b=ny.array(A,dtype=float),ny.array(b,dtype=float)
    n=b.shape[0]
    #b=b.reshape(n,1)
    for t in range(1,n):
        A[t,0]=A[t,0]/A[0,0]
    for k in range(1,n):
        for j in range(k,n):
            s=0
            for m in range(k):
                s=s+A[k,m]*A[m,j]
            A[k,j]=A[k,j]-s
        for i in range(k+1,n):
            s=0
            for m in range(k):
                s=s+A[i,m]*A[m,k]
            A[i,k]=(A[i,k]-s)/A[k,k]
    print(A)
    y=ny.zeros(n)
    y[0]=b[0]
    for i in range(1,n):
        s=0
        for m in range(i):
            s=s+A[i,m]*y[m]
        y[i]=b[i]-s
    x=ny.zeros(n)
    x[n-1]=y[n-1]/A[n-1,n-1]
    for i in range(n-2,-1,-1):
        s=0
        for m in range(i+1,n):
            s=s+A[i,m]*x[m]
        x[i]=(y[i]-s)/A[i,i]
    x=x.reshape((n,1))
    print(x)
def Cholesky(A,b):
    A,b=ny.array(A,dtype=float),ny.array(b,dtype=float)
    n=b.shape[0]
    x,y=ny.zeros(n,dtype=float),ny.zeros(n,dtype=float)
    b=b.reshape(n,1)
    for k in range(n):
        s=0
        for m in range(k):
            s=s+(A[k,m])**2
        A[k,k]=ny.sqrt(A[k,k]-s)
        for i in range(k+1,n):
            s=0
            for m in range(k):
                s=s+A[i,m]*A[k,m]
            A[i,k]=(A[i,k]-s)/A[k,k]
        s=0
        for m in range(k):
            s=s+A[k,m]*y[m]
        y[k]=(b[k]-s)/A[k,k]
    print(y)
    x[n-1]=y[n-1]/A[n-1,n-1]
    for k in range(n-1,-1,-1):
        s=0
        for m in range(k+1,n):
            s=s+A[m,k]*x[m]
        x[k]=(y[k]-s)/A[k,k]
    x=x.reshape((n,1))
    print(x)        
'''
a,b,c=ny.zeros(10),ny.zeros(9),ny.zeros(9)
for i in range(10):
    a[i]=4
for j in range(9):
    c[j],b[j]=-1,-1
d=[7,5,-13,2,6,-12,14,-4,5,-5]
A=[[4,2,-3,-1,2,1,0,0,0,0],
    [8,6,-5,-3,6,5,0,1,0,0],
    [4,2,-2,-1,3,2,-1,0,3,1],
    [0,-2,1,5,-1,3,-1,1,9,4],
    [-4,2,6,-1,6,7,-3,3,2,3],
    [8,6,-8,5,7,17,2,6,-3,5],
    [0,2,-1,3,-4,2,5,3,0,1],
    [16,10,-11,-9,17,34,2,-1,2,2],
    [4,6,2,-7,13,9,2,0,12,4],
    [0,0,-1,8,-3,-24,-8,6,3,-1]]
B=[5,12,3,2,3,46,13,38,19,-21]
print(GAUSS(A,B))
print("===============================")
print(Crout(a,b,c,d))
A=[[4,2,-4,0,2,4,0,0],
    [2,2,-1,-2,1,3,2,0],
    [-4,-1,14,1,-8,-3,5,6],
    [0,-2,1,6,-1,-4,-3,3],
    [2,1,-8,-1,22,4,-10,-3],
    [4,3,-3,-4,4,11,1,-4],
    [0,2,5,-3,-10,1,14,2],
    [0,0,6,3,-3,-4,2,19]]
#A=[[2,1,1],[4,4,3],[6,7,7]]
A=[[4,2,-3,-1,2,1,0,0,0,0],
    [8,6,-5,-3,6,5,0,1,0,0],
    [4,2,-2,-1,3,2,-1,0,3,1],
    [0,-2,1,5,-1,3,-1,1,9,4],
    [-4,2,6,-1,6,7,-3,3,2,3],
    [8,6,-8,5,7,17,2,6,-3,5],
    [0,2,-1,3,-4,2,5,3,0,1],
    [16,10,-11,-9,17,34,2,-1,2,2],
    [4,6,2,-7,13,9,2,0,12,4],
    [0,0,-1,8,-3,-24,-8,6,3,-1]]
B=[5,12,3,2,3,46,13,38,19,-21]
print(GAUSS(A,B))
a,b,c=ny.zeros(10),ny.zeros(9),ny.zeros(9)
for i in range(10):
    a[i]=4
for j in range(9):
    c[j],b[j]=-1,-1
d=[7,5,-13,2,6,-12,14,-4,5,-5]
print(Crout(a,b,c,d))'''
#Doolittle(A1,b1)
A=[[4,2,-4,0,2,4,0,0],
    [2,2,-1,-2,1,3,2,0],
    [-4,-1,14,1,-8,-3,5,6],
    [0,-2,1,6,-1,-4,-3,3],
    [2,1,-8,-1,22,4,-10,-3],
    [4,3,-3,-4,4,11,1,-4],
    [0,2,5,-3,-10,1,14,2],
    [0,0,6,3,-3,-4,2,19]]
b=[0,-6,6,23,11,-22,-15,45]
Doolittle(A,b)