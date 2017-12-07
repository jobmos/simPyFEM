#importing packages etc.
print("importing pacakages")

import numpy as numpy
import math as math
import sys as sys
import scipy as scipy
from scipy import sparse as sparse

#variable declaration
print("variable declaration")

m=4                 # nodes that make up an element
ndof=2              # degrees of freedom per node

Lx=1                # nondimensionalized size of system, x
Ly=1                # nondimensionalized size of system, y

nnx=31              # resolution, x
nny=31              # resolution, y

np=nnx*nny          # number of points

nelx=nnx-1          # number of elements, x
nely=nny-1          # number of elements, y

nel=nelx*nely       # number of elements, total

penalty=1.0e7       # penalty 

viscosity=1.0       # constant viscosity
density=1.0         # constant density  

Nfem=np*ndof        # Total number of degrees of freedom

eps=1.0e-10

gx=0                # gravity, x
gy=1                # gravity, y

#declaring arrays
print("declaring arrays")

x      = numpy.zeros((np),dtype=float)         # x coordinates
y      = numpy.copy(x)                         # y coordinates
u      = numpy.copy(x)                         # x velocities
v      = numpy.copy(x)                         # y velocities
A      = numpy.zeros((Nfem,Nfem),dtype=float)  # matrix of Ax=b
B      = numpy.zeros((Nfem),dtype=float)       # righthand side of Ax=b
sol    = numpy.zeros((Nfem),dtype=float)       # solution vector of Ax=b
icon   = numpy.zeros((m,nel),dtype=int)        # connectivity
bc_fix = numpy.zeros((Nfem),dtype=bool)        # boundary condition, yes/no
bc_val = numpy.zeros((Nfem),dtype=float)       # boundary condition, value
N      = numpy.zeros((m),dtype=float)          # shape functions
dNdx   = numpy.copy(N)                         # derivative of shape functions
dNdy   = numpy.copy(N)                         # " "
dNdr   = numpy.copy(N)                         # " "
dNds   = numpy.copy(N)                         # " "
jcb    = numpy.zeros((2,2),dtype=float)        # jacobian matrix
jcbi   = numpy.copy(jcb)                       # inverse of jacobian matrix
Bmat   = numpy.zeros((3,ndof*m),dtype=float)   # numerical integration matrix
Kmat   = numpy.zeros((3,3),dtype=float)        # " "
Cmat   = numpy.copy(Kmat)                      # " "

Kmat[0,0]=1.0 ; Kmat[0,1]=1.0 ; Kmat[0,2]=0.0  
Kmat[1,0]=1.0 ; Kmat[1,1]=1.0 ; Kmat[1,2]=0.0  
Kmat[2,0]=0.0 ; Kmat[2,1]=0.0 ; Kmat[2,2]=0.0  

Cmat[0,0]=2.0 ; Cmat[0,1]=0.0 ; Cmat[0,2]=0.0  
Cmat[1,0]=0.0 ; Cmat[1,1]=2.0 ; Cmat[1,2]=0.0  
Cmat[2,0]=0.0 ; Cmat[2,1]=0.0 ; Cmat[2,2]=1.0  

#grid point setup
print("grid point setup")
counter=0
for j in range (0,nely+1):
    for i in range (0,nelx+1):
        x[counter]=i*Lx/nelx
        y[counter]=j*Ly/nely
        counter += 1

#connectivity
print("connectivity")

counter=0
for j in range (0,nely):
    for i in range (0,nelx):
        icon[0,counter]=i+   j*(nelx+1)
        icon[1,counter]=i+1+ j*(nelx+1)
        icon[2,counter]=i+1+(j+1)*(nelx+1)
        icon[3,counter]=i  +(j+1)*(nelx+1)
        counter += 1
        
# for iel in range (0,nel):
#         print ("iel=",iel)
#         print ("node 1",icon[0][iel],"at pos.",x[icon[0][iel-1]], y[icon[0][iel-1]])
#         print ("node 2",icon[1][iel],"at pos.",x[icon[1][iel-1]], y[icon[1][iel-1]])
#         print ("node 3",icon[2][iel],"at pos.",x[icon[2][iel-1]], y[icon[2][iel-1]])
#         print ("node 4",icon[3][iel],"at pos.",x[icon[3][iel-1]], y[icon[3][iel-1]])

#defining boundary conditions
print("defining boundary conditions")

for i in range (0,np):
    if x[i] < eps:
        bc_fix[i*ndof] = True ; bc_val[i*ndof]  =0.0
        bc_fix[i*ndof+1]=True ; bc_val[i*ndof+1]=0.0
    if x[i] > (Lx-eps):
        bc_fix[i*ndof] = True ; bc_val[i*ndof]  =0.0
        bc_fix[i*ndof+1]=True ; bc_val[i*ndof+1]=0.0
    if y[i] < eps:
        bc_fix[i*ndof] = True ; bc_val[i*ndof]  =0.0
        bc_fix[i*ndof+1]=True ; bc_val[i*ndof+1]=0.0
    if y[i] > (Ly-eps):
        bc_fix[i*ndof] = True ; bc_val[i*ndof]  =0.0
        bc_fix[i*ndof+1]=True ; bc_val[i*ndof+1]=0.0

#building FE matrix
print("building FE matrix")

for iel in range (0,nel):
        
    #setting 2 arrays to 0 every loop
    Bel = [0 for i in range(0,m*ndof)]                #using list comprehension
    Ael = numpy.zeros((m*ndof,m*ndof),dtype=float)    #can we also use 2D list comprehension?
    
    #integrate viscous term at 4 quadrature points
    for iq in range (-1,3,2):
        for jq in range (-1,3,2):
            rq=iq/math.sqrt(3.0)
            sq=jq/math.sqrt(3.0)
            
            wq=1.0*1.0

            #shape functions
            N[0]=0.25*(1.0-rq)*(1.0-sq)
            N[1]=0.25*(1.0+rq)*(1.0-sq)
            N[2]=0.25*(1.0+rq)*(1.0+sq)
            N[3]=0.25*(1.0-rq)*(1.0+sq)

            #shape function derivatives
            dNdr[0]= - 0.25*(1.0-sq) ; dNds[0]= - 0.25*(1.0-rq)
            dNdr[1]= + 0.25*(1.0-sq) ; dNds[1]= - 0.25*(1.0+rq)
            dNdr[2]= + 0.25*(1.0+sq) ; dNds[2]= + 0.25*(1.0+rq)
            dNdr[3]= - 0.25*(1.0+sq) ; dNds[3]= + 0.25*(1.0-rq)
            
            #jacobian matrix
            jcb = numpy.zeros((2,2),dtype=float)
            for k in range (0,m):
                jcb[0,0]=jcb[0,0]+dNdr[k]*x[icon[k,iel]]
                jcb[0,1]=jcb[0,1]+dNdr[k]*y[icon[k,iel]]
                jcb[1,0]=jcb[1,0]+dNds[k]*x[icon[k,iel]]
                jcb[1,1]=jcb[1,1]+dNds[k]*y[icon[k,iel]]
    
            #determinant of the jacobian
            jcob=jcb[0,0]*jcb[1,1]-jcb[1,0]*jcb[0,1]

            #inverse of the jacobian matrix
            jcbi[0,0]=    jcb[1,1] / jcob
            jcbi[0,1]=  - jcb[0,1] / jcob
            jcbi[1,0]=  - jcb[1,0] / jcob
            jcbi[1,1]=    jcb[0,0] / jcob
            
            #computing dNdx & dNdy
            xq=0.0
            yq=0.0
            uq=0.0
            vq=0.0
            exxq=0.0
            eyyq=0.0
            exyq=0.0
            for k in range(0,m):
                xq=xq+N[k]*x[icon[k,iel]]
                yq=yq+N[k]*y[icon[k,iel]]
                uq=uq+N[k]*u[icon[k,iel]]
                vq=vq+N[k]*v[icon[k,iel]]
                dNdx[k]=jcbi[0,0]*dNdr[k]+jcbi[0,1]*dNds[k]
                dNdy[k]=jcbi[1,0]*dNdr[k]+jcbi[1,1]*dNds[k]
                exxq=exxq+ dNdx[k]*u[icon[k,iel]]
                eyyq=eyyq+ dNdy[k]*v[icon[k,iel]]
                exyq=exyq+ dNdx[k]*v[icon[k,iel]] *0.5 + dNdy[k]*u[icon[k,iel]] *0.5
            
            #constructing 3x8 B matrix
            for i in range(0,m):
                i1=2*i
                i2=2*i+1
                Bmat[0,i1]=dNdx[i] ; Bmat[0,i2]=0
                Bmat[1,i1]=0       ; Bmat[1,i2]=dNdy[i]
                Bmat[2,i1]=dNdy[i] ; Bmat[2,i2]=dNdx[i]
            
            #computing elemental A matrix
            Ael=Ael+numpy.matmul(Bmat.transpose(),numpy.matmul(viscosity*Cmat,Bmat))*wq*jcob
            
            #computing elemental B matrix
            for i in range(0,m):
                i1=2*i
                i2=2*i+1
                Bel[i1]=Bel[i1]+N[i]*jcob*wq*density*gx
                Bel[i2]=Bel[i2]+N[i]*jcob*wq*density*gy
    
    #integrate penalty term at 1 point
    rq=0.0
    sq=0.0
    wq=2.0*2.0
            
    N[0]=0.25*(1.0-rq)*(1.0-sq)
    N[1]=0.25*(1.0+rq)*(1.0-sq)
    N[2]=0.25*(1.0+rq)*(1.0+sq)
    N[3]=0.25*(1.0-rq)*(1.0+sq)

    dNdr[0]= - 0.25*(1.0-sq) ; dNds[0]= - 0.25*(1.0-rq)
    dNdr[1]= + 0.25*(1.0-sq) ; dNds[1]= - 0.25*(1.0+rq)
    dNdr[2]= + 0.25*(1.0+sq) ; dNds[2]= + 0.25*(1.0+rq)
    dNdr[3]= - 0.25*(1.0+sq) ; dNds[3]= + 0.25*(1.0-rq)
            
    for k in range(0,m):
        jcb[0,0]=jcb[0,0]+dNdr[k]*x[icon[k,iel]]
        jcb[0,1]=jcb[0,1]+dNdr[k]*y[icon[k,iel]]
        jcb[1,0]=jcb[1,0]+dNds[k]*x[icon[k,iel]]
        jcb[1,1]=jcb[1,1]+dNds[k]*y[icon[k,iel]]
                
    jcob=jcb[0,0]*jcb[1,1]-jcb[1,0]*jcb[0,1]
            
    jcbi[0,0]=    jcb[1,1] / jcob
    jcbi[0,1]=  - jcb[0,1] / jcob
    jcbi[1,0]=  - jcb[1,0] / jcob
    jcbi[1,1]=    jcb[0,0] / jcob
            
    for k in range(0,m):
        dNdx[k]=jcbi[0,0]*dNdr[k]+jcbi[0,1]*dNds[k]
        dNdy[k]=jcbi[1,0]*dNdr[k]+jcbi[1,1]*dNds[k]
                
    for i in range(0,m):
        i1=2*i
        i2=2*i+1
        Bmat[0,i1]=dNdx[i] ; Bmat[0,i2]=0.0
        Bmat[1,i1]=0.0     ; Bmat[1,i2]=dNdy[i]
        Bmat[2,i1]=dNdy[i] ; Bmat[2,i2]=dNdx[i]
                
    Ael=Ael+numpy.matmul(Bmat.transpose(),numpy.matmul(penalty*Kmat,Bmat))*wq*jcob
    
    #assembly of matrix A and righthand side B
    for k1 in range(0,m):
        ik=icon[k1,iel]
        for i1 in range(0,ndof):
            ikk=ndof*(k1)+i1
            m1=ndof*(ik)+i1
            for k2 in range(0,m):
                jk=icon[k2,iel]
                for i2 in range(0,ndof):
                    jkk=ndof*(k2)+i2
                    m2=ndof*(jk)+i2
                    A[m1,m2]=A[m1,m2]+Ael[ikk,jkk]
            B[m1]=B[m1]+Bel[ikk]
                            
#imposing boundary conditions
print("imposing boundary conditions")

for i in range(0,Nfem):
    if bc_fix[i]==True:
        Aref=A[i,i]
        for j in range(0,Nfem):
            B[j]=B[j]-A[i,j]*bc_val[i]
            A[i,j]=0.0
            A[j,i]=0.0
        A[i,i]=Aref
        B[i]=Aref*bc_val[i]

print("minimum A =",numpy.min(A))
print("maximum A =",numpy.max(A))
print("minimum B =",numpy.min(B))
print("maximum B =",numpy.max(B))
        
#solving system
print("solving system")
sol=numpy.linalg.solve(A,B)

#putting solution into seprate x,y velocity arrays
for i in range(0,np):
    u[i]=sol[i*ndof]
    v[i]=sol[i*ndof+1]
    
print("minimum u =",min(u), "maximum u",max(u))
print("minimum v =",min(v), "maximum v",max(v))

#outputting to file for use with GNUplot
file1=open('velocity_u.dat','w')
file2=open('velocity_v.dat','w')
for i in range(0,np):
    file1.write(str(x[i]) + ' ' + str(y[i]) + ' ' + str(u[i]) + '\n')
    file2.write(str(x[i]) + ' ' + str(y[i]) + ' ' + str(v[i]) + '\n')
file1.close()
file2.close()

print("END")
