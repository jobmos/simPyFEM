#right hand side setup function 1
def b1(x,y):
    b1 =((12.0-24.0*y)*x**4 + (-24.0+48.0*y)*x**3 + 
        (-48.0*y+72.0*y**2-48.0*y**3+12.0)*x**2 +
        (-2.0+24.0*y-72.0*y**2+48.0*y**3)*x + 
        1.0-4.0*y+12.0*y**2-8.0*y**3 )
    return b1

#right hand side setup function 2
def b2(x,y):
    b2 =((8.0-48.0*y+48.0*y**2)*x**3 + 
        (-12.0+72.0*y-72*y**2)*x**2 +
        (4.0-24.0*y+48.0*y**2-48.0*y**3+24.0*y**4)*x - 
        12.0*y**2 + 24.0*y**3 -12.0*y**4)
    return b2

#uzawa (outer) solve, method #3
def uzawa3(KKK,G,Nfem,nel,rhs_f,rhs_h):
    Vsol    = numpy.zeros((Nfem),dtype=float)
    Psol    = numpy.zeros((nel),dtype=float)
    Vsolmem = numpy.zeros((Nfem),dtype=float)
    Psolmem = numpy.zeros((nel),dtype=float)
    KKKmem  = numpy.zeros((Nfem,Nfem),dtype=float)
    phi     = numpy.zeros((Nfem),dtype=float)
    h       = numpy.copy(phi)
    q       = numpy.zeros((nel),dtype=float)
    d       = numpy.copy(q)
    qkp1    = numpy.zeros((nel),dtype=float)
    dkp1    = numpy.copy(q)

    tol=1e-6
    niter=250
    KKKmem=KKK

    V_diff = []
    P_diff = []

    #compute u1
    B=rhs_f-numpy.matmul(G,Psol)
    B=spsolve(KKK,B)
    Vsol=B
    
    q=rhs_h-numpy.matmul(G.transpose(),Vsol)
    d=-q
     
    file1=open('VP_diff_uzawa3.dat','w')
    
    for iter in range(0,niter):
        #compute pk
        phi=numpy.matmul(G,d)
         
        #compute hk=A^-1 pk
        B=phi 
        B=spsolve(KKK,B)
        h=B 
 
        #compute alpha
        alpha=numpy.dot(q,q)/numpy.dot(phi,h)
        
        #update pressure
        Psol=Psol+alpha*d
        
        #update velocity
        Vsol=Vsol-alpha*h
        
        #compute qk+1
        qkp1=rhs_h-numpy.matmul(G.transpose(),Vsol)
        
        #compute beta
        beta=numpy.dot(qkp1,qkp1)/numpy.dot(q,q)
        
        #compute dkp1
        dkp1=-qkp1+beta*d
        
        #check for convergence
        V_diff.append(numpy.max(abs(Vsol-Vsolmem))/numpy.max(abs(Vsol)))
        P_diff.append(numpy.max(abs(Psol-Psolmem))/numpy.max(abs(Psol)))
        
        print('\r','iteration:',iter,'V_diff:',V_diff[iter],'P_diff',P_diff[iter],'max(Psol)',numpy.max(Psol),end='')
        
        Psolmem=Psol
        Vsolmem=Vsol
        d=dkp1
        q=qkp1
        
        file1.write(str(iter+1) + ' ' + str(V_diff[iter]) + ' ' + str(P_diff[iter]) + '\n')
         
        if max(V_diff[iter],P_diff[iter]) < tol or iter == niter-1 :
            print('')
            file1.close()
            return (Vsol,Psol,V_diff,P_diff)

#importing packages etc.
print("importing pacakages")

import numpy as numpy
import math as math
import sys as sys
import scipy as scipy
from scipy.sparse import csr_matrix
from scipy.sparse import linalg
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt
import time
#import matplotlib.cm as cm
#from matplotlib.colors import LogNorm

start_time=time.time()

#variable declaration
print("variable declaration")

m=4                 # nodes that make up an element
ndof=2              # degrees of freedom per node

Lx=1                # nondimensionalized size of system
Ly=1                #

nnx=51              # resolution, x
nny=51              # resolution, y

np=nnx*nny          # number of points

nelx=nnx-1          # number of elements, x
nely=nny-1          # number of elements, y

nel=nelx*nely       # number of elements, total

viscosity=1.0       # constant viscosity
density=1.0         # constant density  

Nfem=np*ndof        # Total number of degrees of freedom

eps=1.0e-10

gx=0                # gravity, x
gy=1                # gravity, y

#declaring arrays
print("declaring arrays")

x      = numpy.zeros((np),dtype=float)           # x coordinates
y      = numpy.copy(x)                           # y coordinates
u      = numpy.copy(x)                           # x velocities
v      = numpy.copy(x)                           # y velocities
A      = numpy.zeros((Nfem,Nfem),dtype=float)    # matrix of Ax=b
B      = numpy.zeros((Nfem),dtype=float)         # righthand side of Ax=b
sol    = numpy.zeros((Nfem),dtype=float)         # solution vector of Ax=b
icon   = numpy.zeros((m,nel),dtype=int)          # connectivity
bc_fix = numpy.zeros((Nfem),dtype=bool)          # boundary condition, yes/no
bc_val = numpy.zeros((Nfem),dtype=float)         # boundary condition, value
N      = numpy.zeros((m),dtype=float)            # shape functions
dNdx   = numpy.copy(N)                           # derivative of shape functions
dNdy   = numpy.copy(N)                           # " "
dNdr   = numpy.copy(N)                           # " "
dNds   = numpy.copy(N)                           # " "
jcb    = numpy.zeros((2,2),dtype=float)          # jacobian matrix
jcbi   = numpy.copy(jcb)                         # inverse of jacobian matrix
Bmat   = numpy.zeros((3,ndof*m),dtype=float)     # numerical integration matrix
Kmat   = numpy.zeros((3,3),dtype=float)          # " "
Cmat   = numpy.copy(Kmat)                        # " "
KKK    = numpy.zeros((Nfem,Nfem),   dtype=float) # 1,1 of block stokes matrix
G      = numpy.zeros((Nfem,nel),    dtype=float) # 1,2 of block stokes matrix
rhs_f  = numpy.zeros((Nfem),        dtype=float) # 1st block in block stokes rhs vector
rhs_h  = numpy.zeros((nel),         dtype=float) # 2nd block in block stokes rhs vector

Cmat[0,0]=2.0
Cmat[1,1]=2.0
Cmat[2,2]=1.0  

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
        
#define boundary conditions
print("define boundary conditions")

for i in range (0,np):
    if x[i] < eps:
        bc_fix[i*ndof]  = True   ; bc_val[i*ndof]  =0.0
        bc_fix[i*ndof+1]= True   ; bc_val[i*ndof+1]=0.0
    if x[i] > (Lx-eps):
        bc_fix[i*ndof]  = True   ; bc_val[i*ndof]  =0.0
        bc_fix[i*ndof+1]= True   ; bc_val[i*ndof+1]=0.0
    if y[i] < eps:
        bc_fix[i*ndof]  = True   ; bc_val[i*ndof]  =0.0
        bc_fix[i*ndof+1]= True   ; bc_val[i*ndof+1]=0.0
    if y[i] > (Ly-eps):
        bc_fix[i*ndof]  = True   ; bc_val[i*ndof]  =0.0
        bc_fix[i*ndof+1]= True   ; bc_val[i*ndof+1]=0.0

#build FE matrix
print("build FE matrix")

for iel in range (0,nel):
    #setting 3 arrays to 0 every loop    
    Kel = numpy.zeros((m*ndof,m*ndof),dtype=float)
    Gel = numpy.zeros((m*ndof,1),dtype=float)
    #list comprehension
    fel = [0 for i in range(0,m*ndof)]
    
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
            dNdr[0]= - 0.25*(1.0-sq)   ;   dNds[0]= - 0.25*(1.0-rq)
            dNdr[1]= + 0.25*(1.0-sq)   ;   dNds[1]= - 0.25*(1.0+rq)
            dNdr[2]= + 0.25*(1.0+sq)   ;   dNds[2]= + 0.25*(1.0+rq)
            dNdr[3]= - 0.25*(1.0+sq)   ;   dNds[3]= + 0.25*(1.0-rq)
            
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
            jcbi[0,0]=    jcb[1,1] /jcob
            jcbi[0,1]=  - jcb[0,1] /jcob
            jcbi[1,0]=  - jcb[1,0] /jcob
            jcbi[1,1]=    jcb[0,0] /jcob
            
            #computing dNdx & dNdy
            xq=0.0e0
            yq=0.0e0
            uq=0.0e0
            vq=0.0e0
            exxq=0.0e0
            eyyq=0.0e0
            exyq=0.0e0
            for k in range(0,m):
                xq=xq+N[k]*x[icon[k,iel]]
                yq=yq+N[k]*y[icon[k,iel]]
                uq=uq+N[k]*u[icon[k,iel]]
                vq=vq+N[k]*v[icon[k,iel]]
                dNdx[k]=jcbi[0,0]*dNdr[k]+jcbi[0,1]*dNds[k]
                dNdy[k]=jcbi[1,0]*dNdr[k]+jcbi[1,1]*dNds[k]
                exxq=exxq+ dNdx[k]*u[icon[k,iel]]
                eyyq=eyyq+ dNdy[k]*v[icon[k,iel]]
                exyq=exyq+ dNdx[k]*v[icon[k,iel]] *0.5e0 + dNdy[k]*u[icon[k,iel]] *0.5e0
            
            #constructing 3x8 B matrix
            for i in range(0,m):
                i1=2*i
                i2=2*i+1
                Bmat[0,i1]=dNdx[i] ; Bmat[0,i2]=0.0
                Bmat[1,i1]=0.0       ; Bmat[1,i2]=dNdy[i]
                Bmat[2,i1]=dNdy[i] ; Bmat[2,i2]=dNdx[i]
            
            #computing elemental K matrix
            Kel=Kel+numpy.matmul(Bmat.transpose(),numpy.matmul(viscosity*Cmat,Bmat))*wq*jcob
            
            #computing elemental B matrix and rhs f
            for i in range(0,m):
                i1=2*i
                i2=2*i+1
                #fel[i1]=fel[i1]+N[i]*jcob*wq*density*gx
                #fel[i2]=fel[i2]+N[i]*jcob*wq*density*gy
                fel[i1]=fel[i1]+N[i]*jcob*wq*b1(xq,yq)
                fel[i2]=fel[i2]+N[i]*jcob*wq*b2(xq,yq)
                Gel[i1,0]=Gel[i1,0]-dNdx[i]*jcob*wq
                Gel[i2,0]=Gel[i2,0]-dNdy[i]*jcob*wq           
    
    #assemble
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
                    KKK[m1,m2]=KKK[m1,m2]+Kel[ikk,jkk]
            rhs_f[m1]=rhs_f[m1]+fel[ikk]
            G[m1,iel]=G[m1,iel]+Gel[ikk,0]
                                       
#impose b.c.
print("impose b.c.")

for i in range(0,Nfem):
    if bc_fix[i]==True:
        Aref=KKK[i,i]
        for j in range(0,Nfem):
            rhs_f[j]=rhs_f[j]-KKK[i,j]*bc_val[i]
            KKK[i,j]=0.0
            KKK[j,i]=0.0
        KKK[i,i]=Aref
        rhs_f[i]=Aref*bc_val[i]
        
        for j in range(0,nel):
            rhs_h[j]=rhs_h[j]-G[i,j]*bc_val[i]
            G[i,j]=0.0

#sparse strorage
print('sparse storage')
KKK_sparse = csr_matrix(KKK)
print('number of nonzero elements in K: ',scipy.sparse.csr_matrix.count_nonzero(KKK_sparse))

#solve system
print('solve system')

(Vsol,Psol,V_diff,P_diff)=uzawa3(KKK_sparse,G,Nfem,nel,rhs_f,rhs_h)  

for i in range(0,np):
    u[i]=Vsol[i*ndof]
    v[i]=Vsol[i*ndof+1]

print("minimum u =",min(u), "maximum u",max(u))
print("minimum v =",min(v), "maximum v",max(v))

print("--- %s seconds ---" % (time.time() - start_time))

print("done, close figures to exit out of program.")

#plotting of solution
fig, axes = plt.subplots(nrows=1, ncols=2,figsize=(10,10))

utemp=numpy.reshape(u,(nnx,nny))
vtemp=numpy.reshape(v,(nnx,nny))

im = axes[0].imshow(utemp, extent=(numpy.amin(x),numpy.amax(x),numpy.amin(y),numpy.amax(y)),cmap='hot', interpolation='nearest')
axes[0].set_title('Velocity, x-direction',fontsize=20,y=1.08)
im = axes[1].imshow(vtemp, extent=(numpy.amin(x),numpy.amax(x),numpy.amin(y),numpy.amax(y)),cmap='hot', interpolation='nearest')
axes[1].set_title('Velocity, y-direction',fontsize=20,y=1.08)
fig.subplots_adjust(right=0.80)
cbar_ax = fig.add_axes([0.85, 0.32, 0.05, 0.39])
fig.colorbar(im, cax=cbar_ax)
plt.show()

#plotting of outer solve convergence
plt.figure(figsize=(5,5))
plt.plot(V_diff,label='V_diff')
plt.plot(P_diff,label='P_diff')
plt.yscale('log')
plt.ylabel('error')
plt.xlabel('#outer iteration')
plt.title('Outer solve convergence',fontsize=20,y=1.08)
plt.grid()
plt.legend()
plt.show()
