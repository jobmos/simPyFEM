import numpy as np
import math as math
import sys as sys
import scipy
from scipy import sparse as sparse
from scipy.sparse.linalg import spsolve
import time as time
import matplotlib.pyplot as plt
import tkinter

# right hand side setup function 1
def b1(x, y):
    b1 = ((12.0 - 24.0 * y) * x ** 4 + (-24.0 + 48.0 * y) * x ** 3 +
          (-48.0 * y + 72.0 * y ** 2 - 48.0 * y ** 3 + 12.0) * x ** 2 +
          (-2.0 + 24.0 * y - 72.0 * y ** 2 + 48.0 * y ** 3) * x +
          1.0 - 4.0 * y + 12.0 * y ** 2 - 8.0 * y ** 3)
    return b1


# right hand side setup function 2
def b2(x, y):
    b2 = ((8.0 - 48.0 * y + 48.0 * y ** 2) * x ** 3 +
          (-12.0 + 72.0 * y - 72 * y ** 2) * x ** 2 +
          (4.0 - 24.0 * y + 48.0 * y ** 2 - 48.0 * y ** 3 + 24.0 * y ** 4) * x -
          12.0 * y ** 2 + 24.0 * y ** 3 - 12.0 * y ** 4)
    return b2


def main():
    # variable declaration
    print("variable declaration")

    m = 4  # nodes that make up an element
    ndof = 2  # degrees of freedom per node

    Lx = 1  # nondimensionalized size of system, x
    Ly = 1  # nondimensionalized size of system, y

    # allowing for argument parsing through command line
    if int(len(sys.argv) == 3):
        nnx = int(sys.argv[1])
        nny = int(sys.argv[2])
    else:
        nnx = 32
        nny = 32

    nnp = nnx * nny  # number of points

    nelx = nnx - 1  # number of elements, x
    nely = nny - 1  # number of elements, y

    nel = nelx * nely  # number of elements, total

    penalty = 1.0e7  # penalty

    viscosity = 1.0  # constant viscosity
    density = 1.0  # constant density

    Nfem = nnp * ndof  # Total number of degrees of freedom

    eps = 1.0e-10

    gx=0                # gravity, x
    gy=1                # gravity, y

    # declaring arrays
    print("declaring arrays")

    x = np.zeros((nnp), dtype=float)  # x coordinates
    y = np.copy(x)  # y coordinates
    u = np.copy(x)  # x velocities
    v = np.copy(x)  # y velocities
    A = np.zeros((Nfem, Nfem), dtype=float)  # matrix of Ax=b
    B = np.zeros((Nfem), dtype=float)  # righthand side of Ax=b
    sol = np.zeros((Nfem), dtype=float)  # solution vector of Ax=b
    icon = np.zeros((m, nel), dtype=int)  # connectivity
    bc_fix = np.zeros((Nfem), dtype=bool)  # boundary condition, yes/no
    bc_val = np.zeros((Nfem), dtype=float)  # boundary condition, value
    N = np.zeros((m), dtype=float)  # shape functions
    dNdx = np.copy(N)  # derivative of shape functions
    dNdy = np.copy(N)  # " "
    dNdr = np.copy(N)  # " "
    dNds = np.copy(N)  # " "
    jcb = np.zeros((2, 2), dtype=float)  # jacobian matrix
    jcbi = np.copy(jcb)  # inverse of jacobian matrix
    Bmat = np.zeros((3, ndof * m), dtype=float)  # numerical integration matrix
    Kmat = np.zeros((3, 3), dtype=float)  # " "
    Cmat = np.copy(Kmat)  # " "

    Kmat[0, 0] = 1.0
    Kmat[0, 1] = 1.0
    Kmat[0, 2] = 0.0
    Kmat[1, 0] = 1.0
    Kmat[1, 1] = 1.0
    Kmat[1, 2] = 0.0
    Kmat[2, 0] = 0.0
    Kmat[2, 1] = 0.0
    Kmat[2, 2] = 0.0

    Cmat[0, 0] = 2.0
    Cmat[0, 1] = 0.0
    Cmat[0, 2] = 0.0
    Cmat[1, 0] = 0.0
    Cmat[1, 1] = 2.0
    Cmat[1, 2] = 0.0
    Cmat[2, 0] = 0.0
    Cmat[2, 1] = 0.0
    Cmat[2, 2] = 1.0

    # grid point setup
    print("grid point setup")
    counter = 0
    for j in range(0, nely + 1):
        for i in range(0, nelx + 1):
            x[counter] = i * Lx / nelx
            y[counter] = j * Ly / nely
            counter += 1

    # connectivity
    print("connectivity")

    counter = 0
    for j in range(0, nely):
        for i in range(0, nelx):
            icon[0, counter] = i + j * (nelx + 1)
            icon[1, counter] = i + 1 + j * (nelx + 1)
            icon[2, counter] = i + 1 + (j + 1) * (nelx + 1)
            icon[3, counter] = i + (j + 1) * (nelx + 1)
            counter += 1

    # for iel in range (0,nel):
    #         print ("iel=",iel)
    #         print ("node 1",icon[0][iel],"at pos.",x[icon[0][iel-1]], y[icon[0][iel-1]])
    #         print ("node 2",icon[1][iel],"at pos.",x[icon[1][iel-1]], y[icon[1][iel-1]])
    #         print ("node 3",icon[2][iel],"at pos.",x[icon[2][iel-1]], y[icon[2][iel-1]])
    #         print ("node 4",icon[3][iel],"at pos.",x[icon[3][iel-1]], y[icon[3][iel-1]])

    # defining boundary conditions
    print("defining boundary conditions")

    for i in range(0, nnp):
        if x[i] < eps:
            bc_fix[i * ndof] = True
            bc_val[i * ndof] = 0.0
            bc_fix[i * ndof + 1] = True
            bc_val[i * ndof + 1] = 0.0
        if x[i] > (Lx - eps):
            bc_fix[i * ndof] = True
            bc_val[i * ndof] = 0.0
            bc_fix[i * ndof + 1] = True
            bc_val[i * ndof + 1] = 0.0
        if y[i] < eps:
            bc_fix[i * ndof] = True
            bc_val[i * ndof] = 0.0
            bc_fix[i * ndof + 1] = True
            bc_val[i * ndof + 1] = 0.0
        if y[i] > (Ly - eps):
            bc_fix[i * ndof] = True
            bc_val[i * ndof] = 0.0
            bc_fix[i * ndof + 1] = True
            bc_val[i * ndof + 1] = 0.0

    # building FE matrix
    print("building FE matrix")

    for iel in range(0, nel):

        # setting 2 arrays to 0 every loop
        Bel = [0 for i in range(0, m * ndof)]  # using list comprehension
        Ael = np.zeros((m * ndof, m * ndof), dtype=float)  # can we also use 2D list comprehension?

        # integrate viscous term at 4 quadrature points
        for iq in range(-1, 3, 2):
            for jq in range(-1, 3, 2):
                rq = iq / math.sqrt(3.0)
                sq = jq / math.sqrt(3.0)

                wq = 1.0 * 1.0

                # shape functions
                N[0] = 0.25 * (1.0 - rq) * (1.0 - sq)
                N[1] = 0.25 * (1.0 + rq) * (1.0 - sq)
                N[2] = 0.25 * (1.0 + rq) * (1.0 + sq)
                N[3] = 0.25 * (1.0 - rq) * (1.0 + sq)

                # shape function derivatives
                dNdr[0] = - 0.25 * (1.0 - sq)
                dNds[0] = - 0.25 * (1.0 - rq)
                dNdr[1] = + 0.25 * (1.0 - sq)
                dNds[1] = - 0.25 * (1.0 + rq)
                dNdr[2] = + 0.25 * (1.0 + sq)
                dNds[2] = + 0.25 * (1.0 + rq)
                dNdr[3] = - 0.25 * (1.0 + sq)
                dNds[3] = + 0.25 * (1.0 - rq)

                # jacobian matrix
                jcb = np.zeros((2, 2), dtype=float)
                for k in range(0, m):
                    jcb[0, 0] = jcb[0, 0] + dNdr[k] * x[icon[k, iel]]
                    jcb[0, 1] = jcb[0, 1] + dNdr[k] * y[icon[k, iel]]
                    jcb[1, 0] = jcb[1, 0] + dNds[k] * x[icon[k, iel]]
                    jcb[1, 1] = jcb[1, 1] + dNds[k] * y[icon[k, iel]]

                # determinant of the jacobian
                jcob = jcb[0, 0] * jcb[1, 1] - jcb[1, 0] * jcb[0, 1]

                # inverse of the jacobian matrix
                jcbi[0, 0] = jcb[1, 1] / jcob
                jcbi[0, 1] = - jcb[0, 1] / jcob
                jcbi[1, 0] = - jcb[1, 0] / jcob
                jcbi[1, 1] = jcb[0, 0] / jcob

                # computing dNdx & dNdy
                xq = 0.0
                yq = 0.0
                uq = 0.0
                vq = 0.0
                exxq = 0.0
                eyyq = 0.0
                exyq = 0.0
                for k in range(0, m):
                    xq = xq + N[k] * x[icon[k, iel]]
                    yq = yq + N[k] * y[icon[k, iel]]
                    uq = uq + N[k] * u[icon[k, iel]]
                    vq = vq + N[k] * v[icon[k, iel]]
                    dNdx[k] = jcbi[0, 0] * dNdr[k] + jcbi[0, 1] * dNds[k]
                    dNdy[k] = jcbi[1, 0] * dNdr[k] + jcbi[1, 1] * dNds[k]
                    exxq = exxq + dNdx[k] * u[icon[k, iel]]
                    eyyq = eyyq + dNdy[k] * v[icon[k, iel]]
                    exyq = exyq + dNdx[k] * v[icon[k, iel]] * 0.5 + dNdy[k] * u[icon[k, iel]] * 0.5

                # constructing 3x8 B matrix
                for i in range(0, m):
                    i1 = 2 * i
                    i2 = 2 * i + 1
                    Bmat[0, i1] = dNdx[i]
                    Bmat[0, i2] = 0
                    Bmat[1, i1] = 0
                    Bmat[1, i2] = dNdy[i]
                    Bmat[2, i1] = dNdy[i]
                    Bmat[2, i2] = dNdx[i]

                # computing elemental A matrix
                Ael = Ael + np.matmul(Bmat.transpose(), np.matmul(viscosity * Cmat, Bmat)) * wq * jcob

                # computing elemental B vector
                for i in range(0, m):
                    i1 = 2 * i
                    i2 = 2 * i + 1
                    # Bel[i1]=Bel[i1]+N[i]*jcob*wq*density*gx
                    # Bel[i2]=Bel[i2]+N[i]*jcob*wq*density*gy
                    Bel[i1] = Bel[i1] + N[i] * jcob * wq * b1(xq, yq)
                    Bel[i2] = Bel[i2] + N[i] * jcob * wq * b2(xq, yq)

        # integrate penalty term at 1 point
        rq = 0.0
        sq = 0.0
        wq = 2.0 * 2.0

        N[0] = 0.25 * (1.0 - rq) * (1.0 - sq)
        N[1] = 0.25 * (1.0 + rq) * (1.0 - sq)
        N[2] = 0.25 * (1.0 + rq) * (1.0 + sq)
        N[3] = 0.25 * (1.0 - rq) * (1.0 + sq)

        dNdr[0] = - 0.25 * (1.0 - sq)
        dNds[0] = - 0.25 * (1.0 - rq)
        dNdr[1] = + 0.25 * (1.0 - sq)
        dNds[1] = - 0.25 * (1.0 + rq)
        dNdr[2] = + 0.25 * (1.0 + sq)
        dNds[2] = + 0.25 * (1.0 + rq)
        dNdr[3] = - 0.25 * (1.0 + sq)
        dNds[3] = + 0.25 * (1.0 - rq)

        for k in range(0, m):
            jcb[0, 0] = jcb[0, 0] + dNdr[k] * x[icon[k, iel]]
            jcb[0, 1] = jcb[0, 1] + dNdr[k] * y[icon[k, iel]]
            jcb[1, 0] = jcb[1, 0] + dNds[k] * x[icon[k, iel]]
            jcb[1, 1] = jcb[1, 1] + dNds[k] * y[icon[k, iel]]

        jcob = jcb[0, 0] * jcb[1, 1] - jcb[1, 0] * jcb[0, 1]

        jcbi[0, 0] = jcb[1, 1] / jcob
        jcbi[0, 1] = - jcb[0, 1] / jcob
        jcbi[1, 0] = - jcb[1, 0] / jcob
        jcbi[1, 1] = jcb[0, 0] / jcob

        for k in range(0, m):
            dNdx[k] = jcbi[0, 0] * dNdr[k] + jcbi[0, 1] * dNds[k]
            dNdy[k] = jcbi[1, 0] * dNdr[k] + jcbi[1, 1] * dNds[k]

        for i in range(0, m):
            i1 = 2 * i
            i2 = 2 * i + 1
            Bmat[0, i1] = dNdx[i]
            Bmat[0, i2] = 0.0
            Bmat[1, i1] = 0.0
            Bmat[1, i2] = dNdy[i]
            Bmat[2, i1] = dNdy[i]
            Bmat[2, i2] = dNdx[i]

        Ael = Ael + np.matmul(Bmat.transpose(), np.matmul(penalty * Kmat, Bmat)) * wq * jcob

        # assembly of matrix A and righthand side B
        for k1 in range(0, m):
            ik = icon[k1, iel]
            for i1 in range(0, ndof):
                ikk = ndof * (k1) + i1
                m1 = ndof * (ik) + i1
                for k2 in range(0, m):
                    jk = icon[k2, iel]
                    for i2 in range(0, ndof):
                        jkk = ndof * (k2) + i2
                        m2 = ndof * (jk) + i2
                        A[m1, m2] = A[m1, m2] + Ael[ikk, jkk]
                B[m1] = B[m1] + Bel[ikk]

    # imposing boundary conditions
    print("imposing boundary conditions")

    for i in range(0, Nfem):
        if bc_fix[i] == True:
            Aref = A[i, i]
            for j in range(0, Nfem):
                B[j] = B[j] - A[i, j] * bc_val[i]
                A[i, j] = 0.0
                A[j, i] = 0.0
            A[i, i] = Aref
            B[i] = Aref * bc_val[i]

    print("minimum A =", np.min(A))
    print("maximum A =", np.max(A))
    print("minimum B =", np.min(B))
    print("maximum B =", np.max(B))

    A_sparse = sparse.csr_matrix(A)

    # solving system
    print("solving system")
    start = time.time()

    # sol=np.linalg.solve(A,B)
    # sol=spsolve(A_sparse,B)
    sol, info = scipy.sparse.linalg.cg(A, B, tol=10e-8)

    # putting solution into seprate x,y velocity arrays
    for i in range(0, nnp):
        u[i] = sol[i * ndof]
        v[i] = sol[i * ndof + 1]

    print("minimum u =", min(u), "maximum u", max(u))
    print("minimum v =", min(v), "maximum v", max(v))

    print("time elapsed:", time.time() - start)

    ##outputting to file for use with GNUplot
    # file1=open('velocity_u.dat','w')
    # file2=open('velocity_v.dat','w')
    # for i in range(0,nnp):
    #    file1.write(str(x[i]) + ' ' + str(y[i]) + ' ' + str(u[i]) + '\n')
    #    file2.write(str(x[i]) + ' ' + str(y[i]) + ' ' + str(v[i]) + '\n')
    # file1.close()
    # file2.close()

    print("done, close figures to exit out of program.")

    # plotting of solution
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 10))

    utemp = np.reshape(u, (nnx, nny))
    vtemp = np.reshape(v, (nnx, nny))

    im = axes[0].imshow(utemp, extent=(np.amin(x), np.amax(x), np.amin(y), np.amax(y)), cmap='hot',
                        interpolation='nearest')
    axes[0].set_title('Velocity, x-direction', fontsize=20, y=1.08)
    im = axes[1].imshow(vtemp, extent=(np.amin(x), np.amax(x), np.amin(y), np.amax(y)), cmap='hot',
                        interpolation='nearest')
    axes[1].set_title('Velocity, y-direction', fontsize=20, y=1.08)
    fig.subplots_adjust(right=0.80)
    cbar_ax = fig.add_axes([0.85, 0.32, 0.05, 0.39])
    fig.colorbar(im, cax=cbar_ax)
    plt.show()

if __name__ == "__main__":
    main()
