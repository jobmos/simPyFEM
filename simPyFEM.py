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
    val = ((12.0 - 24.0 * y) * x ** 4 + (-24.0 + 48.0 * y) * x ** 3 +
           (-48.0 * y + 72.0 * y ** 2 - 48.0 * y ** 3 + 12.0) * x ** 2 +
           (-2.0 + 24.0 * y - 72.0 * y ** 2 + 48.0 * y ** 3) * x +
           1.0 - 4.0 * y + 12.0 * y ** 2 - 8.0 * y ** 3)
    return val


# right hand side setup function 2
def b2(x, y):
    val = ((8.0 - 48.0 * y + 48.0 * y ** 2) * x ** 3 +
           (-12.0 + 72.0 * y - 72 * y ** 2) * x ** 2 +
           (4.0 - 24.0 * y + 48.0 * y ** 2 - 48.0 * y ** 3 + 24.0 * y ** 4) * x -
           12.0 * y ** 2 + 24.0 * y ** 3 - 12.0 * y ** 4)
    return val


def main():
    # declare variables
    print("variable declaration")

    m = 4  # nodes that make up an element
    ndof = 2  # degrees of freedom per node

    lx = 1  # nondimensional size of system, x
    ly = 1  # nondimensional size of system, y

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

    nfem = nnp * ndof  # Total number of degrees of freedom

    eps = 1.0e-10

    gx = 0  # gravity, x
    gy = 1  # gravity, y

    # declare arrays
    print("declaring arrays")

    x = np.zeros(nnp, dtype=float)  # x coordinates
    y = np.copy(x)  # y coordinates
    u = np.copy(x)  # x velocities
    v = np.copy(x)  # y velocities
    a_mat = np.zeros((nfem, nfem), dtype=float)  # matrix of Ax=b
    rhs = np.zeros(nfem, dtype=float)  # right hand side of Ax=b
    icon = np.zeros((m, nel), dtype=int)  # connectivity
    bc_fix = np.zeros(nfem, dtype=bool)  # boundary condition, yes/no
    bc_val = np.zeros(nfem, dtype=float)  # boundary condition, value
    n = np.zeros(m, dtype=float)  # shape functions
    dndx = np.copy(n)  # derivative of shape functions
    dndy = np.copy(n)  # " "
    dndr = np.copy(n)  # " "
    dnds = np.copy(n)  # " "
    jcb = np.zeros((2, 2), dtype=float)  # jacobian matrix
    b_mat = np.zeros((3, ndof * m), dtype=float)  # numerical integration matrix
    k_mat = np.zeros((3, 3), dtype=float)  # " "
    c_mat = np.copy(k_mat)  # " "

    k_mat[0, 0] = 1.0
    k_mat[0, 1] = 1.0
    k_mat[0, 2] = 0.0
    k_mat[1, 0] = 1.0
    k_mat[1, 1] = 1.0
    k_mat[1, 2] = 0.0
    k_mat[2, 0] = 0.0
    k_mat[2, 1] = 0.0
    k_mat[2, 2] = 0.0

    c_mat[0, 0] = 2.0
    c_mat[0, 1] = 0.0
    c_mat[0, 2] = 0.0
    c_mat[1, 0] = 0.0
    c_mat[1, 1] = 2.0
    c_mat[1, 2] = 0.0
    c_mat[2, 0] = 0.0
    c_mat[2, 1] = 0.0
    c_mat[2, 2] = 1.0

    # grid point setup
    print("grid point setup")
    counter = 0
    for j in range(0, nely + 1):
        for i in range(0, nelx + 1):
            x[counter] = i * lx / nelx
            y[counter] = j * ly / nely
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

    # define boundary conditions
    print("defining boundary conditions")

    for i in range(0, nnp):
        if x[i] < eps:
            bc_fix[i * ndof] = True
            bc_val[i * ndof] = False
            bc_fix[i * ndof + 1] = True
            bc_val[i * ndof + 1] = False
        if x[i] > (lx - eps):
            bc_fix[i * ndof] = True
            bc_val[i * ndof] = False
            bc_fix[i * ndof + 1] = True
            bc_val[i * ndof + 1] = False
        if y[i] < eps:
            bc_fix[i * ndof] = True
            bc_val[i * ndof] = False
            bc_fix[i * ndof + 1] = True
            bc_val[i * ndof + 1] = False
        if y[i] > (ly - eps):
            bc_fix[i * ndof] = True
            bc_val[i * ndof] = False
            bc_fix[i * ndof + 1] = True
            bc_val[i * ndof + 1] = False

    # build FE matrix
    print("building FE matrix")

    for iel in range(0, nel):

        # set 2 arrays to 0 every loop
        b_el = np.zeros(m * ndof)
        a_el = np.zeros((m * ndof, m * ndof), dtype=float)  # can we also use 2D list comprehension?

        # integrate viscous term at 4 quadrature points
        for iq in range(-1, 3, 2):
            for jq in range(-1, 3, 2):
                rq = iq / math.sqrt(3.0)
                sq = jq / math.sqrt(3.0)

                wq = 1.0 * 1.0

                # calculate shape functions
                n[0] = 0.25 * (1.0 - rq) * (1.0 - sq)
                n[1] = 0.25 * (1.0 + rq) * (1.0 - sq)
                n[2] = 0.25 * (1.0 + rq) * (1.0 + sq)
                n[3] = 0.25 * (1.0 - rq) * (1.0 + sq)

                # calculate shape function derivatives
                dndr[0] = - 0.25 * (1.0 - sq)
                dnds[0] = - 0.25 * (1.0 - rq)
                dndr[1] = + 0.25 * (1.0 - sq)
                dnds[1] = - 0.25 * (1.0 + rq)
                dndr[2] = + 0.25 * (1.0 + sq)
                dnds[2] = + 0.25 * (1.0 + rq)
                dndr[3] = - 0.25 * (1.0 + sq)
                dnds[3] = + 0.25 * (1.0 - rq)

                # calculate jacobian matrix
                jcb = np.zeros((2, 2), dtype=float)
                for k in range(0, m):
                    jcb[0, 0] = jcb[0, 0] + dndr[k] * x[icon[k, iel]]
                    jcb[0, 1] = jcb[0, 1] + dndr[k] * y[icon[k, iel]]
                    jcb[1, 0] = jcb[1, 0] + dnds[k] * x[icon[k, iel]]
                    jcb[1, 1] = jcb[1, 1] + dnds[k] * y[icon[k, iel]]

                # calculate determinant of the jacobian
                # jcob = jcb[0, 0] * jcb[1, 1] - jcb[1, 0] * jcb[0, 1]
                jcob = np.linalg.det(jcb)

                # calculate inverse of the jacobian matrix
                # jcbi[0, 0] = jcb[1, 1] / jcob
                # jcbi[0, 1] = - jcb[0, 1] / jcob
                # jcbi[1, 0] = - jcb[1, 0] / jcob
                # jcbi[1, 1] = jcb[0, 0] / jcob
                jcbi = np.linalg.inv(jcb)

                # compute dndx & dndy
                xq = 0.0
                yq = 0.0
                uq = 0.0
                vq = 0.0
                exxq = 0.0
                eyyq = 0.0
                exyq = 0.0
                for k in range(0, m):
                    xq = xq + n[k] * x[icon[k, iel]]
                    yq = yq + n[k] * y[icon[k, iel]]
                    uq = uq + n[k] * u[icon[k, iel]]
                    vq = vq + n[k] * v[icon[k, iel]]
                    dndx[k] = jcbi[0, 0] * dndr[k] + jcbi[0, 1] * dnds[k]
                    dndy[k] = jcbi[1, 0] * dndr[k] + jcbi[1, 1] * dnds[k]
                    exxq = exxq + dndx[k] * u[icon[k, iel]]
                    eyyq = eyyq + dndy[k] * v[icon[k, iel]]
                    exyq = exyq + dndx[k] * v[icon[k, iel]] * 0.5 + dndy[k] * u[icon[k, iel]] * 0.5

                # construct 3x8 b_mat matrix
                for i in range(0, m):
                    i1 = 2 * i
                    i2 = 2 * i + 1
                    b_mat[0, i1] = dndx[i]
                    b_mat[0, i2] = 0
                    b_mat[1, i1] = 0
                    b_mat[1, i2] = dndy[i]
                    b_mat[2, i1] = dndy[i]
                    b_mat[2, i2] = dndx[i]

                # compute elemental a_mat matrix
                a_el = a_el + np.matmul(b_mat.transpose(), np.matmul(viscosity * c_mat, b_mat)) * wq * jcob

                # compute elemental rhs vector
                for i in range(0, m):
                    i1 = 2 * i
                    i2 = 2 * i + 1
                    # b_el[i1]=b_el[i1]+n[i]*jcob*wq*density*gx
                    # b_el[i2]=b_el[i2]+n[i]*jcob*wq*density*gy
                    b_el[i1] = b_el[i1] + n[i] * jcob * wq * b1(xq, yq)
                    b_el[i2] = b_el[i2] + n[i] * jcob * wq * b2(xq, yq)

        # integrate penalty term at 1 point
        rq = 0.0
        sq = 0.0
        wq = 2.0 * 2.0

        n[0] = 0.25 * (1.0 - rq) * (1.0 - sq)
        n[1] = 0.25 * (1.0 + rq) * (1.0 - sq)
        n[2] = 0.25 * (1.0 + rq) * (1.0 + sq)
        n[3] = 0.25 * (1.0 - rq) * (1.0 + sq)

        dndr[0] = - 0.25 * (1.0 - sq)
        dnds[0] = - 0.25 * (1.0 - rq)
        dndr[1] = + 0.25 * (1.0 - sq)
        dnds[1] = - 0.25 * (1.0 + rq)
        dndr[2] = + 0.25 * (1.0 + sq)
        dnds[2] = + 0.25 * (1.0 + rq)
        dndr[3] = - 0.25 * (1.0 + sq)
        dnds[3] = + 0.25 * (1.0 - rq)

        for k in range(0, m):
            jcb[0, 0] = jcb[0, 0] + dndr[k] * x[icon[k, iel]]
            jcb[0, 1] = jcb[0, 1] + dndr[k] * y[icon[k, iel]]
            jcb[1, 0] = jcb[1, 0] + dnds[k] * x[icon[k, iel]]
            jcb[1, 1] = jcb[1, 1] + dnds[k] * y[icon[k, iel]]

        # calculate determinant of the jacobian
        jcob = np.linalg.det(jcb)

        # calculate the inverse of the jacobian
        jcbi = np.linalg.inv(jcb)

        for k in range(0, m):
            dndx[k] = jcbi[0, 0] * dndr[k] + jcbi[0, 1] * dnds[k]
            dndy[k] = jcbi[1, 0] * dndr[k] + jcbi[1, 1] * dnds[k]

        for i in range(0, m):
            i1 = 2 * i
            i2 = 2 * i + 1
            b_mat[0, i1] = dndx[i]
            b_mat[0, i2] = 0.0
            b_mat[1, i1] = 0.0
            b_mat[1, i2] = dndy[i]
            b_mat[2, i1] = dndy[i]
            b_mat[2, i2] = dndx[i]

        a_el += np.matmul(b_mat.transpose(), np.matmul(penalty * k_mat, b_mat)) * wq * jcob

        # assembe matrix a_mat and right hand side rhs
        for k1 in range(0, m):
            ik = icon[k1, iel]
            for i1 in range(0, ndof):
                ikk = ndof * k1 + i1
                m1 = ndof * ik + i1
                for k2 in range(0, m):
                    jk = icon[k2, iel]
                    for i2 in range(0, ndof):
                        jkk = ndof * k2 + i2
                        m2 = ndof * jk + i2
                        a_mat[m1, m2] = a_mat[m1, m2] + a_el[ikk, jkk]
                rhs[m1] = rhs[m1] + b_el[ikk]

    # impose boundary conditions
    print("imposing boundary conditions")

    for i in range(0, nfem):
        if bc_fix[i]:
            a_matref = a_mat[i, i]
            for j in range(0, nfem):
                rhs[j] = rhs[j] - a_mat[i, j] * bc_val[i]
                a_mat[i, j] = 0.0
                a_mat[j, i] = 0.0
            a_mat[i, i] = a_matref
            rhs[i] = a_matref * bc_val[i]

    print("minimum a_mat =", np.min(a_mat))
    print("maximum a_mat =", np.max(a_mat))
    print("minimum rhs =", np.min(rhs))
    print("maximum rhs =", np.max(rhs))

    # a_mat_sparse = sparse.csr_matrix(a_mat)

    # solve system
    print("solving system")
    start = time.time()

    # sol=np.linalg.solve(a_mat,rhs)
    # sol=spsolve(a_mat_sparse,rhs)
    sol, info = scipy.sparse.linalg.cg(a_mat, rhs, tol=10e-8)

    # put solution into seprate x,y velocity arrays
    for i in range(0, nnp):
        u[i] = sol[i * ndof]
        v[i] = sol[i * ndof + 1]

    print("minimum u =", min(u), "maximum u", max(u))
    print("minimum v =", min(v), "maximum v", max(v))

    print("time elapsed:", time.time() - start)

    # output to file for use with gnuplot
    # file1=open('velocity_u.dat','w')
    # file2=open('velocity_v.dat','w')
    # for i in range(0,nnp):
    #    file1.write(str(x[i]) + ' ' + str(y[i]) + ' ' + str(u[i]) + '\n')
    #    file2.write(str(x[i]) + ' ' + str(y[i]) + ' ' + str(v[i]) + '\n')
    # file1.close()
    # file2.close()

    print("done, close figures to exit out of program.")

    # plot of solution
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
