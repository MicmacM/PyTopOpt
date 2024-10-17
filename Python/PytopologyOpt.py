import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve
import time
import Graphe


def lk():
	E=1
	nu=0.3
	k=np.array([1/2-nu/6,1/8+nu/8,-1/4-nu/12,-1/8+3*nu/8,-1/4+nu/12,-1/8-nu/8,nu/6,1/8-3*nu/8])
	KE = E/(1-nu**2)*np.array([[k[0], k[1], k[2], k[3], k[4], k[5], k[6], k[7]],
	[k[1], k[0], k[7], k[6], k[5], k[4], k[3], k[2]],
	[k[2], k[7], k[0], k[5], k[6], k[3], k[4], k[1]],
	[k[3], k[6], k[5], k[0], k[7], k[2], k[1], k[4]],
	[k[4], k[5], k[6], k[7], k[0], k[1], k[2], k[3]],
	[k[5], k[4], k[3], k[2], k[1], k[0], k[7], k[6]],
	[k[6], k[3], k[4], k[1], k[2], k[7], k[0], k[5]],
	[k[7], k[2], k[1], k[4], k[3], k[6], k[5], k[0]]])
	return (KE)


def optimise(nx : int, ny : int, volfrac: float, p : float, rmin : float):
    """
    Main loop of the program, it starts by computing as much element as possible to avoid
    redondant code, and then apply the optimization loop.

    Arguments :
        nx : number of element along the x axis
        ny : number of element along the y axis
        volfrac : percentage (in the range [0,1]) of matter to keep in the final design
        p : penalty factor
        rmin : radius of the filter

    Returns :
        None. It prints the density of material (x_phys)
    """
    e0, emin, nu = initialize_material_properties()
    ke = prepare_element_stiffness_matrix(e0, nu)
    edof_mat, i_k, j_k = prepare_fe_analysis(nx, ny)
    f, fixed_dofs, free_dofs = define_loads_and_supports(nx, ny)
    x = initialize_design_variables(nx, ny, volfrac)
    x_phys = x.copy()

    h, hs = prepare_filter(nx, ny, rmin)
    iter = 0
    change = 1
    u = np.zeros((2 * (nx + 1) * (ny + 1),1))


    while change > 1e-2:
        iter += 1

        u = fe_analysis(u, nx, ny, x_phys, i_k, j_k, f, free_dofs, e0, emin, p, ke)
        c, dc, dv = compute_objective_and_sensibility(x_phys, u, edof_mat, ke, e0, emin, p, nx, ny)
        h, hs = prepare_filter(nx, ny, rmin)
        dc = apply_filter(x, h, hs, dc)
        x = update_design_variables(x, dc, dv, volfrac, nx, ny)
        change = compute_change(x, x_phys)
        x_phys = x

        print("Iter :  {}, c : {}, volfrac : {}, change : {} ".format(iter, c, x, change))

    # Plot to screen    
    plt.matshow(-x_phys.reshape((nx,ny)).T)
    plt.show()


def initialize_material_properties():
    """
    Definition of the young modulus, the minimal young modulus to avoid getting 0 terms that 
    mess it up, and of the poisson ratio nu.
    The returned result is a tuple (E, Emin, nu)
    """
    return (1, 1e-9, 0.3)

def prepare_element_stiffness_matrix(e_young : float, nu : float):
    """
    Prepare the element stiffness matrix for a 2D quadrilateral element

    Arguments : 
        e_young : young modulus of the material
        nu : poisson ration of the material

    Returns : 
        Ke : 2D quadrilateral element stiffness matrix
    """
    return lk()
    return Graphe.lk(100, e_young, nu)

def prepare_fe_analysis(nx : int, ny : int):
    """
    Prepare the list i_k and j_k used in constructing the global stiffness matrix k, as well
    as the matrix containing the dofs of each element (8 dofs / element)
    """

    edof_mat = np.zeros((nx * ny, 8), dtype=int)

    for elx in range(nx):
        for ely in range(ny):
            el = ely + (elx * ny)
            n1 = (ny + 1) * elx + ely
            n2 = (ny + 1) * (elx + 1) + ely
            edof_mat[el, 0] = 2 * n1 + 2
            edof_mat[el, 1] = 2 * n1 + 3
            edof_mat[el, 2] = 2 * n2 + 2
            edof_mat[el, 3] = 2 * n2 + 3
            edof_mat[el, 4] = 2 * n2 
            edof_mat[el, 5] = 2 * n2 + 1
            edof_mat[el, 6] = 2 * n1 
            edof_mat[el, 7] = 2 * n1 + 1

    n_el = nx * ny
    i_k = []
    j_k = []

    for element in range(n_el):
        for j in range(8):
            for i in range(8):
                i_k.append(edof_mat[element, i])
                j_k.append(edof_mat[element, j])

    return edof_mat, i_k, j_k

def prepare_filter(nx : int, ny : int, rmin : float):
    """
    """
    nfilter=int(nx*ny*((2*(np.ceil(rmin)-1)+1)**2))
    iH = np.zeros(nfilter)
    jH = np.zeros(nfilter)
    sH = np.zeros(nfilter)
    cc=0
    for i in range(nx):
        for j in range(ny):
            row=i*ny+j
            kk1=int(np.maximum(i-(np.ceil(rmin)-1),0))
            kk2=int(np.minimum(i+np.ceil(rmin),nx))
            ll1=int(np.maximum(j-(np.ceil(rmin)-1),0))
            ll2=int(np.minimum(j+np.ceil(rmin),ny))
            for k in range(kk1,kk2):
                for l in range(ll1,ll2):
                    col=k*ny+l
                    fac=rmin-np.sqrt(((i-k)*(i-k)+(j-l)*(j-l)))
                    iH[cc]=row
                    jH[cc]=col
                    sH[cc]=np.maximum(0.0,fac)
                    cc=cc+1

    H=coo_matrix((sH,(iH,jH)),shape=(nx*ny,nx*ny)).tocsc()	
    Hs=H.sum(1)
    return H, Hs

def initialize_design_variables(nx : int, ny : int, volfrac : float):
    """
    Create the base mesh that is a uniform density (of volfrac)
    """
    return volfrac * np.ones(ny * nx)

def compute_sk(ke, x_phys, emin, e0, penal):
    """
    Compute the value of the stiffness matrix using the SIMP method
    """
    return ((ke.flatten()[np.newaxis]).T*(emin+(x_phys)**penal*(e0-emin))).flatten(order='F')    

def fe_analysis(u, nx, ny, x_phys, i_k, j_k, f, free_dofs, e0, emin, penal, ke):
    """
    """
    s_k = compute_sk(ke, x_phys, emin, e0, penal)
    ndof = 2 * (nx + 1) * (ny + 1)

    k = coo_matrix((s_k,(i_k,j_k)),shape=(ndof,ndof))
    k = k.tocsc()
    
    k = k[free_dofs, :][:, free_dofs]

    u[free_dofs, 0] = spsolve(k, f[free_dofs, 0])

    return u

def compute_ce(u, edof_mat, ke, nx, ny):
    """
    """
    return (np.dot(u[edof_mat].reshape(nx * ny, 8), ke) * u[edof_mat].reshape(nx * ny, 8)).sum(1)

def compute_c(x_phys, ce, emin, e0, penal):
    """

    """
    return ((emin+ x_phys**penal * (e0-emin)) * ce).sum()

def compute_dc(x_phys, ce, emin, e0, penal):
    """

    """
    return (-penal * x_phys**(penal - 1) * (e0 - emin))*ce

def compute_objective_and_sensibility(x_phys, u, edof_mat, ke, e0, emin, penal, nx, ny):
    ce = compute_ce(u, edof_mat, ke, nx, ny)
    c = compute_c(x_phys, ce, emin, e0, penal)
    dc = compute_dc(x_phys, ce, emin, e0, penal)
    dv = np.ones(nx * ny)

    return c, dc, dv

def update_design_variables(x, dc, dv, volfrac, nx, ny):
    l1 = 0
    l2 = 1e9
    m = 0.2

    x_new = np.zeros(nx * ny)

    while (l2-l1) / (l2 + l1) > 1e-3:
        lmid = (l1 + l2) / 2
        x_new[:] = np.maximum(0.0,np.maximum(x-m,np.minimum(1.0,np.minimum(x+m,x*np.sqrt(-dc/dv/lmid)))))
        if x_new.sum() > volfrac * nx * ny:
            l1 = lmid
        else:
            l2 = lmid
    
    return x_new


def apply_filter(x, h, hs, dc):
    dc[:] = np.asarray((h*(x*dc))[np.newaxis].T/hs)[:,0] / np.maximum(0.001,x)
    return dc

def compute_change(x, x_phys):
    return np.linalg.norm(x.reshape(nx * ny,1)-x_phys.reshape(nx * ny,1),np.inf)


# ---------------- PROBLEM DEFINITION -------------

def define_loads_and_supports(nx, ny):
    """
    Returns the boundary conditions, that is the force vector as well as the fixed and free degrees of freedom

    Arguments : 
        nx (int) : number of element along the x axis
        ny (int) : number of element along the y axis

    Returns : 
        f (float list) : vector of size ndof containing the norm of the force being applied to each dof
        fixed (int list) : vector containing the indices of the fixed dofs
        free (int list) : vector containing the indices of the free dofs
    """
    ndof = 2 * (nx + 1) * (ny + 1)

    f=np.zeros((ndof,1))
    f[1]=-1

    dofs=np.arange(2*(nx+1)*(ny+1))
    #fixed = np.union1d(dofs[0:(ny+1)//2:2],dofs[(ny+1):2*(ny+1):2])
    fixed=np.union1d(dofs[0:2*(ny+1):2],np.array([2*(nx+1)*(ny+1)-1]))
    free=np.setdiff1d(dofs,fixed)
    return f, fixed, free


if __name__ == "__main__":
    nx = 180
    ny = 60
    volfrac = 0.5
    penal = 3
    rmin = 1.5
    optimise(nx, ny, volfrac, penal, rmin)
