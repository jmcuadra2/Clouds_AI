"""Real-Time Fluid Dynamics for Games by Jos Stam (2003).

Parts of author's work are also protected
under U. S. patent #6,266,071 B1 [Patent].

Reference:
 https://graphics.cs.cmu.edu/nsp/course/15-464/Fall09/papers/StamFluidforGames.pdf

The code hereby has been used in this paper:

A Novel Method for Virtual Real-Time Cumuliform Fluid Dynamics Simulation Using Deep Recurrent Neural Networks
Mathematics-MDPI paper at: https://doi.org/10.3390/math13172746 

Author of this Python code: 

Copyright (c) 2015 Alberto Santini (MIT license)
https://github.com/albertosantini/python-fluid?tab=MIT-1-ov-file#readme

"""

# Size of 3D grid

M = 30 # X
N = 7 # Y
O = 7 # Z


def IX(i,j,k):
    return ((i)+(M+2)*(j) + (M+2)*(N+2)*(k))

def SWAP(a, b):
	return b,a

def add_source(M,N,O,x,s,dt):
    """Addition of forces: the density increases due to sources."""
    size = (M+2)*(N+2)*(O+2)
    x[0:size] += dt * s[0:size]

def set_bnd(M, N, O, b, x):
    # I assume unbounded fluid. Free space of the atmosphere, then I eliminate this function.

def lin_solve(M, N, O, b, x, x0, a, c):
    """lin_solve."""

    for l in range(0, 10):
         # iterate the solver
		 # update for each cell
        for i in range(1,M+1):
            for j in range(1,N+1):
                for k in range(1,O+1):
                    x[IX(i,j,k)] = (x0[IX(i,j,k)] + a*(x[IX(i-1,j,k)]+x[IX(i+1,j,k)]+x[IX(i,j-1,k)]+x[IX(i,j+1,k)]+x[IX(i,j,k-1)]+x[IX(i,j,k+1)]))/c;

        set_bnd(M,N,O,b,x);
	

def diffuse(M, N, O, b, x, x0, diff, dt):
    """Diffusion: the density diffuses at a certain rate.

    The basic idea behind our method is to find the densities which when
    diffused backward in time yield the densities we started with. The simplest
    iterative solver which works well in practice is Gauss-Seidel relaxation.
    """ 
    m = max(max(M,N), max(N,O)) 
    a = dt * diff * m * m * m
    lin_solve(M,N,O, b, x, x0, a, 1 + 6 * a)


def advect(M, N, O, b, d, d0, u, v, w, dt):
    """Advection: the density follows the velocity field.

    The basic idea behind the advection step. Instead of moving the cell
    centers forward in time through the velocity field, we look for the
    particles which end up exactly at the cell centers by tracing backwards in
    time from the cell centers.
    """
    dtx=dty=dtz=dt*max(max(M, N), max(N, O))
    for i in range(1,M+1):
        for j in range(1,N+1):
            for k in range(1,O+1):
                x = i-dtx*u[IX(i,j,k)]
                y = j-dty*v[IX(i,j,k)]
                z = k-dtz*w[IX(i,j,k)]
                if (x<0.5):
                    x=0.5 
                if (x>M+0.5):
                    x=M+0.5 
                i0=int(x)
                i1=i0+1
                if (y<0.5):
                    y=0.5
                if (y>N+0.5):
                    y=N+0.5
                j0=int(y)
                j1=j0+1
                if (z<0.5):
                    z=0.5	
                if (z>O+0.5):
                    z=O+0.5
                k0=int(z)
                k1=k0+1

                s1 = x-i0
                s0 = 1-s1
                t1 = y-j0
                t0 = 1-t1
                u1 = z-k0
                u0 = 1-u1
               
                d[IX(i,j,k)] = s0*(t0*u0*d0[IX(i0,j0,k0)]+t1*u0*d0[IX(i0,j1,k0)]+t0*u1*d0[IX(i0,j0,k1)]+t1*u1*d0[IX(i0,j1,k1)])+s1*(t0*u0*d0[IX(i1,j0,k0)]+t1*u0*d0[IX(i1,j1,k0)]+t0*u1*d0[IX(i1,j0,k1)]+t1*u1*d0[IX(i1,j1,k1)])
    set_bnd (M, N, O, b, d )


def project(M, N, O, u, v, w, p, div):
    for i in range(1,M+1):
        for j in range(1,N+1):
            for k in range(1,O+1):
	
                div[IX(i,j,k)] = -1.0/3.0*((u[IX(i+1,j,k)]-u[IX(i-1,j,k)])/M+(v[IX(i,j+1,k)]-v[IX(i,j-1,k)])/M+(w[IX(i,j,k+1)]-w[IX(i,j,k-1)])/M)
                p[IX(i,j,k)] = 0
	
    set_bnd ( M, N, O, 0, div )
    set_bnd (M, N, O, 0, p )

    lin_solve ( M, N, O, 0, p, div, 1, 6 );


    for i in range(1,M+1):
        for j in range(1,N+1):
            for k in range(1,O+1):
                u[IX(i,j,k)] -= 0.5*M*(p[IX(i+1,j,k)]-p[IX(i-1,j,k)])
                v[IX(i,j,k)] -= 0.5*M*(p[IX(i,j+1,k)]-p[IX(i,j-1,k)])
                w[IX(i,j,k)] -= 0.5*M*(p[IX(i,j,k+1)]-p[IX(i,j,k-1)])

	
    set_bnd (M, N, O, 1, u)
    set_bnd (M, N, O, 2, v)
    set_bnd (M, N, O, 3, w)


def dens_step(M, N, O, x, x0, u, v, w, diff, dt):
    add_source ( M, N, O, x, x0, dt )
    x0, x = SWAP ( x0, x )
    diffuse ( M, N, O, 0, x, x0, diff, dt )
    x0, x = SWAP ( x0, x )
    advect ( M, N, O, 0, x, x0, u, v, w, dt)

def vel_step(M, N, O, u, v, w, u0, v0, w0, visc, dt):
    add_source ( M, N, O, u, u0, dt)
    add_source ( M, N, O, v, v0, dt)
    add_source ( M, N, O, w, w0, dt)
    u0, u = SWAP ( u0, u )
    diffuse ( M, N, O, 1, u, u0, visc, dt )
    v0, v = SWAP ( v0, v )
    diffuse ( M, N, O, 2, v, v0, visc, dt )
    w0, w = SWAP ( w0, w )
    diffuse ( M, N, O, 3, w, w0, visc, dt )
    project ( M, N, O, u, v, w, u0, v0 );
    u0, u = SWAP ( u0, u )
    v0, v = SWAP ( v0, v )
    w0, w = SWAP ( w0, w )
    advect ( M, N, O, 1, u, u0, u0, v0, w0, dt )
    advect ( M, N, O, 2, v, v0, u0, v0, w0, dt )
    advect ( M, N, O, 3, w, w0, u0, v0, w0, dt )
    project ( M, N, O, u, v, w, u0, v0 )

