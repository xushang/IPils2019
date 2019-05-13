import numpy as np
import scipy.sparse as sp

def solve(q,c,dt,dx,T=1.0,L=1.0,n=1):
	'''Solve n-dim wave equation using Leap-Frog scheme: u_{n+1} = Au_n + Bu_{n-1} + Cq_n'''
	# define some quantities
	gamma = dt*c/dx
	nt = int(T/dt + 1)
	nx = int(2*L/dx + 2)
	#
	q.resize((nx**n,nt))
	# define matrices
	A,B,C,L = getMatrices(gamma,nx,n)
	
	# main loop
	u = np.zeros((nx**n,nt))
	
	for k in range(1,nt-1):
		u[:,k+1] = A@u[:,k] + L@u[:,k] + B@u[:,k-1] + (dx**2)*C@q[:,k]
	
	return u

def multiply(u,c,dt,dx,T=1.0,L=1.0,n=1):
	# define some quantities
	gamma = dt*c/dx
	nt = int(T/dt + 1)
	nx = int(2*L/dx + 2)
	
	#
	u.resize((nx**2,nt))
	
	# define matrices
	A,B,C,L = getMatrices(gamma,nx,n)
	
	# main loop
	q = np.zeros((nx**n,nt))
	
	for k in range(1,nt-1):
		q[k] = (u[:,k+1] - 2*u[:,k] + u[:,k-1] - L@u[:,k])/(dt*c)**2
	
	return u

def sample(xin,xout1,xout2=[]):
	'''Spatial sampling by simple interpolation'''

	if len(xout2):
		n = 2
	else:
		n = 1
		
	m = len(xout1)
	nx = len(xin)
	
	rw = []
	cl = []
	nz = []
	
	if n == 1:
		for k in range(m):
			i = 0
			while xin[i] < xout1[k]:
				i = i + 1
			if i < nx - 1:
				a = (xout1[k] - xin[i+1])/(xin[i] - xin[i+1])
				b = (xout1[k] - xin[i])/(xin[i+1] - xin[i])
				rw.append(k)
				cl.append(i)
				nz.append(a)
				rw.append(k)
				cl.append(i+1)
				nz.append(b) 
		P = sp.coo_matrix((nz,(rw,cl)),shape=(m,nx))
	else:
		for k in range(m):
			i = 0
			j = 0
			while xin[i] < xout1[k]:
				i = i + 1
			while xin[j] < xout2[k]:
				j = j + 1	
			if i < nx - 1 and j < nx - 1:
				a = (xout1[k] - xin[i+1])*(xout2[k] - xin[j+1])/(xin[i] - xin[i+1])/(xin[j] - xin[j+1])
				b = (xout1[k] - xin[i])*(xout2[k] - xin[j+1])/(xin[i+1] - xin[i])/(xin[j] - xin[j+1])
				c = (xout1[k] - xin[i+1])*(xout2[k] - xin[j])/(xin[i] - xin[i+1])/(xin[j+1] - xin[j])
				d = (xout1[k] - xin[i])*(xout2[k] - xin[j])/(xin[i+1] - xin[i])/(xin[j+1] - xin[j])
				
				rw.append(k)
				cl.append(i+nx*j)
				nz.append(a)
				
				rw.append(k)
				cl.append(i+1+nx*j)
				nz.append(b)
				
				rw.append(k)
				cl.append(i+nx*(j+1))
				nz.append(c)
				
				rw.append(k)
				cl.append(i+1+nx*(j+1))
				nz.append(d)
		P = sp.coo_matrix((nz,(rw,cl)),shape=(m,nx*nx))
		
	return P
	
def getMatrices(gamma,nx,n):

# setup matrices
	l = (gamma**2)*np.ones((3,nx))
	l[1,:] = -2*(gamma**2)
	l[1,0] = -gamma
	l[2,0] = gamma
	l[0,nx-2] = gamma
	l[1,nx-1] = -gamma
	
	if n == 1:
		a = 2*np.ones(nx)
		a[0] = 1
		a[nx-1] = 1
	
		b = -np.ones(nx)
		b[0] = 0
		b[nx-1] = 0
	
		c = (gamma)**2*np.ones(nx)
		c[0] = 0
		c[nx-1] = 0

		L = sp.diags(l,[-1, 0, 1],shape=(nx,nx))

	else:
		a = 2*np.ones((nx,nx))
		a[0,:] = 1
		a[nx-1,:] = 1
		a[:,0] = 1
		a[:,nx-1] = 1
		a.resize(nx**2)
		
		b = -np.ones((nx,nx))
		b[0,:] = 0
		b[nx-1,:] = 0
		b[:,0] = 0
		b[:,nx-1] = 0
		b.resize(nx**2)
	
		c = (gamma)**2*np.ones((nx,nx))
		c[0,:] = 0
		c[nx-1,:] = 0
		c[:,0] = 0
		c[:,nx-1] = 0
		c.resize(nx**2)
		
		L = sp.kron(sp.diags(l,[-1, 0, 1],shape=(nx,nx)),sp.eye(nx)) + sp.kron(sp.eye(nx),sp.diags(l,[-1, 0, 1],shape=(nx,nx)))
		
	A = sp.diags(a)
	B = sp.diags(b)
	C = sp.diags(c)
	
	return A,B,C,L