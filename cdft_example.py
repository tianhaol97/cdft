import numpy as np
import gzip, pickle
import scipy.optimize, scipy.spatial
from scipy.optimize import linprog
# from saft_landscape import solveX, full_freeenergy, ideal_chempot, assoc_chempot
import math
import matplotlib.pyplot as plt
from scipy.interpolate import splprep, splev
import networkx as nx
import sys

plt.rc('xtick',labelsize=14)
plt.rc('ytick',labelsize=14)

# idx = int(sys.argv[-1])
idx = 0

bcl = np.array([1.0])
bl = np.array([0.2])
cl = np.array([0.5])
nbcl = len(bcl)
nbl = len(bl)
ncl = len(cl)
para = {i*nbl*ncl+j*ncl+k:(bcl[i],bl[j],cl[k]) for i in range(nbcl) 
        for j in range(nbl) for k in range(ncl)}

bc = para[idx][0]

def fe(rho):
    rhoe = np.zeros(nc+1)
    rhoe[:nc] = np.copy(rho)
    rhoe[-1] = 1-np.sum(rho)
    if np.all(rhoe>=-10**(-9)):
        f = np.dot(np.matmul(rhoe,kai),rhoe)/2
        for i in range(nc+1):
            if rhoe[i]>0:
                f = f+rhoe[i]*math.log(rhoe[i])
    else:
        f = 100000
    return f



def mu0(rho):
    rhot = 1-np.sum(rho)
    dirv = np.zeros(nc)
    for i in range(nc):
        dirv[i] = math.log(rho[i])-math.log(rhot)+rhot*kai[-1,i]+np.dot(kai[i,:-1],rho)-np.dot(kai[-1,:-1],rho)
    return dirv

def partiald(densi):
    pd = np.zeros([npoint,nc])
    for j in range(npoint):
        pd[j,:] = mu0(densi[j+2,:])-muc+np.matmul(-densi[j+4,:]+2*densi[j+2,:]-densi[j,:],m)/4/h**2
    return pd

a = 1.5
b = para[idx][1]
c = para[idx][2]
egv = 0
um = -np.array([[a,0,b],[0,a,b],[b,b,c]])
kai = np.zeros([4,4])
for i in range(3):
    for j in range(3):
        kai[i,j] = 3*(2*um[i,j]-um[i,i]-um[j,j])
    kai[i,3] = -3*um[i,i]
    kai[3,i] = -3*um[i,i]
w,v = np.linalg.eig(-um)
odx = np.argsort(w)
w = w[odx]
v = v[:,odx]
nw = 1
if w[0]<0:
    print('negative eigen = ',w[0])
    nw = w[0]
    w[0] = egv
m= v @ np.diag(w) @ v.transpose()
# m = 2*np.eye(3)
# m = np.diag([2,2,0.5])
# kai = np.array([[0,9,3,4.5],[9,0,3,4.5],[3,3,0,1.5],[4.5,4.5,1.5,0]])
cpoint = 101
npoint = 1000
nc = 3
h = 0.02

resu=np.load('coexre{}_{}_{}.npy'.format(b,c,bc))

phic = resu[:3,:]
muc = resu[3,:]
# phi = sorted(phi, key=lambda p: p.sum())
# muc = mu


phia = phic[-1]
phib = phic[-2]

# omega0 = fe(phia)-np.dot(muc,phia)
thres = 0.001
stp = 0.0005
maxk = 10000

def GD(inid):
    densi = np.copy(inid)
    k = 0
    #### calculate gradient
    pd = partiald(densi)
    #### calculate grand potential
    oomega = tomega(densi)
    while (np.linalg.norm(pd)>thres and k<maxk):
        step = stp
        ndensi[2:-2,:] = np.copy(densi[2:-2,:])-step*pd
        nomega = tomega(ndensi)
        while nomega>oomega:
            step = step/2
            print('step=',step)
            ndensi[2:-2,:] = np.copy(densi[2:-2,:])-step*pd
            nomega = tomega(ndensi)
            print('nomega = {}'.format(nomega))
        densi = np.copy(ndensi)
        pd = partiald(densi)
        oomega = nomega
        k = k+1
        # print('k ={},grad = {}, omega = {}'.format(k,np.linalg.norm(pd),oomega))
    return densi,pd


def omegaz(densi):
    omega = np.zeros(npoint+2)
    omega1 = np.zeros(npoint+2)
    omega2 = np.zeros(npoint+2)
    for i in range(npoint+2):
        if np.all(densi[i+1,:]>0):
            omega1[i] = fe(densi[i+1,:])-np.dot(muc,densi[i+1,:])-omega0
            omega2[i] = np.dot(np.matmul(densi[i+2,:]-densi[i,:],m),
                                                densi[i+2,:]-densi[i,:])/8/h**2
            omega[i] = fe(densi[i+1,:])-np.dot(muc,densi[i+1,:])-omega0+np.dot(np.matmul(densi[i+2,:]-densi[i,:],m),
                                                densi[i+2,:]-densi[i,:])/8/h**2
        else:
            omega[i] = 100000
    return omega,omega1,omega2

def tomega(densi):
    omega = np.zeros(npoint+2)
    for i in range(npoint+2):
        if np.all(densi[i+1,:]>0):
            omega[i] = fe(densi[i+1,:])-np.dot(muc,densi[i+1,:])-omega0+np.dot(np.matmul(densi[i+2,:]-densi[i,:],m),
                                                densi[i+2,:]-densi[i,:])/8/h**2
        else:
            omega[i] = 100000
    omegas = np.sum(omega)
    return omegas

z = np.zeros(npoint+4)

for i in range(npoint+4):
    z[i] = (i-(npoint+3)/2)*h



### one possible way to initialize the density prodile

ss = 300

densi = np.zeros([npoint+4,nc])

densi[:ss,:] = phia

densi[-ss:,:] = phib


densi[ss:-ss,:] = np.linspace(phia,phib,num=len(densi[ss:-ss,1]))

### alternatively, initialize from previous results
# densi = np.load('dprof{0}_{1}_{2}_ab.npy'.format(b,c,bc))


ndensi = np.copy(densi)
muc = mu0(densi[0])
omega0 = fe(densi[0])-np.dot(muc,densi[0])
print('initial',tomega(densi))
for i in range(3):
    npf,fpd = GD(densi)
## symmetrize the density profile (optional)
    if np.argmax(fpd)>1500:

        for i in range(502,1004):
            npf[i,0] = npf[1003-i,1]
            npf[i,1] = npf[1003-i,0]
            npf[i,2] = npf[1003-i,2]
        else:
            for i in range(502):
                npf[i,0] = npf[1003-i,1]
                npf[i,1] = npf[1003-i,0]
                npf[i,2] = npf[1003-i,2]
    densi = np.copy(npf)
    print('i ={},grad = {}'.format(i,np.linalg.norm(fpd)))

