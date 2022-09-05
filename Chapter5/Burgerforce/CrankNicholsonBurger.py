import numpy as np

#Values of N and M you would like to run the method over.
Nset=[5,9,17,33,65,129]
Mset=[5,9,17,33,65,129]

def CrankNicholsonBurger(N,M):
    #definition of numerical parameters
    M = M #number of spatial grid points
    N = N #number of time steps
    dt = 30.0/(N-1) #time step
    L = float(1) #size of grid
    dx = L/(M-1) #grid spacing
    coef=1 #heat diffusion coefficient
    omega=3
    pi=np.pi

    def F(t, x):return 10*np.sin(2*pi*x*omega)*np.cos(t*pi/10)+2*np.abs(np.sin(x*pi*omega)*np.cos(t*pi/10))

    r = coef*0.5*dt/dx**2
    p = 0.25*dt/dx

    #initialize matrices A, B and b array
    A = np.zeros((M-2,M-2))
    B = np.zeros((M-2,M-2))
    b = np.zeros((M-2))

    #initialize grid
    x = np.linspace(0,1,M)
    #initial condition
    u=0*x
    uall=0*x
    #evaluate right hand side at t=0
    bb = B.dot(u[1:-1]) + b

    c = 0
    for time in range(N):
        #define matrices A, B and b array
        for i in range(M-2): #i is spatial variable
            if i==0: #0 here is actually the first spatial point beyond the boundary
                A[i,:] = [1+2*r-p*u[0]+p*u[2] if j==0 else (-r+p*u[1]) if j==1 else 0 for j in range(M-2)]
                B[i,:] = [1-2*r if j==0 else r if j==1 else 0 for j in range(M-2)]
                b[i] = 0.5*dt*(F(time*dt,dx)+F((time+1)*dt,dx)) #boundary condition at i=1, plus forcing
            elif i==M-3:
                A[i,:] = [-r-p*u[M-2] if j==M-4 else 1+2*r+p*u[M-1]-p*u[M-3] if j==M-3 else 0 for j in range(M-2)]
                B[i,:] = [r if j==M-4 else 1-2*r if j==M-3 else 0 for j in range(M-2)]
                b[i] = 0.5*dt*(F(time*dt,(M-1)*dx)+F((time+1)*dt,(M-2)*dx)) #boundary condition at i=N, plus forcing
            else:
                A[i,:] = [-r-p*u[i+1] if j==i-1 else (-r+p*u[i+1]) if j==i+1 else 1+2*r+p*u[i+2]-p*u[i] if j==i else 0 for j in range(M-2)]
                B[i,:] = [r if j==i-1 or j==i+1 else 1-2*r if j==i else 0 for j in range(M-2)]
                b[i] = 0.5*dt*(F(time*dt,(i+1)*dx)+F((time+1)*dt,(i+1)*dx))  #forcing

        # update right hand side
        bb = B.dot(u[1:-1]) + b
        #find solution inside domain
        u[1:-1] = np.linalg.solve(A,bb)
        # update right hand side
        bb = B.dot(u[1:-1]) + b
        if time>=1:
            uall = np.append(uall, u)

    matlabsol = np.loadtxt("burgerforcedmatlab1.txt", comments="#", delimiter=",", unpack=False)
    matlabsubset = matlabsol[0::int(np.size(matlabsol, 0) / (M - 1)), 0::int(np.size(matlabsol, 1) / (N - 1))]

    truesolution = 0*x
    for i in range(1, N):
        truesolution = np.append(truesolution, matlabsubset[:, i])

    maxerror = np.max(np.abs(truesolution - uall))

    results = [N,M,maxerror]

    import csv

    with open("CrankNicholsonBurger.txt", "a") as f:
        wr = csv.writer(f)
        wr.writerow(results)
        f.close()

for N in Nset:
    for M in Mset:
        CrankNicholsonBurger(N,M)
