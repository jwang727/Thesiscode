import numpy as np
from numpy import linalg as la

#Homogeneous Burger's Equation, rational quadratic kernel

#Values of N and M you would like to run the method over.
Nset=[5,9,17,33,65,129]
Mset=[5,9,17,33,65,129]

seed=10
np.random.seed(seed)

coef=0.02 #coefficient of the double x derivative term in the heat equation
pi=np.pi

def nearestPD(A):
    """Find the nearest positive-definite matrix to input
    A Python/Numpy port of John D'Errico's `nearestSPD` MATLAB code [1], which
    credits [2].
    [1] https://www.mathworks.com/matlabcentral/fileexchange/42885-nearestspd
    [2] N.J. Higham, "Computing a nearest symmetric positive semidefinite
    matrix" (1988): https://doi.org/10.1016/0024-3795(88)90223-6
    """

    B = (A + A.T) / 2
    _, s, V = la.svd(B)

    H = np.dot(V.T, np.dot(np.diag(s), V))

    A2 = (B + H) / 2

    A3 = (A2 + A2.T) / 2

    if isPD(A3):
        return A3

    spacing = np.spacing(la.norm(A))
    # The above is different from [1]. It appears that MATLAB's `chol` Cholesky
    # decomposition will accept matrixes with exactly 0-eigenvalue, whereas
    # Numpy's will not. So where [1] uses `eps(mineig)` (where `eps` is Matlab
    # for `np.spacing`), we use the above definition. CAVEAT: our `spacing`
    # will be much larger than [1]'s `eps(mineig)`, since `mineig` is usually on
    # the order of 1e-16, and `eps(1e-16)` is on the order of 1e-34, whereas
    # `spacing` will, for Gaussian random matrixes of small dimension, be on
    # othe order of 1e-16. In practice, both ways converge, as the unit test
    # below suggests.
    I = np.eye(A.shape[0])
    k = 1
    while not isPD(A3):
        mineig = np.min(np.real(la.eigvals(A3)))
        A3 += I * (-mineig * k**2 + spacing)
        k += 1

    return A3

def isPD(B):
    """Returns true when input is positive-definite, via Cholesky"""
    try:
        _ = la.cholesky(B)
        return True
    except la.LinAlgError:
        return False



def Burgerrational(N,M):

    # hyper-parameter(s)
    sig=1
    sig1=1
    alpha=3
    alpha1=3
    beta=1
    beta1=1

    # Kernel functions

    def k (r1,r2):
        r1=np.atleast_1d(r1)
        r2=np.atleast_1d(r2)

        return (sig**2)*(1+np.subtract.outer(r1,r2)**2/(alpha*beta))**(-beta)


    def dr1_k (r1,r2):
        r1=np.atleast_1d(r1)
        r2=np.atleast_1d(r2)

        return -2*(sig**2/alpha)*np.subtract.outer(r1, r2)*(1 + np.subtract.outer(r1, r2) ** 2 / (alpha*beta))**(-beta-1);

    def dr2_k (r1,r2):
        r1 = np.atleast_1d(r1)
        r2 = np.atleast_1d(r2)

        return dr1_k(r2,r1).T

    def dr1r2_k (r1,r2):
        r1 = np.atleast_1d(r1)
        r2 = np.atleast_1d(r2)

        return 2*(sig**2/alpha)*(1 + np.subtract.outer(r1, r2) ** 2 / (alpha*beta))**(-beta-1)-4*((beta+1)/beta)*(sig**2/alpha**2)*np.subtract.outer(r1, r2)**2*(1 + np.subtract.outer(r1, r2) ** 2 / (alpha*beta))**(-beta-2);

    def dr1r1_k (r1,r2):
        r1 = np.atleast_1d(r1)
        r2 = np.atleast_1d(r2)

        return -dr1r2_k (r1,r2)

    def dr2r2_k (r1,r2):
        r1 = np.atleast_1d(r1)
        r2 = np.atleast_1d(r2)

        return dr1r1_k (r1,r2)

    def dr1r1r2_k(r1, r2):
        r1 = np.atleast_1d(r1)
        r2 = np.atleast_1d(r2)

        return -12*(sig**2/alpha**2)*((beta+1)/beta)*np.subtract.outer(r1, r2)*(1 + np.subtract.outer(r1, r2) ** 2 / (alpha*beta))**(-beta-2)+8*((beta+2)*(beta+1)/beta**2)*(sig**2/alpha**3)*np.subtract.outer(r1, r2)**3*(1 + np.subtract.outer(r1, r2) ** 2 / (alpha*beta))**(-beta-3)

    def dr1r2r2_k (r1,r2):
        r1 = np.atleast_1d(r1)
        r2 = np.atleast_1d(r2)

        return dr1r1r2_k(r2, r1).T

    def dr1r1r2r2_k (r1,r2):
        r1 = np.atleast_1d(r1)
        r2 = np.atleast_1d(r2)

        return 12*(sig**2/alpha**2)*((beta+1)/beta)*(1 + np.subtract.outer(r1, r2) ** 2 / (alpha*beta))**(-beta-2)-48*((beta+2)*(beta+1)/beta**2)*(sig**2/alpha**3)*np.subtract.outer(r1, r2)**2*(1 + np.subtract.outer(r1, r2) ** 2 / (alpha*beta))**(-beta-3)+16*((beta+3)*(beta+2)*(beta+1)/beta**3)*(sig**2/alpha**4)*np.subtract.outer(r1, r2)**4*(1 + np.subtract.outer(r1, r2) ** 2 / (alpha*beta))**(-beta-4)


    def c (r1,r2):
        r1=np.atleast_1d(r1)
        r2=np.atleast_1d(r2)

        return (sig1**2)*(1+np.subtract.outer(r1,r2)**2/(alpha1*beta1))**(-beta1)


    def dr1_c (r1,r2):
        r1=np.atleast_1d(r1)
        r2=np.atleast_1d(r2)

        return -2*(sig1**2/alpha1)*np.subtract.outer(r1, r2)*(1 + np.subtract.outer(r1, r2) ** 2 / (alpha1*beta1))**(-beta1-1)

    def dr2_c (r1,r2):
        r1 = np.atleast_1d(r1)
        r2 = np.atleast_1d(r2)

        return dr1_c(r2,r1).T

    def dr1r2_c (r1,r2):
        r1 = np.atleast_1d(r1)
        r2 = np.atleast_1d(r2)

        return 2*(sig1**2/alpha1)*(1 + np.subtract.outer(r1, r2) ** 2 / (alpha1*beta1))**(-beta1-1)-4*((beta1+1)/beta1)*(sig1**2/alpha1**2)*np.subtract.outer(r1, r2)**2*(1 + np.subtract.outer(r1, r2) ** 2 / (alpha1*beta1))**(-beta1-2);

    def dr1r1_c (r1,r2):
        r1 = np.atleast_1d(r1)
        r2 = np.atleast_1d(r2)

        return -dr1r2_k (r1,r2)

    def dr2r2_c (r1,r2):
        r1 = np.atleast_1d(r1)
        r2 = np.atleast_1d(r2)

        return dr1r1_c (r1,r2)

    def dr1r1r2_c(r1, r2):
        r1 = np.atleast_1d(r1)
        r2 = np.atleast_1d(r2)

        return -12*(sig1**2/alpha1**2)*((beta1+1)/beta1)*np.subtract.outer(r1, r2)*(1 + np.subtract.outer(r1, r2) ** 2 / (alpha1*beta1))**(-beta1-2)+8*((beta1+2)*(beta1+1)/beta1**2)*(sig**2/alpha**3)*np.subtract.outer(r1, r2)**3*(1 + np.subtract.outer(r1, r2) ** 2 / (alpha1*beta1))**(-beta1-3)

    def dr1r2r2_c (r1,r2):
        r1 = np.atleast_1d(r1)
        r2 = np.atleast_1d(r2)

        return dr1r1r2_c(r2, r1).T

    def dr1r1r2r2_c (r1,r2):
        r1 = np.atleast_1d(r1)
        r2 = np.atleast_1d(r2)

        return 12*(sig1**2/alpha1**2)*((beta1+1)/beta1)*(1 + np.subtract.outer(r1, r2) ** 2 / (alpha1*beta1))**(-beta1-2)-48*((beta1+2)*(beta1+1)/beta1**2)*(sig1**2/alpha1**3)*np.subtract.outer(r1, r2)**2*(1 + np.subtract.outer(r1, r2) ** 2 / (alpha1*beta1))**(-beta1-3)+16*((beta1+3)*(beta1+2)*(beta1+1)/beta1**3)*(sig1**2/alpha1**4)*np.subtract.outer(r1, r2)**4*(1 + np.subtract.outer(r1, r2) ** 2 / (alpha1*beta1))**(-beta1-4)

    def sigma(t1,t2,r1,r2):
        return np.kron(k(t1,t2),c(r1,r2))

    def d1x_sigma(t1,t2,r1,r2,uhat):
        return -coef*np.kron(k(t1,t2), dr1r1_c(r1, r2))+np.kron(k(t1,t2), dr1_c(r1, r2))*uhat

    def d2x_sigma(t1,t2,r1,r2,uhat):
        return -coef*np.kron(k(t1,t2), dr2r2_c(r1, r2))+np.kron(k(t1,t2), dr2_c(r1, r2))*uhat

    def d1x_d2x_sigma(t1,t2,r1,r2,uhat1,uhat2):
        return (coef**2)*np.kron(k(t1,t2), dr1r1r2r2_c(r1, r2))+np.kron(k(t1,t2), dr1r2_c(r1, r2))*uhat1*uhat2-coef*np.kron(k(t1,t2), dr1r2r2_c(r1, r2))*uhat2-coef*np.kron(k(t1,t2), dr1r1r2_c(r1, r2))*uhat1

    def d1t_sigma(t1,t2,r1,r2,uhat):
        return np.kron(dr1_k(t1,t2),c(r1,r2))

    def d2t_sigma(t1,t2,r1,r2,uhat):
        return np.kron(dr2_k(t1,t2),c(r1,r2))

    def d1t_d2t_sigma(t1,t2,r1,r2,uhat):
        return np.kron(dr1r2_k(t1,t2),c(r1,r2))

    def d1t_d2x_sigma(t1,t2,r1,r2,uhat):
        return -coef*np.kron(dr1_k(t1,t2),dr2r2_c(r1, r2))+np.kron(dr1_k(t1,t2),dr2_c(r1, r2))*uhat

    def d1x_d2t_sigma(t1,t2,r1,r2,uhat):
        return -coef*np.kron(dr2_k(t1,t2),dr1r1_c(r1, r2))+np.kron(dr2_k(t1,t2),dr1_c(r1, r2))*uhat

    def d1_sigma(t1,t2,r1,r2,uhat1,uhat2):
        return d1t_sigma(t1,t2,r1,r2,uhat1)+d1x_sigma(t1,t2,r1,r2,uhat2)

    def d2_sigma(t1,t2,r1,r2,uhat1,uhat2):
        return d2t_sigma(t1,t2,r1,r2,uhat1)+d2x_sigma(t1,t2,r1,r2,uhat2)

    def d1_d2_sigma(t1,t2,r1,r2,uhat1,uhat2,uhat3):
        return d1t_d2t_sigma(t1,t2,r1,r2,uhat1)+d1t_d2x_sigma(t1,t2,r1,r2,uhat2)+d1x_d2t_sigma(t1,t2,r1,r2,uhat3)+d1x_d2x_sigma(t1,t2,r1,r2,uhat2,uhat3)

    # PDE definition
    np.random.seed(seed)
    t = np.linspace(0.00, 30.00, N)
    x = np.linspace(0.00, 2 * pi, M)

    a = 1
    b = 2
    d = 1

    x0 = min(x)
    xL = max(x)
    g_x = (2 * coef * a * d * np.sin(d * x)) / (b + a * np.cos(d * x))

    def F(t, y, x): return 0 * t

    #priors and initial variables
    muprior = np.array([0.0] * N * M)
    muiter = np.array([0.0] * N * M)
    dmuprior = np.array([0.0] * N * M)
    dmuiter = np.array([0.0] * N * M)

    t_0 = np.atleast_1d(t[0])

    noisedim=sigma(t_0, t_0, x[1:-1], x[1:-1]).shape[0]
    noise=0.0000001

    # Gaussian Process conditioning of the initial condition of PDE
    muiter = muprior + sigma(t, t_0, x, x[1:-1]) @ np.linalg.inv(sigma(t_0, t_0, x[1:-1], x[1:-1])+noise*np.eye(noisedim)) @ g_x[1:-1]
    Sigiter = sigma(t, t, x, x) - sigma(t, t_0, x, x[1:-1]) @ np.linalg.inv(sigma(t_0, t_0, x[1:-1], x[1:-1])+noise*np.eye(noisedim)) @ sigma(t_0, t, x[1:-1], x)

    # mean of u(t_{0},x) from the current prior mean to linearise the differential operator at the next timepoint t_0
    muiterrho = np.atleast_1d(muiter[0:M])
    # same as above but also keeps the 'best guess' for all remaining t, using the current mean.
    muiterrhofull = np.copy(muiter)
    muiterrhofull[0:M] = muiterrho

    muitermat = np.array([muiterrhofull, ] * N * M)

    uhatmat0allt = np.array([muiterrhofull, ] * (x.size - 2))

    dmuiter = dmuprior + d1_sigma(t, t_0, x, x[1:-1], 1, uhatmat0allt.T) @ np.linalg.inv(sigma(t_0, t_0, x[1:-1], x[1:-1])+noise*np.eye(noisedim)) @ g_x[1:-1]

    ddSigiter= d1_d2_sigma(t,t,x,x,1,muitermat,muitermat.T)-d1_sigma(t,t_0,x,x[1:-1],1,uhatmat0allt.T) @ np.linalg.inv(sigma(t_0,t_0,x[1:-1],x[1:-1])+noise*np.eye(noisedim)) @ d2_sigma(t_0,t,x[1:-1],x,1,uhatmat0allt)


    stop = N

    amppara = 0

    #Gaussian Process conditioning of boundary condition and gradient at each timepoint
    for i in range(1,stop+1):
        print('i')
        print(i)
        t_i=t[0:i]
        rho=np.repeat(t[0:i],M)
        position = np.tile(x, i)

        boundaryhigh = np.repeat(0, i)
        boundarylow = np.repeat(0, i)

        data = F(rho, muiterrho, position) - dmuprior[0:i*M]
        initialcond = np.hstack((g_x[1:-1] - muprior[1:M-1],boundarylow,boundaryhigh))
        datablock = np.hstack((initialcond, data))

        if i==1:
            #ddSigiter = d1_d2_sigma(t, t, x, x, 1, 0, 0)
            muiterrho0only=np.reshape(muiterrho,(muiterrho.size,1))
            dmuiter0only = np.reshape(dmuiter, (dmuiter.size, 1))
            newblock = ddSigiter[0:M, 0:M]
            newblock = nearestPD(newblock) + 0.000001 * np.eye(newblock.shape[0])
            amppara=amppara+np.matmul(np.matmul(np.reshape(F(t[i-1], muiterrho0only[(i-1)*M:i * M], x) - dmuiter0only[(i-1)*M:i * M,0], (1,x.size)),np.linalg.inv(newblock)),np.reshape(F(t[i-1], muiterrho0only[(i-1)*M:i * M], x) - dmuiter0only[(i-1)*M:i * M,0], (x.size,1)))
        else:
            newblock = ddSigiter[(i - 1) * M:i * M, (i - 1) * M:i * M]
            newblock = nearestPD(newblock) + 0.000001 * np.eye(newblock.shape[0])
            amppara=amppara+np.matmul(np.matmul(np.reshape(F(t[i-1], muiterrho[(i-1)*M:i * M], x) - dmuiter[(i-1)*M:i * M,0], (1,x.size)),np.linalg.inv(newblock)),np.reshape(F(t[i-1], muiterrho[(i-1)*M:i * M], x) - dmuiter[(i-1)*M:i * M,0], (x.size,1)))

        print(amppara)
        uhatmat = np.array([muiterrho, ] * t.size * x.size)
        uhatmat0 = np.array([muiterrho, ] * (x.size-2))
        uhatmatix0= np.array([muiterrho, ] * t_i.size)
        uhatmati = np.array([muiterrho, ] * (t_i.size)*x.size)


        d2_sigma_block = np.column_stack((sigma(t,t_0,x,x[1:-1]), sigma(t,t_i,x,x0),sigma(t,t_i,x,xL), d2_sigma(t, t_i, x, x,1,uhatmat)))

        d1_d2_sigma_block = np.block([
            [sigma(t_0,t_0,x[1:-1],x[1:-1]), sigma(t_0,t_i,x[1:-1],x0),sigma(t_0,t_i,x[1:-1],xL),d2_sigma(t_0, t_i, x[1:-1],x,1,uhatmat0)],
            [sigma(t_i,t_0,x0,x[1:-1]),sigma(t_i,t_i,x0,x0),sigma(t_i,t_i,x0,xL),d2_sigma(t_i, t_i, x0,x,1,uhatmatix0)],
            [sigma(t_i,t_0,xL,x[1:-1]), sigma(t_i, t_i, xL, x0), sigma(t_i, t_i, xL, xL), d2_sigma(t_i, t_i, xL, x, 1, uhatmatix0)],
            [d1_sigma(t_i, t_0, x,x[1:-1],1,uhatmat0.T),d1_sigma(t_i, t_i, x,x0,1,uhatmatix0.T),d1_sigma(t_i, t_i, x,xL,1,uhatmatix0.T),d1_d2_sigma(t_i, t_i, x,x,1,uhatmati,uhatmati.T)]
        ])

        d1_d2_sigma_blocknew = d1_d2_sigma_block + 0.0000001 * np.eye(d1_d2_sigma_block.shape[0])

        d1_d2_sigma_blocknew_inverse=np.linalg.inv(d1_d2_sigma_blocknew)

        muiter = np.reshape(muprior, (muprior.size, 1)) + np.matmul(np.matmul(d2_sigma_block, d1_d2_sigma_blocknew_inverse),np.reshape(datablock, (datablock.size, 1)))

        Sigiter = sigma(t,t,x,x) - np.matmul(np.matmul(d2_sigma_block, d1_d2_sigma_blocknew_inverse), d2_sigma_block.T)

        muiterrhofull = np.copy(muiter[:, 0])
        muiterrhofull[0:i * M] = muiterrho
        muitermat = np.array([muiterrhofull, ] * N * M)
        uhatmat0allt = np.array([muiterrhofull, ] * (x.size - 2))
        uhatmatix0allt = np.array([muiterrhofull, ] * t_i.size)
        uhatmatiallt = np.array([muiterrhofull, ] * (t_i.size) * x.size)

        d1_d2_sigma_block_left = np.column_stack((d1_sigma(t, t_0, x, x[1:-1], 1, uhatmat0allt.T), d1_sigma(t, t_i, x, x0, 1, uhatmatix0allt.T),d1_sigma(t, t_i, x, xL, 1, uhatmatix0allt.T), d1_d2_sigma(t, t_i, x, x, 1,uhatmat,uhatmatiallt.T)))

        dmuiter = np.reshape(dmuprior, (dmuprior.size, 1)) + np.matmul(np.matmul(d1_d2_sigma_block_left, d1_d2_sigma_blocknew_inverse),np.reshape(datablock, (datablock.size, 1)))

        ddSigiter = d1_d2_sigma(t,t,x,x,1,muitermat,muitermat.T) - np.matmul(np.matmul(d1_d2_sigma_block_left, d1_d2_sigma_blocknew_inverse),d1_d2_sigma_block_left.T)

        if i < N:
            # appends the mean at time t_{i+1} from the current mean (estimated at time t_i) to linearise the differential operator at the next timepoint t_{i+1}
            muiterrho = np.append(muiterrho, muiter[i*M:i*M+M,0])


    print('end loop')

    Sigiter = Sigiter *amppara/N

    truesolution=g_x
    truegradient = 0*x
    for i in range(1,N):
        truesolution=np.append(truesolution,(2*coef*a*d*np.exp(-coef*d**2*t[i])*np.sin(d*x))/(b+a*np.exp(-coef*d**2*t[i])*np.cos(d*x)))
        truegradient = np.append(truegradient, 0*x)

    maxgradienterror = np.max(np.abs(truegradient - dmuiter[:, 0]))
    mse = np.sqrt(np.mean((truesolution - muiter[:, 0]) ** 2))
    msegradient = np.sqrt(np.mean((truegradient - dmuiter[:, 0]) ** 2))

    #calculate error and zscore
    maxerror=np.max(np.abs(truesolution-muiter[:,0]))

    zscoretop=np.abs(truesolution-muiter[:,0])
    zscoretop=zscoretop[M:]
    zscoretopnonzero = [x for i, x in enumerate(zscoretop) if (i%M !=0) and (i%M !=M-1)]
    zscorebottom=np.diag(Sigiter)
    zscorebottom=zscorebottom[M:]
    zscorebottomnonzero = [x for i, x in enumerate(zscorebottom) if (i%M !=0) and (i%M !=M-1)]

    zscore = np.max(zscoretopnonzero / np.sqrt(zscorebottomnonzero))

    results = [N,M,maxerror,zscore, mse]

    #save results to txt file
    import csv
    with open("Burgernoforcerationalresults.txt", "a") as f:
        wr = csv.writer(f)
        wr.writerow(results)
    f.close()

    return(maxerror)

for N in Nset: #[5,9,17,33,65,129]
    for M in Mset: #[5,9,17,33,65,129]
        Burgerrational(N,M)
