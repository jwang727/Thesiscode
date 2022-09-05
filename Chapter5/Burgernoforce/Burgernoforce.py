import numpy as np

#Homogeneous Burger's Equation, Matern kernel

#Values of N and M you would like to run the method over.
Nset=[5,9,17,33,65,129]
Mset=[5,9,17,33,65,129]

seed=10
np.random.seed(seed)

coef=0.02 #coefficient of the double x derivative term in the heat equation
pi=np.pi


def Burger(N,M):

    # hyper-parameter(s)
    sig=6/np.sqrt(3)
    sig1=3/np.sqrt(5)
    alpha=1
    alpha1=1

    #Kernel functions

    def k (r1,r2):
        r1=np.atleast_1d(r1)
        r2=np.atleast_1d(r2)

        return alpha * np.multiply(1 + abs(np.subtract.outer(r1, r2)) / sig, np.exp(-abs(np.subtract.outer(r1, r2)) / sig))

    def dr1_k (r1,r2):
        r1=np.atleast_1d(r1)
        r2=np.atleast_1d(r2)

        return -alpha * (np.subtract.outer(r1, r2) / sig ** 2) * np.exp(-abs(np.subtract.outer(r1, r2)) / sig)

    def dr2_k (r1,r2):
        r1 = np.atleast_1d(r1)
        r2 = np.atleast_1d(r2)

        return dr1_k(r2,r1).T

    def dr1r2_k (r1,r2):
        r1 = np.atleast_1d(r1)
        r2 = np.atleast_1d(r2)

        return alpha * (1 / sig ** 2 - (1 / sig ** 3) * abs(np.subtract.outer(r1, r2))) * np.exp(-abs(np.subtract.outer(r1, r2)) / sig)

    def dr1r1_k (r1,r2):
        r1 = np.atleast_1d(r1)
        r2 = np.atleast_1d(r2)

        return alpha * (-1 / sig ** 2 + (1 / sig ** 3) * abs(np.subtract.outer(r1, r2))) * np.exp(-abs(np.subtract.outer(r1, r2)) / sig)

    def dr2r2_k (r1,r2):
        r1 = np.atleast_1d(r1)
        r2 = np.atleast_1d(r2)

        return dr1r1_k (r1,r2)

    def dr1r1r2_k(r1, r2):
        r1 = np.atleast_1d(r1)
        r2 = np.atleast_1d(r2)

        return alpha*(-(1/(sig**4))*np.subtract.outer(r1,r2)+(np.subtract.outer(r1,r2))*abs(np.subtract.outer(r1,r2))/(3*sig**5))*np.exp(-abs(np.subtract.outer(r1,r2))/sig)

    def dr1r2r2_k (r1,r2):
        r1 = np.atleast_1d(r1)
        r2 = np.atleast_1d(r2)

        return dr1r1r2_k(r2, r1).T

    def dr1r1r2r2_k (r1,r2):
        r1 = np.atleast_1d(r1)
        r2 = np.atleast_1d(r2)

        return alpha*(1/(sig**4)-5*abs(np.subtract.outer(r1,r2))/(3*sig**5)+np.subtract.outer(r1,r2)**2/(3*sig**6))*np.exp(-abs(np.subtract.outer(r1,r2))/sig)

    def c (r1,r2):
        r1=np.atleast_1d(r1)
        r2=np.atleast_1d(r2)

        return alpha1*(1+abs(np.subtract.outer(r1,r2))/sig1+np.subtract.outer(r1,r2)**2/(3*sig1**2))*np.exp(-abs(np.subtract.outer(r1,r2))/sig1)


    def dr1_c (r1,r2):
        r1=np.atleast_1d(r1)
        r2=np.atleast_1d(r2)

        return alpha1*(-(1/(3*sig1**2))*np.subtract.outer(r1,r2)-(np.subtract.outer(r1,r2))*abs(np.subtract.outer(r1,r2))/(3*sig1**3))*np.exp(-abs(np.subtract.outer(r1,r2))/sig1)

    def dr2_c (r1,r2):
        r1 = np.atleast_1d(r1)
        r2 = np.atleast_1d(r2)

        return dr1_c(r2,r1).T

    def dr1r2_c (r1,r2):
        r1 = np.atleast_1d(r1)
        r2 = np.atleast_1d(r2)

        return alpha1*(1/(3*sig1**2)+abs(np.subtract.outer(r1,r2))/(3*sig1**3)-np.subtract.outer(r1,r2)**2/(3*sig1**4))*np.exp(-abs(np.subtract.outer(r1,r2))/sig1)

    def dr1r1_c (r1,r2):
        r1 = np.atleast_1d(r1)
        r2 = np.atleast_1d(r2)

        return alpha1*(-1/(3*sig1**2)-abs(np.subtract.outer(r1,r2))/(3*sig1**3)+np.subtract.outer(r1,r2)**2/(3*sig1**4))*np.exp(-abs(np.subtract.outer(r1,r2))/sig1)

    def dr2r2_c (r1,r2):
        r1 = np.atleast_1d(r1)
        r2 = np.atleast_1d(r2)

        return dr1r1_c (r1,r2)

    def dr1r1r2_c(r1, r2):
        r1 = np.atleast_1d(r1)
        r2 = np.atleast_1d(r2)

        return alpha1*(-(1/(sig1**4))*np.subtract.outer(r1,r2)+(np.subtract.outer(r1,r2))*abs(np.subtract.outer(r1,r2))/(3*sig1**5))*np.exp(-abs(np.subtract.outer(r1,r2))/sig1)

    def dr1r2r2_c (r1,r2):
        r1 = np.atleast_1d(r1)
        r2 = np.atleast_1d(r2)

        return dr1r1r2_c(r2, r1).T

    def dr1r1r2r2_c (r1,r2):
        r1 = np.atleast_1d(r1)
        r2 = np.atleast_1d(r2)

        return alpha1*(1/(sig1**4)-5*abs(np.subtract.outer(r1,r2))/(3*sig1**5)+np.subtract.outer(r1,r2)**2/(3*sig1**6))*np.exp(-abs(np.subtract.outer(r1,r2))/sig1)

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


    #PDE definition
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
    muprior=np.array([0.0] * N * M)
    muiter=np.array([0.0] * N * M)
    dmuprior=np.array([0.0] * N * M)
    dmuiter=np.array([0.0] * N * M)

    t_0 = np.atleast_1d(t[0])

    #Gaussian Process conditioning of the initial condition of PDE
    muiter = muprior + sigma(t, t_0, x, x[1:-1]) @ np.linalg.inv(sigma(t_0, t_0, x[1:-1], x[1:-1])) @ g_x[1:-1]
    Sigiter = sigma(t, t, x, x) - sigma(t, t_0, x, x[1:-1]) @ np.linalg.inv(sigma(t_0, t_0, x[1:-1], x[1:-1])) @ sigma(t_0, t, x[1:-1], x)

    #mean of u(t_{0},x) from the current prior mean to linearise the differential operator at the next timepoint t_0
    muiterrho = np.atleast_1d(muiter[0:M])
    #same as above but also keeps the 'best guess' for all remaining t, using the current mean.
    muiterrhofull = np.copy(muiter)
    muiterrhofull[0:M] = muiterrho

    muitermat = np.array([muiterrhofull, ] * N * M)

    uhatmat0allt = np.array([muiterrhofull, ] * (x.size - 2))

    dmuiter = dmuprior + d1_sigma(t, t_0, x, x[1:-1], 1, uhatmat0allt.T) @ np.linalg.inv(sigma(t_0, t_0, x[1:-1], x[1:-1])) @ g_x[1:-1]

    ddSigiter= d1_d2_sigma(t,t,x,x,1,muitermat,muitermat.T)-d1_sigma(t,t_0,x,x[1:-1],1,uhatmat0allt.T) @ np.linalg.inv(sigma(t_0,t_0,x[1:-1],x[1:-1])) @ d2_sigma(t_0,t,x[1:-1],x,1,uhatmat0allt)


    stop=N

    amppara=0

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
            newblock=ddSigiter[0:M, 0:M]
            amppara=amppara+np.matmul(np.matmul(np.reshape(F(t[i-1], muiterrho0only[(i-1)*M:i * M], x) - dmuiter0only[(i-1)*M:i * M,0], (1,x.size)),np.linalg.inv(newblock)),np.reshape(F(t[i-1], muiterrho0only[(i-1)*M:i * M], x) - dmuiter0only[(i-1)*M:i * M,0], (x.size,1)))
        else:
            newblock=ddSigiter[(i - 1) * M:i * M, (i - 1) * M:i * M]
            amppara=amppara+np.matmul(np.matmul(np.reshape(F(t[i-1], muiterrho[(i-1)*M:i * M], x) - dmuiter[(i-1)*M:i * M,0], (1,x.size)),np.linalg.inv(newblock)),np.reshape(F(t[i-1], muiterrho[(i-1)*M:i * M], x) - dmuiter[(i-1)*M:i * M,0], (x.size,1)))

        print(amppara)

        uhatmat = np.array([muiterrho, ] * t.size * x.size)
        uhatmat0 = np.array([muiterrho, ] * (x.size-2)) #(x.size-2) rows of muiterrho
        uhatmatix0= np.array([muiterrho, ] * t_i.size)
        uhatmati = np.array([muiterrho, ] * (t_i.size)*x.size)


        d2_sigma_block = np.column_stack((sigma(t,t_0,x,x[1:-1]), sigma(t,t_i,x,x0),sigma(t,t_i,x,xL), d2_sigma(t, t_i, x, x,1,uhatmat)))

        d1_d2_sigma_block = np.block([
            [sigma(t_0,t_0,x[1:-1],x[1:-1]), sigma(t_0,t_i,x[1:-1],x0),sigma(t_0,t_i,x[1:-1],xL),d2_sigma(t_0, t_i, x[1:-1],x,1,uhatmat0)],
            [sigma(t_i,t_0,x0,x[1:-1]),sigma(t_i,t_i,x0,x0),sigma(t_i,t_i,x0,xL),d2_sigma(t_i, t_i, x0,x,1,uhatmatix0)],
            [sigma(t_i,t_0,xL,x[1:-1]), sigma(t_i, t_i, xL, x0), sigma(t_i, t_i, xL, xL), d2_sigma(t_i, t_i, xL, x, 1, uhatmatix0)],
            [d1_sigma(t_i, t_0, x,x[1:-1],1,uhatmat0.T),d1_sigma(t_i, t_i, x,x0,1,uhatmatix0.T),d1_sigma(t_i, t_i, x,xL,1,uhatmatix0.T),d1_d2_sigma(t_i, t_i, x,x,1,uhatmati,uhatmati.T)]
        ])

        if N>60 and M>100:
            print('N=65,129, M=129')
            d1_d2_sigma_block=d1_d2_sigma_block+0.0000001 * np.eye(d1_d2_sigma_block.shape[0])

        d1_d2_sigma_block_inverse=np.linalg.inv(d1_d2_sigma_block)

        muiter = np.reshape(muprior, (muprior.size, 1)) + np.matmul(np.matmul(d2_sigma_block, d1_d2_sigma_block_inverse),np.reshape(datablock, (datablock.size, 1)))

        Sigiter = sigma(t,t,x,x) - np.matmul(np.matmul(d2_sigma_block, d1_d2_sigma_block_inverse), d2_sigma_block.T)

        muiterrhofull = np.copy(muiter[:, 0])
        muiterrhofull[0:i * M] = muiterrho

        muitermat = np.array([muiterrhofull, ] * N * M)

        uhatmat0allt = np.array([muiterrhofull, ] * (x.size - 2))
        uhatmatix0allt = np.array([muiterrhofull, ] * t_i.size)
        uhatmatiallt = np.array([muiterrhofull, ] * (t_i.size) * x.size)

        d1_d2_sigma_block_left = np.column_stack((d1_sigma(t, t_0, x, x[1:-1], 1, uhatmat0allt.T), d1_sigma(t, t_i, x, x0, 1, uhatmatix0allt.T),d1_sigma(t, t_i, x, xL, 1, uhatmatix0allt.T), d1_d2_sigma(t, t_i, x, x, 1,uhatmat,uhatmatiallt.T)))

        dmuiter = np.reshape(dmuprior, (dmuprior.size, 1)) + np.matmul(np.matmul(d1_d2_sigma_block_left, d1_d2_sigma_block_inverse),np.reshape(datablock, (datablock.size, 1)))

        ddSigiter = d1_d2_sigma(t,t,x,x,1,muitermat,muitermat.T) - np.matmul(np.matmul(d1_d2_sigma_block_left, d1_d2_sigma_block_inverse),d1_d2_sigma_block_left.T)

        if i < N:
            # appends the mean at time t_{i+1} from the current mean (estimated at time t_i) to linearise the differential operator at the next timepoint t_{i+1}
            muiterrho = np.append(muiterrho, muiter[i*M:i*M+M,0])

    print('end loop')

    Sigiter = Sigiter *amppara/N

    truesolution=g_x
    truegradient = 0*x
    for i in range(1,N):
        truesolution=np.append(truesolution,(2*coef*a*d*np.exp(-coef*d**2*t[i])*np.sin(d*x))/(b+a*np.exp(-coef*d**2*t[i])*np.cos(d*x)))
        truegradient = np.append(truegradient, 0 * x)

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

    results = [N,M,maxerror,zscore,mse]

    #save results to txt file
    import csv
    with open("Burgernoforceresults.txt", "a") as f:
        wr = csv.writer(f)
        wr.writerow(results)
    f.close()

    np.savetxt('surfaceplotmean.txt', muiter[:, 0], delimiter=',', fmt='%s')
    np.savetxt('surfaceplotcov.txt', Sigiter, delimiter=',', fmt='%s')


    return(maxerror)

for N in Nset: #[5,9,17,33,65,129]
    for M in Mset: #[5,9,17,33,65,129]
        Burger(N,M)

