import numpy as np

#Porous medium equation

#alternate linearisation, linearises double derivative rather than u

#Values of N and M you would like to run the method over.
Nset=[5,9,17,33,65,129]
Mset=[5,9,17,33,65,129]

seed=10
np.random.seed(seed)

coef=2 #coefficient of the double x derivative term in the heat equation
pi=np.pi


def Porousalt(N,M):

    # hyper-parameter(s)
    sig=1/np.sqrt(3)
    sig1=2/np.sqrt(5)
    alpha=1
    alpha1=1

    # Kernel functions

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

    def d1x_sigma(t1,t2,r1,r2,uhat,uhatgrad):
        return -coef*np.kron(k(t1,t2), c(r1, r2))*uhat-coef*np.kron(k(t1,t2), dr1_c(r1, r2))*uhatgrad

    def d2x_sigma(t1,t2,r1,r2,uhat,uhatgrad):
        return -coef*np.kron(k(t1,t2), c(r1, r2))*uhat-coef*np.kron(k(t1,t2), dr2_c(r1, r2))*uhatgrad

    def d1x_d2x_sigma(t1,t2,r1,r2,uhat1,uhat2,uhat1grad,uhat2grad):
        return (coef**2)*np.kron(k(t1,t2), c(r1, r2))*uhat1*uhat2+(coef**2)*np.kron(k(t1,t2), dr1r2_c(r1, r2))*uhat1grad*uhat2grad+(coef**2)*np.kron(k(t1,t2), dr1_c(r1, r2))*uhat2grad*uhat1+(coef**2)*np.kron(k(t1,t2), dr2_c(r1, r2))*uhat1grad*uhat2

    def d1t_sigma(t1,t2,r1,r2,uhat):
        return np.kron(dr1_k(t1,t2),c(r1,r2))

    def d2t_sigma(t1,t2,r1,r2,uhat):
        return np.kron(dr2_k(t1,t2),c(r1,r2))

    def d1t_d2t_sigma(t1,t2,r1,r2,uhat):
        return np.kron(dr1r2_k(t1,t2),c(r1,r2))

    def d1t_d2x_sigma(t1,t2,r1,r2,uhat,uhatgrad):
        return -coef*np.kron(dr1_k(t1,t2),c(r1, r2))*uhat-coef*np.kron(dr1_k(t1,t2),dr2_c(r1, r2))*uhatgrad

    def d1x_d2t_sigma(t1,t2,r1,r2,uhat,uhatgrad):
        return -coef*np.kron(dr2_k(t1,t2),c(r1, r2))*uhat-coef*np.kron(dr2_k(t1,t2),dr1_c(r1, r2))*uhatgrad

    def d1_sigma(t1,t2,r1,r2,uhat1,uhat2,uhat1grad):
        return d1t_sigma(t1,t2,r1,r2,uhat1)+d1x_sigma(t1,t2,r1,r2,uhat2,uhat1grad)

    def d2_sigma(t1,t2,r1,r2,uhat1,uhat2,uhat1grad):
        return d2t_sigma(t1,t2,r1,r2,uhat1)+d2x_sigma(t1,t2,r1,r2,uhat2,uhat1grad)

    def d1_d2_sigma(t1,t2,r1,r2,uhat1,uhat2,uhat3,uhat1grad,uhat2grad):
        return d1t_d2t_sigma(t1,t2,r1,r2,uhat1)+d1t_d2x_sigma(t1,t2,r1,r2,uhat2,uhat1grad)+d1x_d2t_sigma(t1,t2,r1,r2,uhat3,uhat2grad)+d1x_d2x_sigma(t1,t2,r1,r2,uhat2,uhat3,uhat1grad,uhat2grad)


    def dr1x_sigma(t1,t2,r1,r2):
        return np.kron(k(t1,t2), dr1_c(r1, r2))

    def dr1x_d2_sigma(t1,t2,r1,r2,uhat2,uhat1grad):
        return np.kron(dr2_k(t1,t2),dr1_c(r1,r2))-coef*np.kron(k(t1,t2),dr1_c(r1, r2))*uhat2-coef*np.kron(k(t1,t2),dr1r2_c(r1, r2))*uhat1grad

    def dr1r1x_sigma(t1,t2,r1,r2):
        return np.kron(k(t1,t2), dr1r1_c(r1, r2))

    def dr1r1x_d2_sigma(t1,t2,r1,r2,uhat2,uhat1grad):
        return np.kron(dr2_k(t1,t2),dr1r1_c(r1,r2))-coef*np.kron(k(t1,t2),dr1r1_c(r1, r2))*uhat2-coef*np.kron(k(t1,t2),dr1r1r2_c(r1, r2))*uhat1grad

    #PDE definition
    t = np.linspace(2.00, 10.00, N)
    x = np.linspace(-10, 10, M)

    b = 1

    x0 = min(x)
    xL = max(x)
    g_x = (2) ** (-1 / 3) * np.maximum(0, b - x * x / (12 * 2 ** (2 / 3)))

    def F(t, y, x): return 0 * t

    #priors and initial variables
    muprior=np.array([0.0]*N*M)
    muiter=np.array([0.0]*N*M)
    dmuprior=np.array([0.0]*N*M)
    dmuiter=np.array([0.0]*N*M)

    t_0 = np.atleast_1d(t[0])

    # Gaussian Process conditioning of the initial condition of PDE
    muiter = muprior + sigma(t, t_0, x, x[1:-1]) @ np.linalg.inv(sigma(t_0, t_0, x[1:-1], x[1:-1])) @ g_x[1:-1]
    dxmuiter = dmuprior + dr1x_sigma(t, t_0, x, x[1:-1]) @ np.linalg.inv(sigma(t_0, t_0, x[1:-1], x[1:-1])) @ g_x[1:-1]
    dxxmuiter = dmuprior + dr1r1x_sigma(t, t_0, x, x[1:-1]) @ np.linalg.inv(sigma(t_0, t_0, x[1:-1], x[1:-1])) @ g_x[1:-1]
    Sigiter = sigma(t, t, x, x) - sigma(t, t_0, x, x[1:-1]) @ np.linalg.inv(sigma(t_0, t_0, x[1:-1], x[1:-1])) @ sigma(t_0, t, x[1:-1], x)

    muiterrho = np.atleast_1d(muiter[0:M])
    # mean of du(t_{0},x)/dx, d^2u(t_{0},x)/dx^2 from the current prior mean to linearise the differential operator at the next timepoint t_0
    dxmuiterrho=np.atleast_1d(dxmuiter[0:M])
    dxxmuiterrho = np.atleast_1d(dxxmuiter[0:M])
    #muiterrhofull = np.copy(muiter)
    #muiterrhofull[0:M] = muiterrho
    # same as above but also keeps the 'best guess' for all remaining t, using the current mean.
    dxmuiterrhofull = np.copy(dxmuiter)
    dxmuiterrhofull[0:M] = dxmuiterrho
    dxxmuiterrhofull = np.copy(dxxmuiter)
    dxxmuiterrhofull[0:M] = dxxmuiterrho

    #muitermat = np.array([muiterrhofull, ] * N * M)
    dxmuitermat = np.array([dxmuiterrhofull, ] * N * M)
    dxxmuitermat = np.array([dxxmuiterrhofull, ] * N * M)

    #uhatmat0allt = np.array([muiterrhofull, ] * (x.size - 2))
    uhatmat0alltgrad = np.array([dxmuiterrhofull, ] * (x.size - 2))
    uhatmat0alltgradgrad = np.array([dxxmuiterrhofull, ] * (x.size - 2))

    dmuiter = dmuprior + d1_sigma(t, t_0, x, x[1:-1], 1, uhatmat0alltgradgrad.T, uhatmat0alltgrad.T) @ np.linalg.inv(sigma(t_0, t_0, x[1:-1], x[1:-1])) @ g_x[1:-1]

    ddSigiter= d1_d2_sigma(t,t,x,x,1,dxxmuitermat,dxxmuitermat.T,dxmuitermat,dxmuitermat.T)-d1_sigma(t,t_0,x,x[1:-1],1,uhatmat0alltgradgrad.T, uhatmat0alltgrad.T) @ np.linalg.inv(sigma(t_0,t_0,x[1:-1],x[1:-1])) @ d2_sigma(t_0,t,x[1:-1],x,1,uhatmat0alltgradgrad, uhatmat0alltgrad)


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
            #ddSigiter = d1_d2_sigma(t, t, x, x, 1, 0, 0, 0, 0)
            muiterrho0only=np.reshape(muiterrho,(muiterrho.size,1))
            dmuiter0only = np.reshape(dmuiter, (dmuiter.size, 1))
            newblock=ddSigiter[0:M, 0:M]
            amppara=amppara+np.matmul(np.matmul(np.reshape(F(t[i-1], muiterrho0only[(i-1)*M:i * M], x) - dmuiter0only[(i-1)*M:i * M,0], (1,x.size)),np.linalg.inv(newblock)),np.reshape(F(t[i-1], muiterrho0only[(i-1)*M:i * M], x) - dmuiter0only[(i-1)*M:i * M,0], (x.size,1)))
        else:
            newblock=ddSigiter[(i - 1) * M:i * M, (i - 1) * M:i * M]
            amppara=amppara+np.matmul(np.matmul(np.reshape(F(t[i-1], muiterrho[(i-1)*M:i * M], x) - dmuiter[(i-1)*M:i * M,0], (1,x.size)),np.linalg.inv(newblock)),np.reshape(F(t[i-1], muiterrho[(i-1)*M:i * M], x) - dmuiter[(i-1)*M:i * M,0], (x.size,1)))


        #uhatmat = np.array([muiterrho, ] * t.size * x.size)
        uhatmatgrad = np.array([dxmuiterrho, ] * t.size * x.size)
        uhatmatgradgrad = np.array([dxxmuiterrho, ] * t.size * x.size)

        #uhatmat0 = np.array([muiterrho, ] * (x.size-2))
        #uhatmatix0= np.array([muiterrho, ] * t_i.size)
        #uhatmati = np.array([muiterrho, ] * (t_i.size)*x.size)
        #uhatfull=np.multiply.outer(muiterrho,muiterrho)

        uhatmat0grad = np.array([dxmuiterrho, ] * (x.size - 2))
        uhatmatix0grad = np.array([dxmuiterrho, ] * t_i.size)
        uhatmatigrad = np.array([dxmuiterrho, ] * (t_i.size) * x.size)

        uhatmat0gradgrad = np.array([dxxmuiterrho, ] * (x.size - 2))
        uhatmatix0gradgrad = np.array([dxxmuiterrho, ] * t_i.size)
        uhatmatigradgrad = np.array([dxxmuiterrho, ] * (t_i.size) * x.size)


        d2_sigma_block = np.column_stack((sigma(t,t_0,x,x[1:-1]), sigma(t,t_i,x,x0),sigma(t,t_i,x,xL), d2_sigma(t, t_i, x, x,1,uhatmatgradgrad,uhatmatgrad)))

        d1_d2_sigma_block = np.block([
            [sigma(t_0,t_0,x[1:-1],x[1:-1]), sigma(t_0,t_i,x[1:-1],x0),sigma(t_0,t_i,x[1:-1],xL),d2_sigma(t_0, t_i, x[1:-1],x,1,uhatmat0gradgrad,uhatmat0grad)],
            [sigma(t_i,t_0,x0,x[1:-1]),sigma(t_i,t_i,x0,x0),sigma(t_i,t_i,x0,xL),d2_sigma(t_i, t_i, x0,x,1,uhatmatix0gradgrad,uhatmatix0grad)],
            [sigma(t_i,t_0,xL,x[1:-1]), sigma(t_i, t_i, xL, x0), sigma(t_i, t_i, xL, xL), d2_sigma(t_i, t_i, xL, x, 1, uhatmatix0gradgrad,uhatmatix0grad)],
            [d1_sigma(t_i, t_0, x,x[1:-1],1,uhatmat0gradgrad.T,uhatmat0grad.T),d1_sigma(t_i, t_i, x,x0,1,uhatmatix0gradgrad.T,uhatmatix0grad.T),d1_sigma(t_i, t_i, x,xL,1,uhatmatix0gradgrad.T,uhatmatix0grad.T),d1_d2_sigma(t_i, t_i, x,x,1,uhatmatigradgrad,uhatmatigradgrad.T,uhatmatigrad,uhatmatigrad.T)]
        ])


        d1_d2_sigma_block_inverse=np.linalg.inv(d1_d2_sigma_block)

        dr1x_d2_sigma_block_left = np.column_stack((dr1x_sigma(t, t_0, x, x[1:-1]), dr1x_sigma(t, t_i, x, x0),dr1x_sigma(t, t_i, x, xL), dr1x_d2_sigma(t, t_i, x, x, uhatmatgradgrad,uhatmatgrad)))

        dr1r1x_d2_sigma_block_left = np.column_stack((dr1r1x_sigma(t, t_0, x, x[1:-1]), dr1r1x_sigma(t, t_i, x, x0),dr1r1x_sigma(t, t_i, x, xL), dr1r1x_d2_sigma(t, t_i, x, x, uhatmatgradgrad,uhatmatgrad)))

        muiter = np.reshape(muprior, (muprior.size, 1)) + np.matmul(np.matmul(d2_sigma_block, d1_d2_sigma_block_inverse),np.reshape(datablock, (datablock.size, 1)))

        dxmuiter=np.reshape(dmuprior, (dmuprior.size, 1)) + np.matmul(np.matmul(dr1x_d2_sigma_block_left, d1_d2_sigma_block_inverse),np.reshape(datablock, (datablock.size, 1)))

        dxxmuiter = np.reshape(dmuprior, (dmuprior.size, 1)) + np.matmul(np.matmul(dr1r1x_d2_sigma_block_left, d1_d2_sigma_block_inverse),np.reshape(datablock, (datablock.size, 1)))

        Sigiter = sigma(t,t,x,x) - np.matmul(np.matmul(d2_sigma_block, d1_d2_sigma_block_inverse), d2_sigma_block.T)

        #muiterrhofull = np.copy(muiter[:, 0])
        dxmuiterrhofull = np.copy(dxmuiter[:, 0])
        dxxmuiterrhofull = np.copy(dxxmuiter[:, 0])

        #muiterrhofull[0:i * M] = muiterrho
        dxmuiterrhofull[0:i * M] = dxmuiterrho
        dxxmuiterrhofull[0:i * M] = dxxmuiterrho

        #muitermat = np.array([muiterrhofull, ] * N * M)
        dxmuitermat = np.array([dxmuiterrhofull, ] * N * M)
        dxxmuitermat = np.array([dxxmuiterrhofull, ] * N * M)

        #uhatmat0allt = np.array([muiterrhofull, ] * (x.size - 2))
        #uhatmatix0allt = np.array([muiterrhofull, ] * t_i.size)
        #uhatmatiallt = np.array([muiterrhofull, ] * (t_i.size) * x.size)

        uhatmat0alltgrad = np.array([dxmuiterrhofull, ] * (x.size - 2))
        uhatmatix0alltgrad = np.array([dxmuiterrhofull, ] * t_i.size)
        uhatmatialltgrad = np.array([dxmuiterrhofull, ] * (t_i.size) * x.size)

        uhatmat0alltgradgrad = np.array([dxxmuiterrhofull, ] * (x.size - 2))
        uhatmatix0alltgradgrad = np.array([dxxmuiterrhofull, ] * t_i.size)
        uhatmatialltgradgrad = np.array([dxxmuiterrhofull, ] * (t_i.size) * x.size)

        d1_d2_sigma_block_left = np.column_stack((d1_sigma(t, t_0, x, x[1:-1], 1, uhatmat0alltgradgrad.T,uhatmat0alltgrad.T), d1_sigma(t, t_i, x, x0, 1, uhatmatix0alltgradgrad.T,uhatmatix0alltgrad.T),d1_sigma(t, t_i, x, xL, 1, uhatmatix0alltgradgrad.T,uhatmatix0alltgrad.T), d1_d2_sigma(t, t_i, x, x, 1,uhatmatgradgrad,uhatmatialltgradgrad.T,uhatmatgrad,uhatmatialltgrad.T)))

        dmuiter = np.reshape(dmuprior, (dmuprior.size, 1)) + np.matmul(np.matmul(d1_d2_sigma_block_left, d1_d2_sigma_block_inverse),np.reshape(datablock, (datablock.size, 1)))

        ddSigiter = d1_d2_sigma(t,t,x,x,1,dxxmuitermat,dxxmuitermat.T,dxmuitermat,dxmuitermat.T) - np.matmul(np.matmul(d1_d2_sigma_block_left, d1_d2_sigma_block_inverse),d1_d2_sigma_block_left.T)

        if i < N:
            # appends the mean at time t_{i+1} from the current mean (estimated at time t_i) to linearise the differential operator at the next timepoint t_{i+1}
            muiterrho = np.append(muiterrho, muiter[i*M:i*M+M,0])
            #dmuiterrho = np.append(dmuiterrho, dmuiter[points[i] * M:points[i] * M + M, 0])
            dxmuiterrho = np.append(dxmuiterrho, dxmuiter[i * M:i * M + M, 0])
            dxxmuiterrho = np.append(dxxmuiterrho, dxxmuiter[i * M:i * M + M, 0])
    print('end loop')

    Sigiter = Sigiter *amppara/N

    truesolution=g_x
    truegradient = 0 * x
    for i in range(1,N):
        truesolution=np.append(truesolution,(t[i])**(-1/3)*np.maximum(0,b-x*x/(12*t[i]**(2/3))))
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

    # save results to txt file
    import csv
    with open("Porousaltresults.txt", "a") as f:
        wr = csv.writer(f)
        wr.writerow(results)
    f.close()

    return(maxerror)

for N in Nset:
    for M in Mset:
        Porousalt(N,M)
