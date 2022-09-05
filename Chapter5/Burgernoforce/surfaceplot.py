import numpy as np
import matplotlib.pyplot as plt


seed=123
np.random.seed(seed)

plt.rcParams.update({
    "text.usetex": True,})

mu=np.loadtxt('surfaceplotmean.txt', delimiter=',')
Sig=np.loadtxt('surfaceplotcov.txt', delimiter=',')

#Sig=Sig*(0.9475135677916963/0.9860206401049942)**2

pi=np.pi

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

N=65
t=np.linspace(0.00,30.00,N)

M=65
x=np.linspace(0.00,2*pi,M)

coef=0.02
a = 1
b = 2
d = 1

def truesol(t, x):
    return (2*coef*a*d*np.exp(-coef*d**2*t)*np.sin(d*x))/(b+a*np.exp(-coef*d**2*t)*np.cos(d*x))

T, X = np.meshgrid(t, x)
zs = np.array(truesol(np.ravel(T), np.ravel(X)))
Z = zs.reshape(T.shape)

samples=np.random.multivariate_normal(mean=mu, cov=Sig, size=100, tol=1e-6)
print('check samples shape')
print(samples.shape)
lowquantall = np.quantile(samples, 0.025, axis=0)
highquantall = np.quantile(samples, 0.975, axis=0)

lowquantmat = np.reshape(lowquantall[0*M:0 * M + M], (M, 1))
highquantmat = np.reshape(highquantall[0*M:0 * M + M], (M, 1))
meanmat=np.reshape(mu[0 * M:0 * M + M], (M, 1))
stdmat=np.reshape(np.diagonal(Sig[0*M:0*M+M,0*M:0*M+M]), (M, 1))

for timevar in range(1,N):
    lowquant=lowquantall[timevar*M:timevar * M + M]
    lowquant = np.reshape(lowquant, (M, 1))
    highquant = highquantall[timevar * M:timevar * M + M]
    highquant = np.reshape(highquant, (M, 1))
    lowquantmat = np.hstack((lowquantmat, lowquant))
    highquantmat = np.hstack((highquantmat, highquant))
    #mean at time timevar put into a column vector, and then stacked horizontally
    #meaning column is time, row is position
    meanvec = np.reshape(mu[timevar * M:timevar * M + M], (M, 1))
    meanmat = np.hstack((meanmat, meanvec))
    stdvec = np.reshape(np.diagonal(Sig[timevar * M:timevar * M + M, timevar * M:timevar * M + M]), (M, 1))
    stdmat = np.hstack((stdmat, stdvec))

Tstart=np.repeat(0.00,N)
Tend=np.repeat(30.00,N)

ax.plot(Tstart,x,Z[:,0],color='blue',alpha=1)
ax.plot(Tend,x,Z[:,-1],color='blue',alpha=1)
ax.plot(Tstart,x,meanmat[:,0],color='red',alpha=1)
ax.plot(Tend,x,meanmat[:,-1],color='red',alpha=1)

ax.plot(Tstart,x,lowquantmat[:,0],color='red',alpha=1,linestyle='dashed')
ax.plot(Tend,x,lowquantmat[:,-1],color='red',alpha=1,linestyle='dashed')
ax.plot(Tstart,x,highquantmat[:,0],color='red',alpha=1,linestyle='dashed')
ax.plot(Tend,x,highquantmat[:,-1],color='red',alpha=1,linestyle='dashed')

#ax.plot(Tstart,x,Z[:,0],color='orange',alpha=1)
#ax.plot(Tend,x,meanmat[:,-1],color='orange',alpha=1)

ax.plot_surface(T, X, Z, color='blue',alpha=0.4,antialiased=False)
#ax.plot_surface(T, X, meanmat,alpha=0.8)
ax.plot_surface(T, X, meanmat,color='red',alpha=0.4,antialiased=False)
#ax.contour(T, X, stdmat,alpha=1,offset=-0.02)
#ax1.plot_surface(T, X, stdmat*10**6,alpha=1)
ax.plot_surface(T, X, lowquantmat, alpha=0.2, color='orange',antialiased=False)
ax.plot_surface(T, X, highquantmat, alpha=0.2, color='orange',antialiased=False)


ax.set_xlabel('$t$')
ax.set_ylabel('$x$')
ax.set_zlabel('$u$')

plt.show()

