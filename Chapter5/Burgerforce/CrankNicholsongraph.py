import numpy as np
pderesultslinearlinf = np.loadtxt("CrankNicholsonBurger.txt", comments="#", delimiter=",", unpack=False)

print(type(pderesultslinearlinf))
print(pderesultslinearlinf)

import matplotlib.pyplot as plt
import pandas as pd


N = pderesultslinearlinf[:,0]
linferror = pderesultslinearlinf[:,2]
M =pderesultslinearlinf[:,1]
T=30.0
L=1.0

plt.rcParams.update({
    "text.usetex": True,})

N=N.astype(int)
M=M.astype(int)

df = pd.DataFrame(dict(N=N, linferror=linferror, M=M, logtimeres=np.log(T/N), logxres=np.log(L/M), logerror=np.log(linferror)))

#10 is default font size
plt.rc('font', size=10)

for i in sorted(set(M)):
    plt.plot(-df.loc[df['M'] == i]['logtimeres'], df.loc[df['M'] == i]['logerror'], label='{i}'.format(i=i))
plt.legend(loc='best',title="$m$")
plt.xlabel('$\displaystyle \log(n/T)$')
plt.ylabel('$\displaystyle \log(E_{\infty})$')
plt.savefig('CrankNicholsonBurgervst.pdf',bbox_inches="tight")
plt.show()

for i in sorted(set(N)):
    plt.plot(-df.loc[df['N'] == i]['logxres'], df.loc[df['N'] == i]['logerror'], label='{i}'.format(i=i))
plt.legend(loc='best',title="$n$")
plt.xlabel('$\displaystyle \log(m/L)$')
plt.ylabel('$\displaystyle \log(E_{\infty})$')
plt.savefig('CrankNicholsonBurgervsx.pdf',bbox_inches="tight")
plt.show()