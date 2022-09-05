import numpy as np
pderesultslinearlinf = np.loadtxt("Burgernoforcerationalresults.txt", comments="#", delimiter=",", unpack=False)

print(type(pderesultslinearlinf))
print(pderesultslinearlinf)

import matplotlib.pyplot as plt
import pandas as pd

N = pderesultslinearlinf[:,0]
linferror = pderesultslinearlinf[:,2]
M =pderesultslinearlinf[:,1]
zscore= pderesultslinearlinf[:,3]
T=30.0
L=2*np.pi

plt.rcParams.update({
    "text.usetex": True,})

N=N.astype(int)
M=M.astype(int)

df = pd.DataFrame(dict(N=N, linferror=linferror, M=M, logtimeres=np.log(T/N), logxres=np.log(L/M), logerror=np.log(linferror), zscore=zscore))


for i in sorted(set(M)):
    plt.plot(-df.loc[df['M'] == i]['logtimeres'], df.loc[df['M'] == i]['logerror'], label='{i}'.format(i=i))
plt.legend(loc='best',title="$m$")
plt.xlabel('$\displaystyle \log(n/T)$')
plt.ylabel('$\displaystyle \log(E_{\infty})$')
plt.savefig('Burgernoforcevstrational.pdf',bbox_inches="tight")
plt.show()

for i in sorted(set(N)):
    plt.plot(-df.loc[df['N'] == i]['logxres'], df.loc[df['N'] == i]['logerror'], label='{i}'.format(i=i))
plt.legend(loc='best',title="$n$")
plt.xlabel('$\displaystyle \log(m/L)$')
plt.ylabel('$\displaystyle \log(E_{\infty})$')
plt.savefig('Burgernoforcevsxrational.pdf',bbox_inches="tight")
plt.show()

for i in sorted(set(M)):
    plt.plot(-df.loc[df['M'] == i]['logtimeres'], df.loc[df['M'] == i]['zscore'], label='{i}'.format(i=i))
plt.legend(loc='best',title="$m$")
plt.xlabel('$\displaystyle \log(n/T)$')
plt.ylabel('$\displaystyle Z$')
plt.savefig('Burgernoforcevszscoretrational.pdf',bbox_inches="tight")
plt.show()

for i in sorted(set(N)):
    plt.plot(-df.loc[df['N'] == i]['logxres'], df.loc[df['N'] == i]['zscore'], label='{i}'.format(i=i))
plt.legend(loc='best',title="$n$")
plt.xlabel('$\displaystyle \log(m/L)$')
plt.ylabel('$\displaystyle Z$')
plt.savefig('Burgernoforcevszscorexrational.pdf',bbox_inches="tight")
plt.show()
