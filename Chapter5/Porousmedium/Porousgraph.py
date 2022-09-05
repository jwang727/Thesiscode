import numpy as np
pderesultslinearlinf = np.loadtxt("Porousresults.txt", comments="#", delimiter=",", unpack=False)

print(type(pderesultslinearlinf))
print(pderesultslinearlinf)

import matplotlib.pyplot as plt
import pandas as pd


N = pderesultslinearlinf[:,0]
linferror = pderesultslinearlinf[:,2]
M =pderesultslinearlinf[:,1]
zscore= pderesultslinearlinf[:,3]
T=8.0
L=20.0

plt.rcParams.update({
    "text.usetex": True,})

N=N.astype(int)
M=M.astype(int)

df = pd.DataFrame(dict(N=N, linferror=linferror, M=M, logtimeres=np.log(T/N), logxres=np.log(L/M), logerror=np.log(linferror), zscore=zscore))

#10 is default font size
plt.rc('font', size=10)

for i in sorted(set(M)):
    plt.plot(-df.loc[df['M'] == i]['logtimeres'], df.loc[df['M'] == i]['logerror'], label='{i}'.format(i=i))
plt.legend(loc='best',title="$m$")
plt.xlabel('$\displaystyle \log(n/T)$')
plt.ylabel('$\displaystyle \log(E_{\infty})$')
plt.savefig('Porousvst.pdf',bbox_inches="tight")
plt.show()

for i in sorted(set(N)):
    plt.plot(-df.loc[df['N'] == i]['logxres'], df.loc[df['N'] == i]['logerror'], label='{i}'.format(i=i))
plt.legend(loc='best',title="$n$")
plt.xlabel('$\displaystyle \log(m/L)$')
plt.ylabel('$\displaystyle \log(E_{\infty})$')
plt.savefig('Porousvsx.pdf',bbox_inches="tight")
plt.show()

for i in sorted(set(M)):
    plt.plot(-df.loc[df['M'] == i]['logtimeres'], df.loc[df['M'] == i]['zscore'], label='{i}'.format(i=i))
plt.legend(loc='best',title="$m$")
plt.xlabel('$\displaystyle \log(n/T)$')
plt.ylabel('$\displaystyle Z$')
plt.savefig('Porousvszscoret.pdf',bbox_inches="tight")
plt.show()

for i in sorted(set(N)):
    plt.plot(-df.loc[df['N'] == i]['logxres'], df.loc[df['N'] == i]['zscore'], label='{i}'.format(i=i))
plt.legend(loc='best',title="$n$")
plt.xlabel('$\displaystyle \log(m/L)$')
plt.ylabel('$\displaystyle Z$')
plt.savefig('Porousvszscorex.pdf',bbox_inches="tight")
plt.show()
