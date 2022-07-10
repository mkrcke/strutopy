import matplotlib.pyplot as plt
import scipy.stats
import numpy as np


mean = np.array([0, 0, 0, -2, 2.0])
sigma = np.array([0.2, 1, 2, .5, .5])


for i in range(5): 

    x_min = -10
    x_max = 10
    x = np.linspace(x_min, x_max, 100)

    y = scipy.stats.norm.pdf(x,mean[i],np.sqrt(sigma[i]))

    plt.plot(x,y, alpha = 0.4, label = r"$\mu$ = %.1f, $\sigma^2$ = %.1f"% (mean[i],sigma[i]))

    plt.grid()

    plt.xlim(x_min,x_max)
    plt.ylim(0,1)

    #plt.title(r"Sampling Scheme for $\gamma_k$", fontsize=10)
    plt.legend()

    xarea = np.append(x, x[-1]) 
    yarea = np.append(y, 0 )
    plt.fill(xarea,yarea)
    plt.xlabel(r"Varying values for $\mu$")
    plt.ylabel(r"$p(x|\mu, \sigma^2$)")

#plt.show()
plt.savefig('gamma_scheme.png', bbox_inches='tight', dpi=360)