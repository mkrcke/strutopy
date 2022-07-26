import matplotlib.pyplot as plt
import scipy.stats
import numpy as np


mean = np.array([-2, -1, 0, 1, 2])
sigma = np.array([1, 1, 1, 1, 1])


for i in range(5): 

    x_min = -10
    x_max = 10
    x = np.linspace(x_min, x_max, 100)

    y = scipy.stats.norm.pdf(x,mean[i],np.sqrt(sigma[i]))

    plt.plot(x,y, alpha = 0.4, label = r"$\mu$ = %.1f"% (mean[i]))

    plt.grid()

    plt.xlim(x_min,x_max)
    plt.ylim(0,.6)

    #plt.title(r"Choosing different values for $\gamma_k$", fontsize=10)
    plt.legend()

    xarea = np.append(x, x[-1]) 
    yarea = np.append(y, 0 )
    plt.fill(xarea,yarea)
    plt.xlabel(r"Varying values for $\mu$")
    plt.ylabel(r"$p(x|\mu, \sigma^2$)")

#plt.show()
plt.savefig('img/gamma_scheme.png', bbox_inches='tight', dpi=360)