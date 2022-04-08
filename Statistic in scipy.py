import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
plt.style.use('ggplot')
# print(stats.rv_continuous)

rv_bernoulli = stats.bernoulli(p=0.3)
#print(rv_bernoulli.rvs(14))

#rv_binom = stats.binom(100, p=0.9)
# print(rv_binom.rvs(8))
#plt.hist(rv_binom.rvs(80), bins=10)
#plt.show()

a = 5
b = 10
rv_uniform = stats.uniform(a, b - a)
#print(rv_uniform.cdf(11), rv_uniform.cdf(4),
 #     rv_uniform.cdf(5.5)
 #     ) #функция распределения в точке
#print(rv_uniform.pdf(11), rv_uniform.pdf(4),
#      rv_uniform.pdf(5.5) ) #плотность вероятности

#X = np.linspace(a - 2, b + 2, 1000) #график плотности распределения
#pdf = rv_uniform.pdf(X)
#plt.plot(X, pdf)
#plt.ylabel('f(x)')
#plt.xlabel('x')
#plt.xlim([4,12])
#plt.title(u'PDF for uniform')
#plt.show()


mu = 10
sigmas = [1, 2, 4, 6]
for sigma in sigmas:
    rv_norm = stats.norm(loc=mu, scale=sigma)
    #print(rv_norm.rvs(17))
    x = np.linspace(0, 20, 200)
    pdf = rv_norm.pdf(x)
    plt.plot(x, pdf, label=sigma)
    plt.ylabel('f(x)')
    plt.xlabel('x')
plt.legend()
plt.show()
mean, var, skew = rv_norm.stats(moments='mvs')
print(skew) #параметр смещенности (ассиметричности)



