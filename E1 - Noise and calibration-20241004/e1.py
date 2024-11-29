import matplotlib.pyplot as plt
import numpy as np

C = np.linspace(1,1000,1000)

Vref = 50e-3
R0 = 62e3
RB = 50e3
b = -0.631
v0 = Vref*(1+RB/(R0*C**b))

plt.plot(C,v0)
plt.xlabel("CO2 ppm")
plt.ylabel("Vout")
plt.show()