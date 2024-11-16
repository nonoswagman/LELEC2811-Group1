import numpy as np
R3 = 100*10**3
Q = 1/np.sqrt(2)

#Q = 1/(3-K) --> 3-K = 1/Q --> 
K = (-1/Q) + 3

#K = 1 + R4/R3
R4 = (K-1)*R3

print("R4 = ", R4)


##### NOISE #####

# Filtre, juste du TLV354

# Ampli 1 GIGA formule