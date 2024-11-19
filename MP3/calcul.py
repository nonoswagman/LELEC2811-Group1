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
# Puissance du bruit ramené à l'entrée (V**2)/sqrt(Hz) / Gain
bruitV = np.sqrt((7.5*10**(-9) / 10**(4/20))**2)*39.5
print("Bruit V = ", bruitV,"nV")
bruitI = np.sqrt((50*10**(-15) / 10**(4/20))**2)*39.5
print("Bruit I = ", bruitI,"pI")

# Ampli 1 GIGA formule (unités pas bonnes ?)
Aire = 39.5*(280/2)+110*39.5
print("Aire = ", Aire,"nV*sqrtH(z)")
KTA = bruitV                                #Pas bon
Pente = ((1000*10-9)**2 - KTA) *2*np.pi*1   #Pas bon
print("Pente = ", Pente,"nV**2/Hz")