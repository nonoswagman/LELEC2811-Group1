# Fonctionne pas

# Trouver pour chaque HU et t l'abscisse du maximum en dB
# les données se trouvent dans le fichier Draft1.txt qui contient:
"""
Freq.	V(n003)
Step Information: Hu=40 T=0  (Step: 1/52)
0.00000000000000e+00	(9.51888265837593e+00dB,0.00000000000000e+00�)
1.00000000000000e+04	(-5.98787587366519e+00dB,1.58846743569789e+02�)
2.00000000000000e+04	(-3.60559076106012e+00dB,1.37827681513492e+02�)
3.00000000000000e+04	(5.41139038344222e+00dB,1.17098108041489e+02�)
...
Step Information: Hu=40 T=20  (Step: 2/52)
"""

import numpy as np
import matplotlib.pyplot as plt

# Lecture des données
data = np.loadtxt('Draft1.txt', skiprows=2, usecols=(0,1), comments='(')
f = data[:,0]
v = data[:,1]

# Recherche des indices des lignes de début de chaque bloc
indices = np.where(np.isnan(v))[0]

# Initialisation des tableaux
freq_max = np.zeros(len(indices)-1)
v_max = np.zeros(len(indices)-1)

# Boucle sur les blocs
for i in range(len(indices)-1):
    # Extraction des données du bloc
    f_bloc = f[indices[i]:indices[i+1]]
    v_bloc = v[indices[i]:indices[i+1]]
    # Recherche de l'indice du maximum
    index_max = np.argmax(v_bloc)
    # Stockage des résultats
    freq_max[i] = f_bloc[index_max]
    v_max[i] = v_bloc[index_max]

# Affichage des résultats
plt.plot(freq_max, v_max, 'o')
plt.xlabel('Fréquence (Hz)')
plt.ylabel('Maximum (dB)')
plt.show()
