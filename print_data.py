import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from numpy.fft import fft, ifft

motSep = 0

#data_capteur = np.loadtxt("le log", skiprows=1, delimiter='\t')
data_capteur = [0 , 1,2,3,4,5,5,5, 0 , 1,4,6,8,10,10,10, 0 ,0.5,1,1.5,2,2.5,2.5,2.5 ]
data_split = []

# Split en fonction du mot délimitateur
temp = []
for tension in data_capteur:
    if tension != motSep:
        temp.append(tension)
    else:
        if len(temp)!= 0:
            data_split.append(temp)
        temp = []
if len(temp)!= 0:
    data_split.append(temp)

print("séparation",data_split)

# Supprime les premiers éléments de chaque mesures
for i in range(len(data_split)):
    distSuppr = round(1*len(data_split[i])/100)
    data_split[i] = data_split[i][distSuppr:]
        
print("Retirer 1%",data_split)
#Parfait merci Loïc <3

# Récupère le PPM
R0 = 400e3
R_bias = 100e3
Vint_max = 3.3
ppm_min = 1
ppm_max = 300
ppm = np.linspace(ppm_min, ppm_max,100)
RS = R0*((ppm**(-0.632))*10**0.1090)
Rs_max = RS[-1]
V_ref = Vint_max/(1+R_bias/Rs_max)
A = -0.632
B = 0.1090

for boisson in data_split:
    for j in range(len(boisson)):
        Rs = R_bias / ( boisson[j]/V_ref -1)
        boisson[j] = (Rs/ (R0 * 10**B))**(1/A)

print("PPM:",data_split)

# Normalisation des datas en fonction de la jupilère: Tous les ppm = ppm-ppm_JUP
Jup = data_split[0]
for boisson in data_split:
    for i in range(len(boisson)):
        if(i<len(Jup)):
            boisson[i] = boisson[i]/Jup[i]
        else:
            boisson[i] = 0 # Pas sû normalisé, on met 0
print("Normalisé:",data_split)


##############################
# Classification
##############################

BoisonAClassifier = data_split[0]

# Classifier trouver à quelle type de boisson BoisonAClassifier est la plus proche.

# Dataset = chaque éléments de data_split: Jupilère, .... A COMPLETER ET CREER

Dataset = data_split

nbrBoisson = len(data_split)

# Train, fit, ....

# Trouver le type de boisson de BoisonAClassifier
predict = ...
print("Prédiction:",predict)