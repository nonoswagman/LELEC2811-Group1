import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from numpy.polynomial.polynomial import Polynomial

# Séparateur des données
start = "Press 'R' to read the data"
motSep = 0.000000
lenght = 0

###############
# Fonctions
###############

# Supprime les données avant la ligne "Press 'R' to read the data"
def delData(data):
    """ Supprime les données avant la ligne "Press 'R' to read the data"
    """
    toret = []
    for tension in data:
        if tension == start:
            toret = []
        else:
            toret.append(tension)
    return toret

# Split en fonction du mot délimitateur
def sep(data, motSep):
    """ Sépare les données en fonction du mot délimitateur motSep
    """
    toret = []
    temp = []
    for tension in data:
        if tension != motSep:
            temp.append(tension)
        else:
            if len(temp)>200:
                toret.append(temp)
                temp = []
    if len(temp)>200:
        toret.append(temp)
    return toret

# Supprime les datas qui ont une longuer inférieure à length
def delDat(data, length):
    """ Supprime les datas qui ont une longuer inférieure à length
    """
    toret = []
    for tension in data:
        if len(tension) > length:
            toret.append(tension)
    return toret

# Supprime les premiers éléments de chaque mesures et les derniers éléments de chaque mesures
def suppr(data_split):
    """ Supprime les premiers éléments de chaque mesures et les derniers éléments de chaque mesures
    """
    for boisson in data_split:
        boisson.pop(0)
        boisson.pop(-1)
    return data_split

# Récupère le PPM
def ppm(data_split):
    R0 = 380e3
    R_bias = 100e3
    Vint_min = 0.314
    ppm_min = 1.2
    
    #ppm = np.linspace(ppm_min, ppm_max,100)
    Rs_min = R0*((ppm_min**(-0.632))*10**0.1090)
    V_ref = Vint_min/(1+R_bias/Rs_min)
    A = -0.632
    B = 0.1090

    for boisson in data_split:
        for j in range(len(boisson)):
            Rs = R_bias / ( boisson[j]/V_ref -1)
            base = Rs / (R0 * 10**B)
            if base > 0:
                boisson[j] = (Rs / (R0 * 10**B))**(1/A)
            else:
                boisson[j] = -1
                #print("Invalid base:", base, ( boisson[j]/V_ref -1))
    return data_split

# Callibration des datas en fonction de la jupilère: Tous les ppm = ppm-ppm_JUP
def callibration(data_split, Jup):
    for boisson in data_split:
        for i in range(len(boisson)):
            if(i<len(Jup)):
                boisson[i] = boisson[i]-Jup[i]
            else:
                boisson[i] = 0 # Pas sû normalisé, on met 0
    return data_split

def printData(data, labels, title):
    plt.plot(data[0], label=labels[0])
    plt.plot(data[1], label=labels[1])
    plt.plot(data[2], label=labels[2])
    plt.plot(data[3], label=labels[3])
    plt.legend()
    plt.title(title)
    plt.show()


##############################
# Classification
##############################
# Classifier: trouver à quelle type de boisson BoisonAClassifier est la plus proche.

# Dataset connu: Jupiler , Taras Boulba , Maes Radler , Leffe Blonde , Grimbergen Triple , Rochefort 10
# On a 4 mesures pour chaque boisson, prend la 4 eme colonne de chaque mesure
data_ref = np.loadtxt("datas/ju_grim_roch_t1.log",usecols=3)
data_ref = np.concatenate((data_ref, np.loadtxt("datas/ju_grim_roch_t2.log",usecols=3)), axis=0)
data_ref = np.concatenate((data_ref, np.loadtxt("datas/ju_grim_roch_t3.log",usecols=3)), axis=0)
data_ref = np.concatenate((data_ref, np.loadtxt("datas/ju_grim_roch_t1.log",usecols=3)), axis=0)

#reponses = ["Jupiler", "Taras Boulba", "Maes Radler", "Leffe Blonde", "Grimbergen Triple", "Rochefort 10", "Eau","Vodka"]
reponses = ["Jupiler","Grimbergen Triple","Rochefort 10","Jupiler","Jupiler","Grimbergen Triple","Rochefort 10","Jupiler","Jupiler","Grimbergen Triple","Rochefort 10","Jupiler","Jupiler","Grimbergen Triple","Rochefort 10","Jupiler"]
data_ref = sep(data_ref, motSep)
data_ref = delDat(data_ref, lenght)
data_ref = suppr(data_ref)
data_ref = ppm(data_ref)
data_ref_sansCallibration = data_ref
Jup = data_ref[0]
data_ref = callibration(data_ref, Jup)
#print("Callibration :",len(data_ref))

# Boissons à classifier:
BoisonsAClassifier = list(np.loadtxt("datas/ju_grim_roch_t4.log",usecols=3))
BoisonsAClassifier = sep(BoisonsAClassifier, motSep)
BoisonsAClassifier = delDat(BoisonsAClassifier, lenght)
BoisonsAClassifier = suppr(BoisonsAClassifier)
BoisonsAClassifier = ppm(BoisonsAClassifier)
BoisonsAClassifier_sansCallibration = BoisonsAClassifier
Jup = BoisonsAClassifier[0]
BoisonsAClassifier = callibration(BoisonsAClassifier, Jup)
#print("Callibration :",BoisonsAClassifier)


def fit_polynomial(boissons, degree=3):
    """
    Ajuste un polynôme à chaque série de données ppm.
    Retourne les coefficients des polynômes pour chaque boisson.
    """
    X = []
    for boisson in boissons:
        if len(boisson) > 0:
            poly = Polynomial.fit(range(len(boisson)), boisson, degree)
            X.append(poly.coef)
    return np.array(X)

def plot_polynomial(poly, ax, label=None):
    """
    Trace un polynôme sur un graphique.
    """
    x = np.linspace(0, 100, 100)
    y = poly(x)
    ax.plot(x, y, label=label)

# Récupération des coéficients des courbes de ppm des boissons connues
degree = 3  # Degré du polynôme
X_train = fit_polynomial(data_ref, degree=degree)
#X_train = data_ref

# Générer les étiquettes correspondantes, chaque boisson a un identifiant unique
y_train = reponses

# Affichage des courbes de ppm des boissons connues, les données sont dans X_train (Jupiler, Grimbergen Triple, Rochefort 10)
printData(data_ref_sansCallibration[0:4], ["Jupiler","Grimbergen Triple","Rochefort 10","Jupiler"], "Courbes de ppm des boissons connues T1")

printData(data_ref_sansCallibration[4:8], ["Jupiler","Grimbergen Triple","Rochefort 10","Jupiler"], "Courbes de ppm des boissons connues T2")

printData(data_ref_sansCallibration[8:12], ["Jupiler","Grimbergen Triple","Rochefort 10","Jupiler"], "Courbes de ppm des boissons à classifier")

printData(data_ref_sansCallibration[12:16], ["Jupiler","Grimbergen Triple","Rochefort 10","Jupiler"], "Courbes de ppm des boissons à classifier")



# Mise à l'échelle pour éviter des biais liés aux différentes échelles des coefficients
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Modèle k-NN
k = 1
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(X_train_scaled, y_train)

# Prédiction
X_test = fit_polynomial(BoisonsAClassifier, degree=degree)
X_test_scaled = scaler.transform(X_test)
predictions = knn.predict(X_test_scaled)

# Donne le pourcentage de prédiction
predictions_proba = knn.predict_proba(X_test_scaled)

# Si le pourcentage de prédiction est inférieur à 0.5, il s'agit d'une prédiction incertaine
threshold = 0.5
uncertain_predictions = predictions_proba.max(axis=1) < threshold
predictions[uncertain_predictions] = -1  # -1 signifie une prédiction incertain

# Résultats, shoud be Jupiler, Taras Boulba, Maes Radler, Leffe Blonde, Grimbergen Triple, Rochefort 10
printData(BoisonsAClassifier_sansCallibration, predictions, f"Résultats de la classification: {predictions}")

# Enregistre les résultats dans un fichier
np.savetxt("datas/results.txt", predictions, fmt="%s")