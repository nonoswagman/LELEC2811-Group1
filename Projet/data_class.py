import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from numpy.polynomial.polynomial import Polynomial

# Séparateur des données
motSep = 0.000000
lenght = 0

###############
# Fonctions
###############

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
            if len(temp)!= 0:
                toret.append(temp)
                temp = []
    if len(temp)!= 0:
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


##############################
# Classification
##############################
# Classifier: trouver à quelle type de boisson BoisonAClassifier est la plus proche.

# Dataset connu: Jupiler , Taras Boulba , Maes Radler , Leffe Blonde , Grimbergen Triple , Rochefort 10
data_ref = np.loadtxt("datas/putty.log")
#reponses = ["Jupiler", "Taras Boulba", "Maes Radler", "Leffe Blonde", "Grimbergen Triple", "Rochefort 10", "Eau","Vodka"]
reponses = ["Jupiler","Maes Radler"]
data_ref = sep(data_ref, motSep)
data_ref = delDat(data_ref, lenght)
data_ref = suppr(data_ref)
data_ref = ppm(data_ref)
Jup = data_ref[0]
data_ref = callibration(data_ref, Jup)
#print("Callibration :",data_ref)

# Boissons à classifier:
BoisonsAClassifier = np.loadtxt("datas/putty.log")
BoisonsAClassifier = sep(BoisonsAClassifier, motSep)
BoisonsAClassifier = delDat(BoisonsAClassifier, lenght)
BoisonsAClassifier = suppr(BoisonsAClassifier)
BoisonsAClassifier = ppm(BoisonsAClassifier)
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
degree = 1  # Degré du polynôme
X_train = fit_polynomial(data_ref, degree=degree)
#X_train = data_ref

# Générer les étiquettes correspondantes, chaque boisson a un identifiant unique
y_train = reponses

# Affichage des courbes de ppm des boissons connues, les données sont dans X_train
plt.plot(data_ref[0], label="Jupiler")
plt.plot(data_ref[1], label="Maes Radler")
plt.title("Courbes de ppm des boissons connues")
plt.show()


# Mise à l'échelle pour éviter des biais liés aux différentes échelles des coefficients
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Modèle k-NN
k = 1
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(X_train_scaled, y_train)

# Prédiction
X_test = fit_polynomial(BoisonsAClassifier, degree=degree)
#X_test = BoisonsAClassifier
X_test_scaled = scaler.transform(X_test)
predictions = knn.predict(X_test_scaled)

# Donne le pourcentage de prédiction
predictions_proba = knn.predict_proba(X_test_scaled)

# Si le pourcentage de prédiction est inférieur à 0.5, il s'agit d'une prédiction incertaine
threshold = 0.5
uncertain_predictions = predictions_proba.max(axis=1) < threshold
predictions[uncertain_predictions] = -1  # -1 signifie une prédiction incertain

# Résultats, shoud be Jupiler, Taras Boulba, Maes Radler, Leffe Blonde, Grimbergen Triple, Rochefort 10
print("Résultats de la classification :")
for i, prediction in enumerate(predictions):
    print(f"Boisson {i + 1} : {prediction}")

# Enregistre les résultats dans un fichier
np.savetxt("datas/results.txt", predictions, fmt="%s")

# Affichage des courbes de ppm des boissons à classifier
plt.plot(BoisonsAClassifier[0], label="Boisson 1")
plt.plot(BoisonsAClassifier[1], label="Boisson 2")
plt.title("Courbes de ppm des boissons à classifier")
plt.show()