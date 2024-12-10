import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from numpy.polynomial.polynomial import Polynomial
from scipy.signal import butter, lfilter, freqz, freqs,sosfilt
from scipy.optimize import curve_fit

# Séparateur des données
start = "Press 'R' to read the data"
motSep = 0.000000
lenght = 0

###############
# Fonctions
###############
def printData(data, labels, title):
    plt.plot(data[0], label=labels[0])
    plt.plot(data[1], label=labels[1])
    plt.plot(data[2], label=labels[2])
    plt.plot(data[3], label=labels[3])
    plt.legend()
    plt.title(title)
    plt.savefig("normal.pdf")
    plt.show()

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
            if len(temp)>150:
                toret.append(temp)
                temp = []
    if len(temp)>150:
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
    R_bias = 20e3
    Vint_min = 0.314
    ppm_min = 1.2
    
    #ppm = np.linspace(ppm_min, ppm_max,100)
    Rs_min = R0*((ppm_min**(-0.632))*10**0.1090)
    V_ref = Vint_min/(1+R_bias/Rs_min)
    A = -0.632
    B = 0.1090
    f = 10**(-B/A)

    for boisson in data_split:
        for j in range(len(boisson)):
            Rs = R_bias / ( boisson[j]/V_ref -1)
            base = Rs / (R0 * 10**B)
            if base > 0:
                boisson[j] = (Rs/R0)**A*f
            else:
                boisson[j] = -1
                #print("Invalid base:", base, ( boisson[j]/V_ref -1))
    return data_split

# Callibration des datas en fonction de la jupilèr: Tous les ppm au temps i = ppm-i-ppm_JUP_i
def calibration_ref(data_split, Jup):
    for j,boisson in enumerate(data_split):
            if j<4 or (j>=20 and j<23) :#T1 mesure 1 et 2
                for i in range(len(boisson)):
                    if(i<len(Jup[0])):
                        boisson[i] = boisson[i]-Jup[0][i]
                    else:
                        boisson[i] = 0 # Pas sû normalisé, on met 0
            if (j>=4 and j<8) or (j>=23 and j<27):#T2 mesure 1 et 2
                for i in range(len(boisson)):
                    if(i<len(Jup[1])):
                        boisson[i] = boisson[i]-Jup[1][i]
                    else:
                        boisson[i] = 0 # Pas sû normalisé, on met 0
            if (j>=8 and j<12)or (j>=27 and j<30):#T3 mesure 1 et 2
                for i in range(len(boisson)):
                    if(i<len(Jup[2])):
                        boisson[i] = boisson[i]-Jup[2][i]
                    else:
                        boisson[i] = 0 # Pas sû normalisé, on met 0
            if (j>=12 and j<16)or (j>=30 and j<34):#T4 mesure 1 et 2
                for i in range(len(boisson)):
                    if(i<len(Jup[3])):
                        boisson[i] = boisson[i]-Jup[3][i]
                    else:
                        boisson[i] = 0 # Pas sû normalisé, on met 0
            if (j>=16 and j<20) or (j>=34 and j<37):#T5 mesure 1 et2
                for i in range(len(boisson)):
                    if(i<len(Jup[4])):
                        boisson[i] = boisson[i]-Jup[4][i]
                    else:
                        boisson[i] = 0 # Pas sû normalisé, on met 0
    return data_split

def calibration_test(data_split, Jup):
    for boisson in range(len(data_split)):
            for i in range(len(data_split[boisson])):
                if(i<len(Jup)):
                    data_split[boisson][i] = data_split[boisson][i]-Jup[i]
                else:
                    data_split[boisson][i] = 0 # Pas sû normalisé, on met 0
    return data_split



##############################
# Classification
##############################
# Classifier: trouver à quelle type de boisson BoisonAClassifier est la plus proche.

# Dataset connu: Jupiler , Taras Boulba , Maes Radler , Leffe Blonde , Grimbergen Triple , Rochefort 10
# On a 4 mesures pour chaque boisson, prend la 4 eme colonne de chaque mesure

data_test = np.loadtxt("datas/contest.log")
data_ref = np.loadtxt("datas/ju_grim_roch_t1.log",usecols=3)
data_ref = np.concatenate((data_ref, np.loadtxt("datas/ju_grim_roch_t2.log",usecols=3)), axis=0)
data_ref = np.concatenate((data_ref, np.loadtxt("datas/ju_grim_roch_t3.log",usecols=3)), axis=0)
data_ref = np.concatenate((data_ref, np.loadtxt("datas/ju_grim_roch_t4.log",usecols=3)), axis=0)
data_ref = np.concatenate((data_ref, np.loadtxt("datas/ju_grim_roch_t5.log",usecols=3)), axis=0)
data_ref = np.concatenate((data_ref, np.loadtxt("datas/2_T1.log",usecols=3)), axis=0)
data_ref = np.concatenate((data_ref, np.loadtxt("datas/2_T2.log",usecols=3)), axis=0)
data_ref = np.concatenate((data_ref, np.loadtxt("datas/2_T3.log",usecols=3)), axis=0)
data_ref = np.concatenate((data_ref, np.loadtxt("datas/2_T4.log",usecols=3)), axis=0)
data_ref = np.concatenate((data_ref, np.loadtxt("datas/2_T5.log",usecols=3)), axis=0)

#reponses = ["Jupiler", "Taras Boulba", "Maes Radler", "Leffe Blonde", "Grimbergen Triple", "Rochefort 10", "Eau","Vodka"]
reponses = ["Jupiler","Grimbergen Triple","Rochefort 10","Jupiler",\
            "Jupiler","Grimbergen Triple","Rochefort 10","Jupiler",\
                "Jupiler","Grimbergen Triple","Rochefort 10","Jupiler",\
                    "Jupiler","Grimbergen Triple","Rochefort 10","Jupiler",\
                        "Jupiler","Grimbergen Triple","Rochefort 10","Jupiler",\
                            "Radler", "Bulba", "Leffe",\
                                "Radler", "Bulba", "Leffe", "Leffe",\
                                    "Radler", "Bulba", "Leffe",\
                                            "Radler", "Bulba", "Leffe", "Leffe",\
                                                "Radler", "Bulba", "Leffe"]



reponses_r = [ 5.2, 9, 11.3, 5.2,\
              5.2, 9, 11.3, 5.2,\
              5.2, 9, 11.3, 5.2,\
              5.2, 9, 11.3, 5.2,\
              5.2, 9, 11.3, 5.2,\
              2.0, 4.5, 6.6,\
              2.0, 4.5, 6.6, 6.6,\
              2.0, 4.5, 6.6,\
              2.0, 4.5, 6.6, 6.6,\
              2.0, 4.5, 6.6]

reponse_t = [4.5,2,4.5,6.6]

#print(len(reponses_r), len(data_ref))
data_ref = sep(data_ref, motSep)
data_test = sep(data_test, motSep)
#print(f"len data_ref {len(data_ref)}")
data_ref = delDat(data_ref, lenght)
data_ref = suppr(data_ref)
data_ref = ppm(data_ref)
data_test = delDat(data_test, lenght)
data_test = suppr(data_test)
data_test = ppm(data_test)

plt.plot(np.concatenate(data_test))
plt.show()
#print(f"len data_ref {len(data_ref)}")
data_ref_sansCallibration = data_ref.copy()
#####################################Creer array avec Jup T1,T2,T3,T4,T5
Jup_ref=[]
for i in range(5):
    Jup_ref.append(data_ref[i*4])

Jup_test=data_test[6] #A changer par data_test_0 !!!

print(len(data_test))
data_ref = calibration_ref(data_ref, Jup_ref)
data_test = calibration_test(data_test, Jup_test)

plt.plot(np.concatenate(data_test))
plt.show()
#print("Callibration :",len(data_ref))

# Boissons à classifier:
"""
BoisonsAClassifier = list(np.loadtxt("datas/ju_grim_roch_t4.log",usecols=3))
BoisonsAClassifier = sep(BoisonsAClassifier, motSep)
BoisonsAClassifier = delDat(BoisonsAClassifier, lenght)
BoisonsAClassifier = suppr(BoisonsAClassifier)
BoisonsAClassifier = ppm(BoisonsAClassifier)
BoisonsAClassifier_sansCallibration = BoisonsAClassifier.copy()
Jup = BoisonsAClassifier[0]
BoisonsAClassifier = callibration(BoisonsAClassifier, Jup)
#print("Calibration :",BoisonsAClassifier)
"""


def fit_polynomial(boissons, degree=3):
    """
    Ajuste un polynôme à chaque série de données ppm.
    Retourne les coefficients des polynômes pour chaque boisson.
    """
    X = []
    for boisson in boissons:
        if len(boisson) > 0:
            poly = Polynomial.fit(range(len(boisson)), boisson, degree)
            x = np.arange(0, 100, 1)
            y = poly(x)
            plt.plot(poly(x))
            plt.plot(boisson)
            plt.show()
            X.append(poly.coef)
    return np.array(X)

def fit_log_w(boissons):
    """
    Ajuste un polynôme à chaque série de données ppm.
    Retourne les coefficients des polynômes pour chaque boisson.
    """
    X = []
    for boisson in boissons:
        if len(boisson) > 100:
            #poly = np.array(fit_log(boisson))
            a = np.quantile(boisson, 0.90)
            #print(a)
            r = np.array([0, 0, 0,0,a])
            X.append(r)
    return np.array(X)

def fit_log(boissons):
    """
    Fit the first 150 elements of the "boissons" array with the function:
    f(x) = a * log_b(c * x) + d

    Parameters:
        boissons (array-like): Input data to fit.

    Returns:
        popt (tuple): Optimized parameters (a, b, c, d).
    """
    def log_func(x, a, b, c, d):
        return a * np.log(c * x) / np.log(b) + d

    # Limit to the first 150 elements
    x_data = np.arange(1, min(100, len(boissons)) + 1)  # Avoid log(0)
    y_data = boissons[:100]

    # Initial parameter guesses: a=1, b=2, c=1, d=0
    initial_guess = [1, 5, 1, 3]

    # Curve fitting
    (a,b,c,d), _ = curve_fit(log_func, x_data, y_data, p0=initial_guess)

    """absci = np.arange(0,100,1)
    ordo = log_func(absci,a,b,c,d)
    plt.plot(absci, boissons[:100])
    plt.plot(absci, ordo)
    plt.show()"""

    return a,b,c,d

def plot_polynomial(poly, ax, label=None):
    """
    Trace un polynôme sur un graphique.
    """
    x = np.linspace(0, 100, 100)
    y = poly(x)
    ax.plot(x, y, label=label)

def butter_lowpass(cutoff, fs, order=100):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    sos = butter(order, normal_cutoff, btype='low', output='sos')
    return sos

def butter_lowpass_filter(data, cutoff, fs, order=20):
    sos = butter_lowpass(cutoff, fs, order=order)
    y = sosfilt(sos, data)
    return y[order//2:]

def filter_loop(data, cutoff, fs, order=20):
    res = []
    for d in data:
        res.append(butter_lowpass_filter(d, cutoff, fs, order=20))
    return res
def printData_f(data, labels, title):
    for i in range(len(data)):
        printData_2(data[i], labels[i], title)

def printData_2(data, labels, title):
    plt.plot(data, label=labels)
    plt.plot(butter_lowpass_filter(np.array(data),1,10), label=labels)
    plt.legend()
    plt.title(title)
    plt.savefig("normal.pdf")
    plt.show()


data_ref = filter_loop(data_ref,1,10)
data_test = filter_loop(data_test,1,10)

plt.plot(np.concatenate(data_test))
plt.show()
# Récupération des coéficients des courbes de ppm des boissons connues
degree = 3  # Degré du polynôme
X_train = fit_log_w(data_ref)
X_test = fit_log_w(data_test)
#print(X_train.shape)
# Générer les étiquettes correspondantes, chaque boisson a un identifiant unique
y_train = reponses

# Affichage des courbes de ppm des boissons connues, les données sont dans X_train (Jupiler, Grimbergen Triple, Rochefort 10)
#printData(data_ref_sansCallibration[0:4], ["Jupiler","Grimbergen Triple","Rochefort 10","Jupiler"], "Courbes de ppm des boissons connues T1")
#printData_f(data_ref[20:], reponses[20:], "Courbes de ppm des boissons connues T1")

"""#printData(data_ref_sansCallibration[4:8], ["Jupiler","Grimbergen Triple","Rochefort 10","Jupiler"], "Courbes de ppm des boissons connues T2")
printData_f(data_ref_sansCallibration[4:8], ["Jupiler","Grimbergen Triple","Rochefort 10","Jupiler"], "Courbes de ppm des boissons connues T1")

#printData(data_ref_sansCallibration[8:12], ["Jupiler","Grimbergen Triple","Rochefort 10","Jupiler"], "Courbes de ppm des boissons à classifier")
printData_f(data_ref_sansCallibration[8:12], ["Jupiler","Grimbergen Triple","Rochefort 10","Jupiler"], "Courbes de ppm des boissons connues T1")

#printData(data_ref_sansCallibration[12:16], ["Jupiler","Grimbergen Triple","Rochefort 10","Jupiler"], "Courbes de ppm des boissons à classifier")
printData_f(data_ref_sansCallibration[12:16], ["Jupiler","Grimbergen Triple","Rochefort 10","Jupiler"], "Courbes de ppm des boissons connues T1")
"""
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor


model = RandomForestRegressor(n_estimators=10)

#for i in X_train:
    #print(i.shape)
model.fit(X_train, reponses_r)
result = (model.predict(X_train))

def RMSE(x,y):
    diff = (x-y)**2
    return (np.sum(diff)/len(x))**0.5
print("RMSE",RMSE(result, reponses_r))

plt.scatter(reponses_r, result, color="orange")

xx = np.linspace(2,12,1000)
plt.plot(xx,xx)
plt.show()
plt.plot(X_test[:,4])
plt.show()
result_test = (model.predict(X_test))
#print(f"RMSE test {RMSE(result_test, reponse_t)}")
print("Vrai pourcentage:",reponse_t)
print("Pourcentage_estimé",result_test)


from sklearn.ensemble import RandomForestClassifier


model = RandomForestClassifier(n_estimators=5)

#for i in X_train:
    #print(i.shape)
model.fit(X_train, reponses)
result = (model.predict(X_train))

def RMSE(x,y):
    diff = (x-y)**2
    return (np.sum(diff)/len(x))**0.5
#print(result)
#print(reponses)

plt.scatter(reponses, result, color="orange")

xx = np.linspace(2,12,1000)
plt.plot(xx,xx)
plt.show()

result_test = (model.predict(X_test))

# If d>3.2 it's vodka, if d<0.55 it's a soft
for i in range(len(X_test)):
    print(X_test[i])
    if X_test[i][4] > 35:
        result_test[i] = "Vodka"
    elif X_test[i][4] < 5:
        result_test[i] = "Soft"

print("Vraie bière",["bulba", "radler", "bulba", "leffe"])
print("Bière estimée",result_test)



"""# Mise à l'échelle pour éviter des biais liés aux différentes échelles des coefficients
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Modèle k-NN
k = 2
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
"""