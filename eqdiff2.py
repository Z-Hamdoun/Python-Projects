# Le 8 Septembre 2023
# Auteur: Hamdoun Zakaria
# LM6E MPSI-1
# ----------------------------------------------------------------------------------------------------------------------------------
# Description: Ce fichier est un outil permettant la résolution numérique (méthode d'Euler) des équations differentielles linéaires 
# homogenes de deuxieme ordre qui s'écrivent sous forme d2x/dt2 +2alphadx/dt + omega0^2 x = 0
# ----------------------------------------------------------------------------------------------------------------------------------
# Instructions: le fichier doit être mis et executé dans un nouveau dossier car il se peut qu'il supprime quelques fichiers sous 
# format PNG. Une fois executé, le programme demande à l'utilisateur de saisir les valeurs de alpha et omega0, ainsi que celles
# des conditions initiales (x(t=0) et dx/dt]t=0 ) qui doivent être des nombres réels. Ensuite, l'utilisateur doit saisir au plus 10
# valeurs pour le pas (exemple: 1, 0.1, 0.01, 0.001) , une fois terminé, il faut saisir la lettre "q" pour arrêter le programme,
# qui regroupe les differents graphes pour chaque pas dans un fichier GIF dans le même dossier qui permet de comparer l'allure de 
# la courbe pour chaque valeur prise par le pas. Il permet aussi de tracer la trajectoire de phase, cependant un petit changement 
# doit être effectué. La courbe exacte (solution analytique) est aussi tracée pour faciliter la comparaison.
# ----------------------------------------------------------------------------------------------------------------------------------

import matplotlib.pyplot as plt
import numpy as np
import glob
from PIL import Image
import os


def f(t, alpha, omega0):
    # Cette fonction est la solution analytique de l'ED
    delta = alpha**2 - omega0**2
    if delta == 0:
        return np.exp(-alpha*t)*((dx0 + alpha*x0)*t + x0)
    elif delta > 0:
        beta = np.sqrt(alpha**2 - omega0**2)
        return np.exp(-alpha*t)*(((dx0 + x0*(alpha + beta))/(2*beta))*np.exp(beta*t) + ((x0*(beta-alpha)-dx0)/(2*beta))*np.exp(-beta*t))
    elif delta < 0:
        omega = np.sqrt(omega0**2 - alpha**2)
        return np.exp(-alpha*t)*(x0*np.cos(omega*t)+ ((dx0+alpha*x0)/omega)*np.sin(omega*t))

def graph(p, alpha, omega0, x0, dx0):
    X = np.arange(0, 20, p)
    Xa = np.arange(0, 20, p)

    # Solution analytique
    Ya = f(Xa, alpha, omega0)

    # Solution aproximative
    x = np.zeros(len(X))
    dx = np.zeros(len(X))
    x[0] = x0
    dx[0] = dx0

    for i in range(0, len(x)-1):
        x[i+1] = ((x[i]*(1+(2*alpha*p))+p*dx[i])/((omega0*p)**2 + 1 +(2*alpha*p)))
        dx[i+1] = (x[i+1]-x[i])/p
    
    plt.plot(Xa, Ya, 'g', label='analytique')
    plt.plot(X, x,'r', label='aproximative')

    plt.grid()
    plt.legend(loc='lower right')
    plt.title(f"Pas: {str(p)}")

def make_gif(dossier):
    # Créer un fichier GIF et supprimer les fichiers png
    frames = [Image.open(image) for image in sorted(glob.glob(f"{dossier}/*.png"))]
    frame_one = frames[0]
    frame_one.save("Comparaison", format="GIF", append_images=frames, save_all=True, duration=len(frames)*500, loop=0)
    for image in glob.glob(f"{dossier}/*.png"):
        os.remove(image)


alpha = float(input("alpha: "))
omega0 = float(input("omega0: "))
x0 = float(input("x0: "))
dx0 = float(input("dx0: "))

for i in range(10):
    p = input("pas: ")
    if p == "q":
        make_gif(".")
        exit()
    else:
        graph(float(p), alpha, omega0, x0, dx0)
        plt.savefig(f"plot{str(i)}.png")
        plt.clf()

