import csv
import heapq
import math
import random
from collections import deque
from sys import maxsize
from typing import List

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

import prioqueue
from Graphe import Graphe
import itertools
import threading
import time

import sys

######################################################################################################
# Driver Code


'''def animate_loading(method: object) -> object:
    def animated():

        done = False

        # here is the animation
        def animate():
            time.sleep(0.000001)
            for c in itertools.cycle(['|', '/', '-', '\\']):
                if done:
                    break
                sys.stdout.write('\rloading ' + c)
                sys.stdout.flush()
                time.sleep(0.1)
            sys.stdout.write('\rDone!     ')

        t = threading.Thread(target=animate)
        t.start()

        # process method here
        method()
        done = True

    return animated()'''

file = open("edge.csv")
csvreader = csv.reader(file)

header = next(csvreader)

rows = []
for row in csvreader:
    rows.append(row)

file.close()

EDGES = []
for row in rows:
    row1 = row[0].split(';')
    row2 = []
    for s in row1:
        row2.append(int(s))

    EDGES.append(row2)

file2 = open("demande.csv")
csvreader2 = csv.reader(file2)

header2 = next(csvreader2)

rows2 = []
for row in csvreader2:
    rows2.append(row)

file2.close()

DEMANDE = []
for row in rows2:
    rowA = row[0].split(';')
    rowB = []
    for s in rowA:
        rowB.append(float(s))

    DEMANDE.append(rowB)

file1 = open("noeux.csv")
csvreader1 = csv.reader(file1)

header1 = next(csvreader1)

rows1 = []
for row in csvreader1:
    rows1.append(row)

file.close()

NODES = []
for row in rows1:
    row3 = row[0].split(';')
    row4 = []
    for s in row3:
        row4.append(s)

    NODES.append(row4)
##########################################################################################################


nombre_edge = len(EDGES)
nombre_noeuds = len(NODES)
nombre_demande = len(DEMANDE)

# weights = [row_norm[i][3] for i in range(nombre_edge)]
WEIGHTS_global = [random.randint(1, 5) for i in range(nombre_edge)]


###########################################################################################################

##########################################################################################

#################################################################################################################


def create_nx(weight):
    g = nx.Graph()
    for i in range(nombre_noeuds):
        g.add_node(i, label=NODES[i][2], col='blue')

    for i in range(nombre_edge):
        g.add_edge(EDGES[i][1], EDGES[i][2], weight=weight[EDGES[i][0] - 1]
                   , styl='solid')
    return g


def Loss_weight_nx(weight):
    CHARGES_weight = [[] for i in range(nombre_edge)]
    graph = create_nx(weight)

    for row in DEMANDE:

        nodeA = int(row[0])

        nodeB = int(row[1])

        demande = row[2]

        paths_unique = nx.all_shortest_paths(graph, nodeA, nodeB, weight='weight')
        paths_unique = list(paths_unique)
        nombre_passage = [0 for i in range(nombre_edge)]
        lenpath = len(paths_unique)

        for path in paths_unique:

            for i in range(len(path) - 1):
                nodeprev, nodesuiv = path[i], path[i + 1]

                for rowa in EDGES:
                    if ((nodeprev == rowa[1] and nodesuiv == rowa[2])
                            or (nodeprev == rowa[2] and nodesuiv == rowa[1])):
                        nombre_passage[rowa[0] - 1] += 1

        for path1 in paths_unique:

            for i in range(len(path1) - 1):
                nodeprev, nodesuiv = path1[i], path1[i + 1]

                for rowa in EDGES:
                    if ((nodeprev == rowa[1] and nodesuiv == rowa[2])
                            or (nodeprev == rowa[2] and nodesuiv == rowa[1])):
                        CHARGES_weight[rowa[0] - 1].append(demande * nombre_passage[rowa[0] - 1] / lenpath)

    charges_tot_weight = []

    for charge in CHARGES_weight:
        charges_tot_weight.append(int(10 * sum(charge)) / 10)

    charge_capacite = []
    for i in range(len(charges_tot_weight)):
        charge_capacite.append(int(10 * charges_tot_weight[i] / EDGES[i][4]) / 10)

    MAX = max(charge_capacite)
    iMAX = charge_capacite.index(MAX)

    return iMAX, MAX, charge_capacite, graph


##############################################################################################

def afficher_graphe(charges_tot_weight, graph):
    G1 = graph
    for i in range(nombre_noeuds):
        G1.add_node(i, label=NODES[i][2], col='blue')

    for i in range(nombre_edge):
        G1.add_edge(EDGES[i][1], EDGES[i][2], weight=charges_tot_weight[EDGES[i][0] - 1]
                    , styl='solid', col='black')
    node_list=list(range(nombre_noeuds))
    pos = {i: (float(NODES[i][5]), float(NODES[i][4]))
           for i in range(nombre_noeuds)}
    print(pos)
    liste = list(G1.nodes(data='col'))

    colorNodes = {}
    for noeud in liste:
        colorNodes[noeud[0]] = noeud[1]

    colorList = [colorNodes[node] for node in colorNodes]

    liste = list(G1.nodes(data='label'))
    labels_nodes = {}
    for noeud in liste:
        labels_nodes[noeud[0]] = noeud[1]

    labels_edges = {}
    labels_edges = {edge: G1.edges[edge]['weight'] for edge in G1.edges}

    # nodes
    nx.draw_networkx_nodes(G1,pos,nodelist=node_list, node_size=100, node_color=colorList, alpha=0.9)

    maxy = max(charges_tot_weight)
    imax = charges_tot_weight.index(maxy)

    miny = min(charges_tot_weight)
    imin = charges_tot_weight.index(miny)

    colorEdge = []
    for i in range(nombre_edge):
        if i == imax:
            colorEdge.append('red')
        elif i == imin:
            colorEdge.append('blue')
        else:
            colorEdge.append('black')

    # labels
    nx.draw_networkx_labels(G1, pos, labels=labels_nodes,
                            font_size=15,
                            font_color='black',
                            font_family='sans-serif')

    # edges
    nx.draw_networkx_edges(G1, pos, width=3, edge_color=colorEdge)
    nx.draw_networkx_edge_labels(G1, pos, edge_labels=labels_edges, font_color='red')
    plt.axis('off')
    plt.savefig('fig1.png')

    plt.show()


###############################################################################################

# INITIALEMENT

indice, Loss, charge_capa, graph = Loss_weight_nx(WEIGHTS_global)

print(WEIGHTS_global)
print(indice, Loss)
#afficher_graphe(charge_capa, graph)


###############################################################################################


###########################RANDOM METHOD n°else######################################################

####################################################################################################################
#########################methode min max simple (1)####################################################################


def Loss_weight_nx_2(weight):
    CHARGES_weight = [[] for i in range(nombre_edge)]
    graph = create_nx(weight)

    for row in DEMANDE:

        nodeA = int(row[0])

        nodeB = int(row[1])

        demande = row[2]

        paths_unique = nx.all_shortest_paths(graph, nodeA, nodeB, weight='weight')
        paths_unique = list(paths_unique)
        nombre_passage = [0 for i in range(nombre_edge)]
        lenpath = len(paths_unique)

        for path in paths_unique:

            for i in range(len(path) - 1):
                nodeprev, nodesuiv = path[i], path[i + 1]

                for rowa in EDGES:
                    if ((nodeprev == rowa[1] and nodesuiv == rowa[2])
                            or (nodeprev == rowa[2] and nodesuiv == rowa[1])):
                        nombre_passage[rowa[0] - 1] += 1

        for path1 in paths_unique:

            for i in range(len(path1) - 1):
                nodeprev, nodesuiv = path1[i], path1[i + 1]

                for rowa in EDGES:
                    if ((nodeprev == rowa[1] and nodesuiv == rowa[2])
                            or (nodeprev == rowa[2] and nodesuiv == rowa[1])):
                        CHARGES_weight[rowa[0] - 1].append(demande * nombre_passage[rowa[0] - 1] / lenpath)

    charges_tot_weight = []

    for charge in CHARGES_weight:
        charges_tot_weight.append(int(10 * sum(charge)) / 10)

    charge_capacite = []
    for i in range(len(charges_tot_weight)):
        charge_capacite.append(int(10 * charges_tot_weight[i] / EDGES[i][4]) / 10)

    MAX = max(charge_capacite)
    iMAX = charge_capacite.index(MAX)
    MIN = min(charge_capacite)
    iMIN = charge_capacite.index(MIN)
    return iMAX, MAX, iMIN, MIN, charge_capacite, graph


def un_pas_min_max(weight, n):
    res = weight.copy()
    i1, L1, i2, L2, ch, gr = Loss_weight_nx_2(weight)
    res[i1] += 1 / n
    res[i2] = max(1, res[i2] - 1 / n)
    return res


##### Méthode N max (on augmente le poids de N max) (2)###############################"


def N_max_elements(charge, N):
    result_list = []
    l = charge.copy()

    for i in range(0, N):
        maximum = 0

        for j in range(len(l)):
            if l[j] > maximum:
                maximum = l[j]
        index = charge.index(maximum)
        l.remove(maximum)
        result_list.append((index, maximum))

    return result_list


def N_min_elements(charge, N):
    result_list = []
    l = charge.copy()

    for i in range(0, N):
        minimum = 0

        for j in range(len(l)):
            if l[j] < minimum:
                minimum = l[j]
        index = charge.index(minimum)
        l.remove(minimum)
        result_list.append((index, minimum))

    return result_list


def Loss_weight_nx_3(weight, N):
    CHARGES_weight = [[] for i in range(nombre_edge)]
    graph = create_nx(weight)

    for row in DEMANDE:

        nodeA = int(row[0])

        nodeB = int(row[1])

        demande = row[2]

        paths_unique = nx.all_shortest_paths(graph, nodeA, nodeB, weight='weight')
        paths_unique = list(paths_unique)
        nombre_passage = [0 for i in range(nombre_edge)]
        lenpath = len(paths_unique)

        for path in paths_unique:

            for i in range(len(path) - 1):
                nodeprev, nodesuiv = path[i], path[i + 1]

                for rowa in EDGES:
                    if ((nodeprev == rowa[1] and nodesuiv == rowa[2])
                            or (nodeprev == rowa[2] and nodesuiv == rowa[1])):
                        nombre_passage[rowa[0] - 1] += 1

        for path1 in paths_unique:

            for i in range(len(path1) - 1):
                nodeprev, nodesuiv = path1[i], path1[i + 1]

                for rowa in EDGES:
                    if ((nodeprev == rowa[1] and nodesuiv == rowa[2])
                            or (nodeprev == rowa[2] and nodesuiv == rowa[1])):
                        CHARGES_weight[rowa[0] - 1].append(demande * nombre_passage[rowa[0] - 1] / lenpath)

    charges_tot_weight = []

    for charge in CHARGES_weight:
        charges_tot_weight.append(int(10 * sum(charge)) / 10)

    charge_capacite = []
    for i in range(len(charges_tot_weight)):
        charge_capacite.append(int(10 * charges_tot_weight[i] / EDGES[i][4]) / 10)

    MAX = max(charge_capacite)
    iMAX = charge_capacite.index(MAX)
    res_max = N_max_elements(charge_capacite, N)
    res_min = N_min_elements(charge_capacite, N)
    return iMAX, MAX, res_max, res_min, charge_capacite


def un_pas_N_max(weight, N, n):
    res = weight.copy()
    imax, maxi, list_max, liste_min, ch = Loss_weight_nx_3(weight, N)
    for x in list_max:
        res[x[0]] += 1 / n

    for x in liste_min:
        res[x[0]] = max(1, res[x[0]] - 1 / n)
    return res


###### Méthode epsilon (on augmente le poids de tous ceux proche du max à epsilon près) (3)


def Loss_weight_nx_eps(weight, eps):
    CHARGES_weight = [[] for i in range(nombre_edge)]
    graph = create_nx(weight)

    for row in DEMANDE:

        nodeA = int(row[0])

        nodeB = int(row[1])

        demande = row[2]

        paths_unique = nx.all_shortest_paths(graph, nodeA, nodeB, weight='weight')
        paths_unique = list(paths_unique)
        nombre_passage = [0 for i in range(nombre_edge)]
        lenpath = len(paths_unique)

        for path in paths_unique:

            for i in range(len(path) - 1):
                nodeprev, nodesuiv = path[i], path[i + 1]

                for rowa in EDGES:
                    if ((nodeprev == rowa[1] and nodesuiv == rowa[2])
                            or (nodeprev == rowa[2] and nodesuiv == rowa[1])):
                        nombre_passage[rowa[0] - 1] += 1

        for path1 in paths_unique:

            for i in range(len(path1) - 1):
                nodeprev, nodesuiv = path1[i], path1[i + 1]

                for rowa in EDGES:
                    if ((nodeprev == rowa[1] and nodesuiv == rowa[2])
                            or (nodeprev == rowa[2] and nodesuiv == rowa[1])):
                        CHARGES_weight[rowa[0] - 1].append(demande * nombre_passage[rowa[0] - 1] / lenpath)

    charges_tot_weight = []

    for charge in CHARGES_weight:
        charges_tot_weight.append(int(10 * sum(charge)) / 10)

    charge_capacite = []
    for i in range(len(charges_tot_weight)):
        charge_capacite.append(int(10 * charges_tot_weight[i] / EDGES[i][4]) / 10)

    MAX = max(charge_capacite)
    iMAX = charge_capacite.index(MAX)
    MIN = min(charge_capacite)
    iMIN = charge_capacite.index(MIN)
    tab_max = []
    tab_min = []
    for i in range(len(charge_capacite)):
        if charge_capacite[i] >= MAX - eps:
            tab_max.append((i, charge_capacite[i]))
        if charge_capacite[i] <= MIN + eps:
            tab_min.append((i, charge_capacite[i]))

    return MAX, iMAX, tab_max, tab_min, charge_capacite


def un_pas_max_eps(weight, eps, n):
    res = weight.copy()
    M, iM, tab_max, tab_min, ch = Loss_weight_nx_eps(weight, eps)
    for x in tab_max:
        res[x[0]] += 1 / n
    for y in tab_min:
        res[y[0]] = max(1, res[y[0]] - 1 / n)
    return res


################################################################################################################
########################### METHODE D4ARRETES VOISINES###################### (4)


def trouver_voisins(arrete):
    nodeA = EDGES[arrete - 1][1]
    nodeB = EDGES[arrete - 1][2]
    voisins = []
    for row in EDGES:
        if row[0] != arrete:
            if nodeA == row[1] or nodeA == row[2] or nodeB == row[1] or nodeB == row[2]:
                voisins.append(row[0])
    return voisins


def Loss_weight_nx_voisins(weight):
    CHARGES_weight = [[] for i in range(nombre_edge)]
    graph = create_nx(weight)

    for row in DEMANDE:

        nodeA = int(row[0])

        nodeB = int(row[1])

        demande = row[2]

        paths_unique = nx.all_shortest_paths(graph, nodeA, nodeB, weight='weight')
        paths_unique = list(paths_unique)
        nombre_passage = [0 for i in range(nombre_edge)]
        lenpath = len(paths_unique)

        for path in paths_unique:

            for i in range(len(path) - 1):
                nodeprev, nodesuiv = path[i], path[i + 1]

                for rowa in EDGES:
                    if ((nodeprev == rowa[1] and nodesuiv == rowa[2])
                            or (nodeprev == rowa[2] and nodesuiv == rowa[1])):
                        nombre_passage[rowa[0] - 1] += 1

        for path1 in paths_unique:

            for i in range(len(path1) - 1):
                nodeprev, nodesuiv = path1[i], path1[i + 1]

                for rowa in EDGES:
                    if ((nodeprev == rowa[1] and nodesuiv == rowa[2])
                            or (nodeprev == rowa[2] and nodesuiv == rowa[1])):
                        CHARGES_weight[rowa[0] - 1].append(demande * nombre_passage[rowa[0] - 1] / lenpath)

    charges_tot_weight = []

    for charge in CHARGES_weight:
        charges_tot_weight.append(int(10 * sum(charge)) / 10)

    charge_capacite = []
    for i in range(len(charges_tot_weight)):
        charge_capacite.append(int(10 * charges_tot_weight[i] / EDGES[i][4]) / 10)

    MAX = max(charge_capacite)
    iMAX = charge_capacite.index(MAX)
    voisins_max = trouver_voisins(iMAX)
    MIN = min(charge_capacite)
    iMIN = charge_capacite.index(MIN)
    voisins_min = trouver_voisins(iMIN)
    return MAX, iMAX, MIN, iMIN, voisins_max, voisins_min, charge_capacite, graph


def un_pas_voisins(weight, n):
    res = weight.copy()
    M, iM, m, im, tab_max, tab_min, ch, graph = Loss_weight_nx_voisins(weight)
    for x in tab_max:
        res[x - 1] += 1 / n
    for y in tab_min:
        res[y - 1] = max(1, res[y - 1] - 1 / n)
    return res


##################################################################################################

def METHODE(numero_methode, n, k, graph):
    LOSS = [Loss]
    W = [WEIGHTS_global]
    CHARGE = [charge_capa]
    GRAPHE = [graph]
    weights = WEIGHTS_global

    N = 1
    eps = 0.1
    if numero_methode == 1:

        weights = WEIGHTS_global
        for i in range(k):
            weights = un_pas_min_max(weights, n)
            i1, L1, i2, L2, charge_capa1, graphi = Loss_weight_nx_2(weights)
            LOSS.append(L1)
            W.append(weights)
            CHARGE.append(charge_capa1)
            GRAPHE.append(graphi)

        min1 = min(LOSS)
        imin = LOSS.index(min1)
        wmin = W[imin]
        charges_min = CHARGE[imin]
        graph_min = GRAPHE[imin]

        # print(min1, wmin)

        # afficher_graphe(charges_min, graph_min)
        return min1

    elif numero_methode == 2:

        for i in range(k):
            weights = un_pas_N_max(weights, N, n)
            imax, maxi, tab_max, tab_min, ch = Loss_weight_nx_3(weights, N)
            LOSS.append(maxi)
            W.append(weights)
            CHARGE.append(ch)

        min1 = min(LOSS)
        imin = LOSS.index(min1)
        wmin = W[imin]
        charges_min = CHARGE[imin]

        # print(min1, wmin)

        # afficher_graphe(charges_min, graph)
        return min1

    elif numero_methode == 3:

        for i in range(k):
            weights = un_pas_max_eps(weights, eps, n)
            M, iM, l1, l2, ch = Loss_weight_nx_eps(weights, eps)
            W.append(weights)
            LOSS.append(M)
            CHARGE.append(ch)

        min1 = min(LOSS)
        imin = LOSS.index(min1)
        wmin = W[imin]
        charges_min = CHARGE[imin]

        # print(min1, wmin)

        # afficher_graphe(charges_min, graph)
        return min1

    elif numero_methode == 4:

        for i in range(k):
            weights = un_pas_voisins(weights, n)
            M, iM, m, im, tab_max, tab_min, ch, graph = Loss_weight_nx_voisins(weights)
            W.append(weights)
            LOSS.append(M)
            CHARGE.append(ch)

        min1 = min(LOSS)
        imin = LOSS.index(min1)
        wmin = W[imin]
        charges_min = CHARGE[imin]

        # print(min1, wmin)

        # afficher_graphe(charges_min, graph)
        return min1

    else:
        for i in range(k):
            weight = [random.randint(1, 5) for j in range(nombre_edge)]
            W.append(weight)
            ind, Loss1, charges_tot_weight2, graph = Loss_weight_nx(weight)
            CHARGE.append(charges_tot_weight2)
            GRAPHE.append(graph)
            LOSS.append(Loss1)

        min1 = min(LOSS)
        imin = LOSS.index(min1)
        wmin = W[imin]
        charges_min = CHARGE[imin]
        graph_min = GRAPHE[imin]

        # print(min1, wmin)
        # afficher_graphe(charges_min, graph_min)
        return min1


'''
print('methode rand')
METHODE(5, 1, 1000, graph)

print('methode voisins')
METHODE(4, 1, 100, graph)

print('methode eps')
METHODE(3, 1, 100, graph)

print('methode nmax')
METHODE(2, 1, 100, graph)

print('methode minmax simple')
METHODE(1, 1, 100, graph)

'''


def Test_seuil(numero_methode):
    seuil = METHODE(5, 1, 100, graph)

    methode_minmax = seuil + 1
    iter = 0
    MIN = []
    while methode_minmax > seuil:
        methode_minmax = METHODE(numero_methode, 1, 5, graph)
        iter += 1
        MIN.append(methode_minmax)
    print('iterations', iter)
    x = [i for i in range(iter)]
    plt.plot(x, MIN)

    # naming the x axis
    plt.xlabel('iteration - axis')
    # naming the y axis
    plt.ylabel('methode_minmax - axis')

    # giving a title to my graph
    plt.title('Efficacité de la methode minmax')

    # function to show the plot
    plt.show()


def Test_static(numero_methode, delta):
    seuil = METHODE(numero_methode, 1, 100, graph)

    methode_minmax = seuil + 1
    iter = 0
    MIN = []
    while abs(methode_minmax - seuil) > delta:
        seuil = methode_minmax.copy()
        methode_minmax = METHODE(numero_methode, 1, 1, graph)
        iter += 1
        MIN.append(methode_minmax)
    print('iterations', iter)
    x = [i for i in range(iter)]
    plt.plot(x, MIN)

    # naming the x axis
    plt.xlabel('iteration - axis')
    # naming the y axis
    plt.ylabel('methode_minmax - axis')

    # giving a title to my graph
    plt.title('Efficacité de la methode minmax')

    # function to show the plot
    plt.show()


def Test_temps(numero_methode, itermax):
    iter = 0
    TEMPS = []
    while iter < itermax:
        t1 = time.time()
        methode_minmax = METHODE(numero_methode, 1, 1, graph)
        t2 = time.time()
        t = t2 - t1
        TEMPS.append(t)

    x = [i for i in range(iter)]
    plt.plot(x, TEMPS)

    # naming the x axis
    plt.xlabel('iteration - axis')
    # naming the y axis
    plt.ylabel('methode_minmax - axis')

    # giving a title to my graph
    plt.title('Efficacité de la methode minmax')

    # function to show the plot
    plt.show()
def df(i, weight):
    dweight = weight.copy()
    dweight[i] += 1
    indicedf, Lossdf,charge, charges_tot_weightdf = Loss_weight_nx(dweight)
    indice1, Loss1, charge, charges_tot_weight1 = Loss_weight_nx(weight)

    return (Lossdf - Loss1)


def grad(weight):
    gradient = []
    for i in range(nombre_edge):
        gradient.append(df(i, weight))
    return gradient


def un_pas(pas, weight):
    # passe beaucoup mieux avec pas =1/100
    weightnouv = (np.array(weight) - pas*np.array(grad(weight)))
    for i in range (len(weightnouv)):
        if (weightnouv[i]<=0):
            weightnouv[i]=1/100
    return weightnouv
def un_pas_boule(pas,v,w1,w2):
    weightnouv = np.array(w1) - pas*np.array(grad(w1)) +v*(np.array(w1)-np.array(w2))
    for i in range (len(weightnouv)):
        if (weightnouv[i]<=0):
            weightnouv[i]=1/100
    return weightnouv
def norme(w):
        som=0
        for i in range(len(w)):
            som+=w[i]*w[i]
        return math.sqrt(som)
def un_pas_sous(pas,i,w1):

    if(norme(grad(w1))==0):
        weightnouv = np.array(w1) - pas*np.array(grad(w1))
    else :
        pas=1/(i+1)
        weightnouv=np.array(w1)-(pas/norme(grad(w1)))*np.array(grad(w1))
    for i in range (len(weightnouv)):
        if (weightnouv[i]<=0):
            weightnouv[i]=1/100
    return weightnouv
#################################################################################################################
## 1612.1 est un minimum local ,et 1236.6 en 52,910 le meilleire minimum que j'ai puis atteint est 816
#weight=[int(random.randint(1,10)) +1 for i in range(len(WEIGHTS_global))]
weight=WEIGHTS_global
w2=weight.copy()
for i in range(100):

    indice, Loss, charge_capa, graph = Loss_weight_nx(weight)

    print(weight)
    print(indice, Loss)
    #w = un_pas_boule(1 / 100,0.5, weight,w2)
    w=un_pas(1/100,weight)
    #w=un_pas_sous(1/100,i,weight)
    w2 = weight.copy()
    weight = w.copy()