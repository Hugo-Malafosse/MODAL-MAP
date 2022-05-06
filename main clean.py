import csv
import heapq
import random
from collections import deque
from sys import maxsize
from typing import List

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from mpl_toolkits.basemap import Basemap

import prioqueue
from Graphe import Graphe

######################################################################################################
# Driver Code

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
                    , styl='solid')

    pos = {i: (float(NODES[i][5]), float(NODES[i][4]))
           for i in range(nombre_noeuds)}

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
    nx.draw_networkx_nodes(G1, pos, node_size=100, node_color=colorList, alpha=0.9)

    # labels
    nx.draw_networkx_labels(G1, pos, labels=labels_nodes,
                            font_size=15,
                            font_color='black',
                            font_family='sans-serif')

    # edges
    nx.draw_networkx_edges(G1, pos, width=3)
    nx.draw_networkx_edge_labels(G1, pos, edge_labels=labels_edges, font_color='red')
    plt.axis('off')
    plt.savefig('fig1.png')

    plt.show()


###############################################################################################

# INITIALEMENT

indice, Loss, charge_capa, graph = Loss_weight_nx(WEIGHTS_global)

print(WEIGHTS_global)
print(indice, Loss)
afficher_graphe(charge_capa, graph)

###############################################################################################
'''
def df(i, weight, d):
    dweight = weight.copy()
    dweight[i] += d
    indicedf, Lossdf, charges_tot_weightdf = Loss_weight(dweight)
    indice1, Loss1, charges_tot_weight1 = Loss_weight(weight)

    return (Lossdf - Loss1)


def grad(weight, d, pas):
    gradient = []
    for i in range(nombre_edge):
        gradient.append(pas*df(i, weight, d))
    print(gradient)
    return gradient


def un_pas(pas, weight, d):
    weightnouv = np.array(weight) - np.array(grad(weight, d, pas))
    weightint = []

    for i in range(len(weightnouv)):
        weightint.append(max(1, int(weightnouv[i])))

    return weightint'''

#################################################################################################################

LOSS = []
W = []
CHARGE = []
GRAPHE = []

W.append(WEIGHTS_global)
LOSS.append(Loss)
CHARGE.append(charge_capa)
GRAPHE.append(graph)
k = 1000
for i in range(k):
    ratio = i/k*100
    print(ratio)
    weight = [random.randint(1,5) for j in range(nombre_edge)]
    W.append(weight)
    ind, Loss1, charges_tot_weight2, graph = Loss_weight_nx(weight)
    CHARGE.append(charges_tot_weight2)
    GRAPHE.append(graph)
    LOSS.append(Loss1)

min = min(LOSS)
imin = LOSS.index(min)
wmin = W[imin]
charges_min = CHARGE[imin]
graph_min = GRAPHE[imin]



print(min, wmin)
afficher_graphe(charges_min, graph_min)


'''
k=10

d=1
lossbis = []
weightbis = []

for pas in range(1, 10):


    for i in range(k):
        weightbis1 = un_pas(pas, WEIGHTS_global, d)
        weightdif = []
        for i in range(len(weightbis1)):
            weightdif.append(weightbis1[i] - WEIGHTS_global[i])
        print(weightdif)
        WEIGHTS_global = weightbis1


    indice1, Loss1, charge_capa1 = Loss_weight(WEIGHTS_global)


    print(WEIGHTS_global)
    print(indice1, Loss1)
    afficher_graphe(WEIGHTS_global)

    lossbis.append(Loss1)
    weightbis.append(WEIGHTS_global)

min = min(lossbis)
imin = lossbis.index(min)
wmin = weightbis[imin]

print(min, wmin)'''
'''
d=1
for i in range(k):
    weights = un_pas(1, weights, d)

indice1, Loss1, charge_capa1 = Loss_weight(weights)

print(weights)
print(indice1, Loss1)
afficher_graphe(weights)

'''

#################################################################################################################
