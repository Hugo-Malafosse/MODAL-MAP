#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 30 20:27:04 2022

@author: pietro
"""
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

row_norm = []
for row in rows:
    row1 = row[0].split(';')
    row2 = []
    for s in row1:
        row2.append(int(s))

    row_norm.append(row2)



file2 = open("demande.csv")
csvreader2 = csv.reader(file2)

header2 = next(csvreader2)

rows2 = []
for row in csvreader2:
    rows2.append(row)

file2.close()

row_norm2 = []
for row in rows2:
    rowA = row[0].split(';')
    rowB = []
    for s in rowA:
        rowB.append(float(s))

    row_norm2.append(rowB)

file1 = open("noeux.csv")
csvreader1 = csv.reader(file1)

header1 = next(csvreader1)

rows1 = []
for row in csvreader1:
    rows1.append(row)

file.close()

row_norm1 = []
for row in rows1:
    row3 = row[0].split(';')
    row4 = []
    for s in row3:
        row4.append(s)

    row_norm1.append(row4)
##########################################################################################################


nombre_edge = len(row_norm)
nombre_noeuds = len(row_norm1)
nombre_demande = len(row_norm2)

#weights = [row_norm[i][3] for i in range(nombre_edge)]
weights = [random.randint(1,5)for i in range(nombre_edge)]



###########################################################################################################

##########################################################################################
def findindexedge(u, v):
    for rowa in row_norm:
        if ((u == rowa[1] and v == rowa[2])
                or (u == rowa[2] and v == rowa[1])):
            return rowa[0] - 1


def add_edge(adj: List[List[int]],
             src: int, dest: int) -> None:
    adj[src].append(dest)
    adj[dest].append(src)


# Function which finds all the paths
# and stores it in paths array
def find_paths(paths: List[List[int]], path: List[int],
               parent: List[List[int]], n: int, u: int) -> None:
    # Base Case
    if (u == -1):
        paths.append(path.copy())
        return

    # Loop for all the parents
    # of the given vertex
    for par in parent[u]:
        # Insert the current
        # vertex in path
        path.append(u)

        # Recursive call for its parent
        find_paths(paths, path, parent, n, par)

        # Remove the current vertex
        path.pop()


# Function which performs bfs
# from the given source vertex
def bfs(adj: List[List[int]],
        parent: List[List[int]], n: int,
        start: int) -> None:
    # dist will contain shortest distance
    # from start to every other vertex
    dist = [maxsize for _ in range(n)]
    q = deque()

    # Insert source vertex in queue and make
    # its parent -1 and distance 0
    q.append(start)
    parent[start] = [-1]
    dist[start] = 0

    # Until Queue is empty
    while q:
        u = q[0]
        q.popleft()
        for v in adj[u]:
            if (dist[v] > dist[u] + 1):

                # A shorter distance is found
                # So erase all the previous parents
                # and insert new parent u in parent[v]

                dist[v] = dist[u] + 1
                q.append(v)
                parent[v].clear()
                parent[v].append(u)

            elif (dist[v] == dist[u] + 1):

                # Another candidate parent for
                # shortes path found
                parent[v].append(u)


# Function which prints all the paths
# from start to end
def print_paths(adj: List[List[int]], n: int,
                start: int, end: int) -> None:
    paths = []
    path = []
    parent = [[] for _ in range(n)]

    # Function call to bfs
    bfs(adj, parent, n, start)

    # Function call to find_paths
    find_paths(paths, path, parent, n, end)
    paths_unique = []
    for v in paths:
        if v not in paths_unique:
            paths_unique.append(v)
    return paths_unique


#################################################################################################################
def create_graph(weight):
    nombre_noeuds_max = nombre_noeuds + sum(weight) - len(weight)


    # Number of vertices
    n = nombre_noeuds_max

    # array of vectors is used
    # to store the graph
    # in the form of an adjacency list
    adj = [[] for _ in range(n)]
    compteur = nombre_noeuds
    for i in range(nombre_edge):
        if weight[i] == 1:
            add_edge(adj, row_norm[i][2], row_norm[i][1])
        else:
            noeudprev = row_norm[i][1]
            for j in range(weight[i] - 1):
                noeudsuiv = compteur
                compteur+=1

                add_edge(adj, noeudsuiv, noeudprev)
                noeudprev = noeudsuiv

            add_edge(adj, row_norm[i][2], noeudprev)

        # Given source and destination

    return adj, n
    # Function Call

def normalisation(chemins):
    chemins_norm = []
    for chem in chemins:
        chem_norm = []
        for node in chem:
            if node < nombre_noeuds:
                chem_norm.append(node)
        chemins_norm.append(chem_norm)
    return chemins_norm





def Loss_weight(weight):

    CHARGES_weight = [[] for i in range(nombre_edge)]
    adj, n= create_graph(weight)
    for row in row_norm2:

        nodeA = int(row[0])

        nodeB = int(row[1])

        demande = row[2]

        paths_unique = print_paths(adj, n, nodeA, nodeB)
        paths_unique = normalisation(paths_unique)

        for path in paths_unique:

            for i in range(len(path) - 1):
                nodeprev, nodesuiv = path[i], path[i + 1]

                for rowa in row_norm:
                    if ((nodeprev == rowa[1] and nodesuiv == rowa[2])
                            or (nodeprev == rowa[2] and nodesuiv == rowa[1])):
                        CHARGES_weight[rowa[0] - 1].append(demande / len(paths_unique))

    charges_tot_weight = []

    for charge in CHARGES_weight:
        charges_tot_weight.append(int(10 * sum(charge)) / 10)

    charge_capacite = []
    for i in range(len(charges_tot_weight)):
        charge_capacite.append(int(10 * charges_tot_weight[i] / row_norm[i][4]) / 10)

    MAX = max(charge_capacite)
    iMAX = charge_capacite.index(MAX)


    return iMAX, MAX, charge_capacite


##############################################################################################

def afficher_graphe(weight):
    indice, Loss, charges_tot_weight = Loss_weight(weight)
    G1 = nx.Graph()
    for i in range(nombre_noeuds):
        G1.add_node(i, label=row_norm1[i][2], col='blue')

    for i in range(nombre_edge):
        G1.add_edge(row_norm[i][1]-1, row_norm[i][2]-1, weight=charges_tot_weight[row_norm[i][0] - 1]
                    , styl='solid')

    pos = {i: (float(row_norm1[i][5]), float(row_norm1[i][4]))
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

indice, Loss, charge_capa = Loss_weight(weights)


print(weights)
print(indice, Loss)
afficher_graphe(weights)



###############################################################################################

def df(i, weight,d):
    dweight = weight.copy()
    dweight[i] += d
    indicedf, Lossdf, charges_tot_weightdf = Loss_weight(dweight)
    indice1, Loss1, charges_tot_weight1 = Loss_weight(weight)

    return (Lossdf - Loss1)


def grad(weight,d):
    gradient = []
    for i in range(nombre_edge):
        gradient.append(df(i, weight,d))
    
    return gradient


def un_pas(weight,d):
    res = weight.copy()
    i1, L1, ch1 = Loss_weight(weight)
    a=0
    for p in range(10):
        weightnouv = np.array(weight) - p * np.array(grad(weight,d))
        weightint = [max(1,round(weightnouv[i])) for i in range(len(weightnouv))]
        i2, L2, ch2 = Loss_weight(weightint)
        if L2 < L1 :
            res = weightint
            L1 = L2
            a = p
    return res , a


#################################################################################################################
'''
LOSS = []
W = []

W.append(weights)
LOSS.append(Loss)
k = 1000
for i in range(k):
    weight = [random.randint(1, 5) for j in range(nombre_edge)]
    W.append(weight)
    ind, Loss1, charges_tot_weight2 = Loss_weight(weight)
    LOSS.append(Loss1)

min = min(LOSS)
imin = LOSS.index(min)
wmin = W[imin]

print(W)
print(LOSS)
print(min, wmin)
afficher_graphe(wmin)

'''
'''
k=10
for i in range(k):
    tab, a = un_pas(weights,1)
    weights = tab
    indice1, Loss1, charge_capa1 = Loss_weight(weights)
    print(weights)
    print(indice1, Loss1)
afficher_graphe(weights)
'''
a = 1
while a != 0:
    tab, b = un_pas(weights,1)
    weights = tab
    a = b
    indice1, Loss1, charge_capa1 = Loss_weight(weights)
    print(weights)
    print(indice1, Loss1)
    print(a)
afficher_graphe(weights)





#################################################################################################################


'''m = Basemap(width=12000000, height=9000000, projection='lcc',
            resolution='c', lat_1=45., lat_2=55, lat_0=50, lon_0=-107.)
# draw coastlines.
m.drawcoastlines()
# draw a boundary around the map, fill the background.
# this background will end up being the ocean color, since
# the continents will be drawn on top.
m.drawmapboundary(fill_color='aqua')
# fill continents, set lake color same as ocean color.
m.fillcontinents(color='coral', lake_color='aqua')
plt.show()'''


def constru_graphe():
    graphe = list()
    for i in range(nombre_noeuds):
        graphe.append([])

    for row in row_norm:
        graphe[row[1]-1].append((row[2]-1, row[3]-1))
        graphe[row[2]-1].append((row[1]-1, row[3]-1))

    return graphe


Blanc = 0
Noir = 1


def insertOrReplace(minHeap, element, weight):
    # Insert if does not exist
    if element not in [x[1] for x in minHeap]:
        heapq.heappush(minHeap, (weight, element))

    # Replace otherwise
    else:
        indexToUpdate = [x[1] for x in minHeap].index(element)
        minHeap[indexToUpdate] = (weight, element)
        heapq.heapify(minHeap)


# Dijkstra takes as an input a source node and a graph
# It outputs the lengths of the shortest paths from the initial node to the others
def dijkstra2(graph, sourceNode):
    # We first initialize the useful data structures
    # Distances is the result of the algorithm, with the lengths of the path from the source node to node i
    # MinHeap is a min-heap that will store the nodes to visit and the associated weight
    # If a node is not in minHeap, it has been visited
    # We initialize the algorithm with the source node at distance 0
    distances = [[float("inf"), []] for i in range(len(graph))]
    minHeap = [(0, sourceNode)]
    distances[sourceNode] = [0, []]
    liste_edge = []
    # Main loop
    while len(minHeap) != 0:

        # We extract the closest node from the heap
        (closestNodeDistance, closestNode) = heapq.heappop(minHeap)

        liste_edge = []
        # We update the distance to the neighbors of this node
        for (neighbor, weight) in graph[closestNode]:
            liste_edge.append(closestNode)
            neighborDistance = closestNodeDistance + weight

            if neighborDistance < distances[neighbor][0]:
                liste_edge.append(neighbor)
                insertOrReplace(minHeap, neighbor, neighborDistance)
                distances[neighbor] = [neighborDistance, liste_edge]

    # We return the distances
    return distances


def dijkstra(graph, sourceNode):
    # We first initialize the useful data structures
    # Distances is the result of the algorithm, with the lengths of the path from the source node to node i
    # MinHeap is a min-heap that will store the nodes to visit and the associated weight
    # If a node is not in minHeap, it has been visited
    # We initialize the algorithm with the source node at distance 0
    distances = [float("inf") for i in range(len(graph))]
    minHeap = [(0, sourceNode)]
    distances[sourceNode] = 0

    # Main loop
    while len(minHeap) != 0:

        # We extract the closest node from the heap
        (closestNodeDistance, closestNode) = heapq.heappop(minHeap)

        # We update the distance to the neighbors of this node
        for (neighbor, weight) in graph[closestNode]:

            neighborDistance = closestNodeDistance + weight

            if neighborDistance < distances[neighbor]:
                insertOrReplace(minHeap, neighbor, neighborDistance)
                distances[neighbor] = neighborDistance

    # We return the distances
    return distances


graphe_arrete = constru_graphe()

gex = Graphe(graphe_arrete)


def visiter_voisins(G, u, f):
    for (v, d) in G.aretes[u]:
        if G.couleurs[v] == Blanc and (G.dist[v] > G.dist[u] + d or G.dist[v] == -1):
            G.peres[v] = u
            G.dist[v] = G.dist[u] + d
            if not f.is_in(v):
                f.push(v, G.dist[v])
            else:
                f.decrease_prio(v, G.dist[v])


def dijkstra3(G, r):
    f = prioqueue.Prioqueue()
    G.init_data()
    G.dist[r] = 0
    f.push(r, 0)
    while not (f.is_empty()):
        (u, p) = f.pop()
        G.couleurs[u] = Noir
        visiter_voisins(G, u, f)
    return (G.peres, G.dist)


p, d = dijkstra3(gex, 0)


def chemin(peres, u, r):
    c = []
    while u != -1 and u != r:
        c.append(u)
        u = peres[u]
    if u == r:
        c.append(r)
        return c
    else:
        raise Exception


def tous_chemins(peres, r):
    n = len(peres)
    cs = []
    for i in range(n):
        if peres[i] != -1:
            cs.append(chemin(peres, i, r))
    return cs


#################################################################################################################
#################################################################################################################

#################################################################################################################


'''''######################################################################################################
def distance(chemin):
    dist = 0
    for i in range(len(chemin) - 1):
        nodeprev = chemin[i]
        nodesuiv = chemin[i + 1]
        for rowa in row_norm:
            if ((nodeprev == rowa[1] and nodesuiv == rowa[2])
                    or (nodeprev == rowa[2] and nodesuiv == rowa[1])):
                dist += rowa[3]
    return dist
CHARGES_weight = [[] for i in range(nombre_edge)]
chemins = []
g = Graph2(nombre_noeuds)
for row in row_norm:
    g.addEdge(row[1], row[2])
    g.addEdge(row[2], row[1])
for row in row_norm2:
    nodeA = int(row[0])
    nodeB = int(row[1])
    demande = row[2]
    chemin = []
    g.printAllPaths(nodeA, nodeB, chemin)
    distchem = []
    for chem in chemin:
        distchem.append(distance(chem))
    cheminmin = []
    mindist = min(distchem)
    for i in range(len(distchem)):
        if distchem[i] == mindist:
            cheminmin.append(chemin[i])
    n = len(cheminmin)
    for chem in cheminmin:
        for j in range(len(chem)-1):
            nodeprev, nodesuiv = chem[j], chem[j + 1]
            for rowa in row_norm:
                if ((nodeprev == rowa[1] and nodesuiv == rowa[2])
                        or (nodeprev == rowa[2] and nodesuiv == rowa[1])):
                    CHARGES_weight[rowa[0] - 1].append(demande / n)
charges_tot_weight = []
for charge in CHARGES_weight:
    charges_tot_weight.append(int(10 * sum(charge)) / 10)
print(charges_tot_weight)
MAX = max(charges_tot_weight)
iMAX = charges_tot_weight.index(MAX) + 1
print(MAX, iMAX)'''

''''###############################################################################################
def constru_graphe_weight(weight):
    graphe = list()
    for i in range(nombre_noeuds):
        graphe.append([])
    for i, row in enumerate(row_norm):
        graphe[row[1]].append((row[2], weight[i]))
        graphe[row[2]].append((row[1], weight[i]))
    return graphe
def Loss_weight(weight):
    graphe_arrete_weight = constru_graphe_weight(weight)
    CHARGES_weight = [[] for i in range(nombre_edge)]
    gex_weight = Graphe(graphe_arrete_weight)
    for row in row_norm2:
        nodeA = int(row[0])
        nodeB = int(row[1])
        demande = row[2]
        p1, d1 = dijkstra3(gex_weight, nodeA)
        cheminB = chemin(p1, nodeB, nodeA)
        for i in range(len(cheminB) - 1):
            nodeprev, nodesuiv = cheminB[i], cheminB[i + 1]
            for rowa in row_norm:
                if ((nodeprev == rowa[1] and nodesuiv == rowa[2])
                        or (nodeprev == rowa[2] and nodesuiv == rowa[1])):
                    CHARGES_weight[rowa[0] - 1].append(demande)
    charges_tot_weight = []
    for charge in CHARGES_weight:
        charges_tot_weight.append(int(10 * sum(charge)) / 10)
    print(CHARGES_weight)
    print(charges_tot_weight)
    MAX = max(charges_tot_weight)
    iMAX = charges_tot_weight.index(MAX) + 1
    print(MAX, iMAX)
    return MAX, charges_tot_weight
Loss, charges_tot_weight = Loss_weight(weights)'''
###############################################################################################