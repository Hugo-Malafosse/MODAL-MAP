import prioq
import utils
import csv
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from prioq.base import PriorityQueue
import heapq
import csv
from collections import defaultdict
from mpl_toolkits.basemap import Basemap

import matplotlib.pyplot as plt
import networkx as nx
from numpy import array
import prioqueue
import Graphe
from Graphe import Graphe




file = open("edge.csv")
csvreader = csv.reader(file)

header = next(csvreader)


nombre_noeuds = 37
nombre_edge = 57
nombre_demande = 72

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


print(row_norm)

def constru_graphe():
    graphe = list()
    for i in range(nombre_noeuds):
        graphe.append([])

    for row in row_norm:
        graphe[row[1]].append((row[2], row[3]))
        graphe[row[2]].append((row[1], row[3]))

    return graphe



graphe_arrete = constru_graphe()


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


###################################################################

# Dijkstra takes as an input a source node and a graph
# It outputs the lengths of the shortest paths from the initial node to the others
def dijkstra2(graph, sourceNode):
    # We first initialize the useful data structures
    # Distances is the result of the algorithm, with the lengths of the path from the source node to node i
    # MinHeap is a min-heap that will store the nodes to visit and the associated weight
    # If a node is not in minHeap, it has been visited
    # We initialize the algorithm with the source node at distance 0
    distances = [[float("inf"),[]] for i in range(len(graph))]
    minHeap = [(0, sourceNode)]
    distances[sourceNode] = [0,[]]
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


gex = Graphe(graphe_arrete)



def visiter_voisins(G, u, f):
    for (v, d) in G.aretes[u]:
        if G.couleurs[v] == Blanc and (G.dist[v] > G.dist[u] + d or G.dist[v]== -1):
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
    else: raise Exception

def tous_chemins(peres, r):
    n = len(peres)
    cs = []
    for i in range(n):
        if peres[i] != -1:
            cs.append(chemin(peres, i, r))
    return cs



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


print(row_norm2)
print(row_norm)


CHARGES = [[] for i in range(nombre_edge)]

for row in row_norm2:

    nodeA = int(row[0])
    nodeB = int(row[1])


    demande = row[2]


    p1, d1 = dijkstra3(gex, nodeA)

    cheminB = chemin(p1, nodeB, nodeA)
    for i in range(len(cheminB)-1):
        nodeprev, nodesuiv = cheminB[i], cheminB[i+1]
        for rowa in row_norm :
            if ((nodeprev == rowa[1] and nodesuiv == rowa[2])
                    or (nodeprev == rowa[2] and nodesuiv == rowa[1])):
                CHARGES[rowa[0]-1].append(demande)


charges_tot = []

'''sum1 = 0
for edge in CHARGES:
    for demandeedge in edge :
        sum1 +=1
print(sum1)'''

for charge in CHARGES:
    charges_tot.append(int(10*sum(charge))/10)


print(CHARGES)
print(charges_tot)

MAX = max(charges_tot)
iMAX = charges_tot.index(MAX) +1
print(MAX, iMAX)




G1 = nx.Graph()
for i in range(nombre_noeuds):
    G1.add_node(i, label=row_norm1[i][2], col='blue')

for i in range(nombre_edge):
    G1.add_edge(row_norm[i][1], row_norm[i][2], weight=charges_tot[row_norm[i][0]-1]
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

m = Basemap(width=12000000,height=9000000,projection='lcc',
            resolution='c',lat_1=45.,lat_2=55,lat_0=50,lon_0=-107.)
# draw coastlines.
m.drawcoastlines()
# draw a boundary around the map, fill the background.
# this background will end up being the ocean color, since
# the continents will be drawn on top.
m.drawmapboundary(fill_color='aqua')
# fill continents, set lake color same as ocean color.
m.fillcontinents(color='coral',lake_color='aqua')
plt.show()



