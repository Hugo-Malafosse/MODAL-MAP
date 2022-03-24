import csv
import heapq
from collections import deque
from sys import maxsize
from typing import List

import matplotlib.pyplot as plt
import networkx as nx
from mpl_toolkits.basemap import Basemap

import prioqueue
from Graphe import Graphe


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



# Driver Code

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

weights = [row_norm[i][3] for i in range(nombre_edge)]


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

###################################################################

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


    # Number of vertices
n = nombre_noeuds

    # array of vectors is used
    # to store the graph
    # in the form of an adjacency list
adj = [[] for _ in range(n)]


for row in row_norm:
    add_edge(adj, row[1], row[2])
    add_edge(adj, row[2], row[1])

    # Given source and destination


    # Function Call




def Loss_weight(weight):

    CHARGES_weight = [[] for i in range(nombre_edge)]


    for row in row_norm2:

        nodeA = int(row[0])

        nodeB = int(row[1])

        demande = row[2]

        paths_unique = print_paths(adj, n, nodeA, nodeB)



        for path in paths_unique:

            for i in range(len(path) - 1):
                nodeprev, nodesuiv = path[i], path[i + 1]

                for rowa in row_norm:
                    if ((nodeprev == rowa[1] and nodesuiv == rowa[2])
                            or (nodeprev == rowa[2] and nodesuiv == rowa[1])):
                        CHARGES_weight[rowa[0] - 1].append(demande/len(paths_unique))

    charges_tot_weight = []

    for charge in CHARGES_weight:
        charges_tot_weight.append(int(10 * sum(charge)) / 10)

    print(CHARGES_weight)
    print(charges_tot_weight)

    MAX = max(charges_tot_weight)
    iMAX = charges_tot_weight.index(MAX) + 1
    print(MAX, iMAX)
    return MAX, charges_tot_weight


Loss, charges_tot_weight = Loss_weight(weights)
###############################################################################################

G1 = nx.Graph()
for i in range(nombre_noeuds):
    G1.add_node(i, label=row_norm1[i][2], col='blue')

for i in range(nombre_edge):
    G1.add_edge(row_norm[i][1], row_norm[i][2], weight=charges_tot_weight[row_norm[i][0] - 1]
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

m = Basemap(width=12000000, height=9000000, projection='lcc',
            resolution='c', lat_1=45., lat_2=55, lat_0=50, lon_0=-107.)
# draw coastlines.
m.drawcoastlines()
# draw a boundary around the map, fill the background.
# this background will end up being the ocean color, since
# the continents will be drawn on top.
m.drawmapboundary(fill_color='aqua')
# fill continents, set lake color same as ocean color.
m.fillcontinents(color='coral', lake_color='aqua')
plt.show()
