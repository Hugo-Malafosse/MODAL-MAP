
import prioq
import utils
import csv
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from prioq.base import PriorityQueue
import heapq
import csv
from collections import defaultdict
import matplotlib.pyplot as plt
import networkx as nx
from numpy import array



'''class Graphe:

    def __init__(self, aretes):
        self.aretes = aretes
        self.init_data()

 # Nombre de sommets
    def __len__(self):
        return len(self.aretes)

    def init_data(self):
        n = len(self.aretes)
        self.peres = n * [-1]
        self.dist = n * [-1]
        self.couleurs = n * [Blanc]'''


'''columns = defaultdict(list) # each value in each column is appended to a list

with open('edge.csv') as f:
    reader = csv.DictReader(f) # read rows into a dictionary format
    for row in reader: # read a row as {column1: value1, column2: value2,...}
        for (k,v) in row.items(): # go over each column name and value
            columns[k].append(v) # append the value into the appropriate list
                                 # based on column name k'''

file = open("edge.csv")
csvreader = csv.reader(file)
print(file)
print(csvreader)
header = next(csvreader)
print(header)

nombre_noeuds =37
nombre_edge = 57

rows = []
for row in csvreader:
    rows.append(row)
print(rows)
file.close()

row_norm = []
for row in rows :
    row1 = row[0].split(';')
    row2 = []
    for s in row1 :
        row2.append(int(s))


    row_norm.append(row2)

print(row_norm)

def constru_graphe():
    graphe = list()
    for i in range(nombre_noeuds):
        graphe.append([])


    for row in row_norm :
        graphe[row[1]].append((row[2], row[3]))
        graphe[row[2]].append((row[1], row[3]))


    return graphe



print(constru_graphe())
graphe_arrete = constru_graphe()
print(len(graphe_arrete))




file1 = open("noeux.csv")
csvreader1 = csv.reader(file1)

header1 = next(csvreader1)
print(header1)


rows1 = []
for row in csvreader1:
    rows1.append(row)
print(rows1)
file.close()

row_norm1 = []
for row in rows1:
    row3 = row[0].split(';')
    row4 = []
    for s in row3 :
        row4.append(s)


    row_norm1.append(row4)


print(row_norm1)


G = nx.Graph()
for i in range(nombre_noeuds):
    G.add_node(i, label= row_norm1[i][2], col = 'blue')

for i in range(nombre_edge):
    G.add_edge(row_norm[i][1], row_norm[i][2], weight = row_norm[i][3], styl = 'solid')



liste = list(G.nodes(data='col'))
colorNodes = {}
for noeud in liste:
    colorNodes[noeud[0]]=noeud[1]

colorList=[colorNodes[node] for node in colorNodes]

liste = list(G.nodes(data='label'))
labels_nodes = {}
for noeud in liste:
    labels_nodes[noeud[0]]=noeud[1]

labels_edges = {}
labels_edges = {edge:G.edges[edge]['weight'] for edge in G.edges}





pos = nx.spring_layout(G)

# nodes
nx.draw_networkx_nodes(G, pos, node_size=100, node_color=colorList, alpha=0.9)

# labels
nx.draw_networkx_labels(G, pos, labels=labels_nodes,
                        font_size=15,
                        font_color='black',
                        font_family='sans-serif')

# edges
nx.draw_networkx_edges(G, pos, width=3)
nx.draw_networkx_edge_labels(G, pos, edge_labels=labels_edges, font_color='red')

plt.axis('off')
plt.savefig('fig1.png')
plt.show()





'''def visiter_voisins(G, u, f):
    for (v, d) in G.aretes[u]:
        if G.couleurs[v] == Blanc and (G.dist[v] > G.dist[u] + d or G.dist[v]== -1):
            G.peres[v] = u
            G.dist[v] = G.dist[u] + d
            if not f.is_in(v):
                f.push(v, G.dist[v])
            else:
                f.decrease_prio(v, G.dist[v])



def dijkstra(G, r):
    f = PriorityQueue()
    G.init_data()
    G.dist[r] = 0
    f.push(r)
    while not (f.__len__() == 0):
        (u, p) = f.pop()
        G.couleurs[u] = Noir
        visiter_voisins(G, u, f)
    return (G.peres, G.dist)


dijkstra(graphe_arrete, 3)'''


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


print(dijkstra(graphe_arrete, 0))

