####### Méthode "min max"
def Loss_weight_2(weight):

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
    MIN = min(charge_capacite)
    iMIN = charge_capacite.index(MIN)



    return iMAX, MAX, iMIN, MIN, charge_capacite
  
  def un_pas_min_max(weight):
    res = weight.copy()
    i1,L1,i2,L2,ch = Loss_weight_2(weight)
    res[i1]+=1
    res[i2] = max(1,res[i2]-1)
    return res

k=100
for i in range(k):
    weights= un_pas_min_max(weights)
    i1, L1,i2,L2, charge_capa1 = Loss_weight_2(weights)
    print(weights)
    print(i1, L1)
    print(i2,L2)
afficher_graphe(weights)

##### Méthode N max (on augmente le poids de N max)
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
        result_list.append((index,maximum))
          
    return result_list


def Loss_weight_3(weight,N):

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

    res = N_max_elements(charge_capacite,N)

    return res, charge_capacite
  
  def un_pas_N_max(weight,N):
    res = weight.copy()
    list_max,ch = Loss_weight_3(weight,N)
    for x in list_max:
        res[x[0]]+=1
    return res

k=10
N=4
for i in range(k):
    weights = un_pas_N_max(weights,N)
    tab,ch = Loss_weight_3(weights, N)
    print(tab)
    print(weights)
afficher_graphe(weights) 

###### Méthode epsilon (on augmente le poids de tous ceux proche du max à epsilon près)

def Loss_weight_eps(weight,eps):

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
    MIN = min(charge_capacite)
    iMIN = charge_capacite.index(MIN)
    tab_max = []
    tab_min = []
    for i in range(len(charge_capacite)):
        if charge_capacite[i]>= MAX - eps :
            tab_max.append((i,charge_capacite[i]))
        if charge_capacite[i]<= MIN + eps:
            tab_min.append((i,charge_capacite[i]))
            
    
    return MAX,iMAX,tab_max,tab_min, charge_capacite
  
  def un_pas_max_eps(weight,eps):
    res = weight.copy()
    M,iM,tab_max,tab_min,ch = Loss_weight_eps(weight,eps)
    for x in tab_max:
        res[x[0]]+=1
    for y in tab_min:
        res[y[0]]=max(1,res[y[0]]-1)
    return res
    
k=50
eps = 0.3
for i in range(k):
    weights= un_pas_max_eps(weights,eps)
    M,iM,l1,l2,ch = Loss_weight_eps(weights,eps)
    print(weights)
    print(iM,M)
afficher_graphe(weights)  

####


