# coding: UTF-8

# Files de priorité min
# Marc Lorenzi
# Février 2018

class PrioqueueError(Exception): pass

class Empty_Queue(PrioqueueError):
        def __init__(self): pass


# Echange les éléents d'indices i et j dans la structure t

def exchange_struct(t, i, j):
    t[i], t[j] = t[j], t[i]

# père, fils gauche, fils droit

def father(k): return (k - 1) // 2
def left(k): return 2 * k + 1
def right(k): return  2 * k + 2

# --------------------------------------------

class Prioqueue:

    def __init__(self):
        self.size = 0
        self.data = []
        self.prio = []
        self.index = dict([])
        self.nbpush = 0
        self.nbpop = 0
        self.bigsize = 0

    def reset_stats(self):
        self.nbpush = 0
        self.nbpop = 0
        self.bigsize = 0

    def get_stats(self):
        return (self.nbpush, self.nbpop, self.bigsize)

    # La file est-elle vide ?
    def is_empty(self): return self.size == 0

    # l'objet x est-il dans la file ?
    def is_in(self, x):
        return x in self.index

    # index de l'objet x dans la file
    def index_of(self, x):
        return self.index[x]

    # Taille de la file
    def length(self): return self.size
 
    # Echange les éléments d'indices i et j dans la file

    def exchange(self, i, j):
        exchange_struct(self.index, self.data[i], self.data[j])
        exchange_struct(self.data, i, j)
        exchange_struct(self.prio, i, j)

    # Elément de priorité minimale dans la file

    def top(self): 
        if self.size > 0:    
            return (self.data[0], self.prio[0])
        else: raise Empty_Queue


    # Mettre à la bonne position dans la file un élément dont la
    # priorité risque d'être plus grande que celle de l'un de ses fils

    def bubble_down(self, k):
        m = k
        if left(k) < self.size and self.prio[left(k)] < self.prio[k]:
            m = left(k)
        if right(k) < self.size and self.prio[right(k)] < self.prio[m]:
            m = right(k)
        if m != k:
            self.exchange(m, k)
            self.bubble_down(m)
  
    # Mettre à la bonne position dans la file un élément dont la
    # priorité risque d'être plus petite que celle de son père

    def bubble_up(self, k):
        m = k
        if m > 0 and self.prio[father(m)] > self.prio[m]: 
            m = father(k)
        if k != m:
            self.exchange(k, m)
            self.bubble_up(m)


    # Supprime de la file l'élément de priorité minimale.
    # Renvoie cet élement ainsi que sa priorité

    def pop(self):
        if self.size == 0: raise Empty_Queue
        else:
            self.nbpop = self.nbpop + 1
            x = self.data[0]
            p = self.prio[0]
            # print("size : ", self.size, " - popping ", x, p)
            del self.index[x]
            self.data[0] = self.data[self.size - 1]
            self.prio[0] = self.prio[self.size - 1]
            if self.size > 1: self.index[self.data[0]] = 0
            self.data.pop()
            self.prio.pop()
            self.size = self.size - 1
            self.bubble_down(0)
        return (x, p)

    # Ajoute à la file l'objet x de priorité p

    def push(self, x, p):
        self.nbpush = self.nbpush + 1
        self.size = self.size + 1
        if self.size > self.bigsize: self.bigsize = self.size       
        self.data.append(x)
        self.prio.append(p)
        self.index[x] = self.size - 1
        self.bubble_up(self.size - 1)

    # Diminue la priorité de l'objet x à la valeur p
    # Ne fait rien si la nouvelle priorité est supérieure
    # à l'ancienne

    def decrease_prio(self, x, p):
        k = self.index[x]
        #print(self.prio, self.data, self.index)
        #print(k)
        if self.prio[k] > p:
            self.prio[k] = p
            self.bubble_up(k)
 
    # Augmente la priorité de l'objet x à la valeur p
    # Ne fait rien si la nouvelle priorité est inférieure
    # à l'ancienne

    def increase_prio(self, x, p):
        k = self.index[x]
        if self.prio[k] < p:
            self.prio[k] = p
            self.bubble_down(k)

    # Modifie la priorité de l'objet x à la valeur p      
    # Ne fait rien si la nouvelle priorité est identique
    # à l'ancienne

    def modify_prio(self, x, p):
        k = self.index[x]
        if self.prio[k] < p: self.increase_prio(x, p)
        elif self.prio[k] > p: self.decrease_prio(x, p)

