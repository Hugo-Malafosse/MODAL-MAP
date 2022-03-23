
Blanc = 0
Noir = 1



class Graphe:

    def __init__(self, aretes):
        self.aretes = aretes
        self.init_data()

 # Nombre de sommets
    def __len__(self):
        return len(self.aretes)

 # Initialisation des champs du graphe (hormis ses arêtes)
    def init_data(self):
        n = len(self.aretes)
        self.peres = n * [-1]
        self.dist = n * [-1]

        self.couleurs = n * [Blanc]