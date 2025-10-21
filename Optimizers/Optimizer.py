from abc import ABC, abstractmethod

class Optimizer(ABC):
    @abstractmethod
    def update(self, params, grads):
        """
        Met à jour les paramètres du modèle en fonction des gradients.

        params : liste ou dictionnaire des paramètres (poids, biais)
        grads : liste ou dictionnaire des gradients correspondants

        Cette méthode doit être implémentée dans les sous-classes.
        """
        pass