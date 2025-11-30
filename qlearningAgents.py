# qlearningAgents.py
# ------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from game import *
from learningAgents import ReinforcementAgent
from featureExtractors import *

import random,util,math

import numpy as np
from sklearn.neural_network import MLPRegressor

class QLearningAgent(ReinforcementAgent):
    """
      Q-Learning Agent

      Functions you should fill in:
        - computeValueFromQValues
        - computeActionFromQValues
        - getQValue
        - getAction
        - update

      Instance variables you have access to
        - self.epsilon (exploration prob)
        - self.alpha (learning rate)
        - self.discount (discount rate)

      Functions you should use
        - self.getLegalActions(state)
          which returns legal actions for a state
    """
    def __init__(self, **args):
        "You can initialize Q-values here..."
        ReinforcementAgent.__init__(self, **args)
        # Je stocke les Q-valeurs dans un compteur
        # Clé = (state, action). Au début tout est 0 donc inconnu.
        self.qValues = util.Counter()

    def getQValue(self, state, action):
        """Retourne Q(s,a). Si jamais vu 0. On accède via self.qValues.
        """
        return self.qValues[(state, action)]


    def computeValueFromQValues(self, state):
        """V(s) = max_a Q(s,a). Si aucune action légale c'est l'état terminal donc 0.
        On passe par getQValue pour respecter l'abstraction.
        """
        actions = self.getLegalActions(state)
        if len(actions) == 0:
            return 0.0
        best = float('-inf')
        for a in actions:
            q = self.getQValue(state, a)
            if q > best:
                best = q
        return best

    def computeActionFromQValues(self, state):
        """on utilise une politique greedy: retourne une action a qui maximise Q(s,a).
        On doit briser les égalités ALÉATOIREMENT (random.choice sur les meilleurs).
        Si pas d'actions (terminal) => None.
        """
        actions = self.getLegalActions(state)
        if len(actions) == 0:
            return None
        bestValue = float('-inf')
        bestActions = []
        for a in actions:
            q = self.getQValue(state, a)
            if q > bestValue:
                bestValue = q
                bestActions = [a]
            elif q == bestValue:
                bestActions.append(a)
        return random.choice(bestActions)

    def getAction(self, state):
        """
          Compute the action to take in the current state.  With
          probability self.epsilon, we should take a random action and
          take the best policy action otherwise.  Note that if there are
          no legal actions, which is the case at the terminal state, you
          should choose None as the action.

          HINT: You might want to use util.flipCoin(prob)
          HINT: To pick randomly from a list, use random.choice(list)
        """
        # Pick Action
        legalActions = self.getLegalActions(state)
        if len(legalActions) == 0:
            return None
        # Avec probabilité epsilon on explore (action aléatoire) cela permet d'éviter
        # de rester bloqué dans une politique sous-optimale.
        if util.flipCoin(self.epsilon):
            return random.choice(legalActions)
        else:
            return self.computeActionFromQValues(state)

    def update(self, state, action, nextState, reward):
        """Mise à jour Q-learning:
        """
        oldQ = self.getQValue(state, action)
        futureValue = self.computeValueFromQValues(nextState)
        sample = reward + self.discount * futureValue
        newQ = (1 - self.alpha) * oldQ + self.alpha * sample
        self.qValues[(state, action)] = newQ

    def getPolicy(self, state):
        return self.computeActionFromQValues(state)

    def getValue(self, state):
        return self.computeValueFromQValues(state)


class PacmanQAgent(QLearningAgent):

    def __init__(self, epsilon=0.05,gamma=0.8,alpha=0.2, numTraining=0, **args):

        args['epsilon'] = epsilon
        args['gamma'] = gamma
        args['alpha'] = alpha
        args['numTraining'] = numTraining
        self.index = 0  
        QLearningAgent.__init__(self, **args)

    def getAction(self, state):
        """
        Simply calls the getAction method of QLearningAgent and then
        informs parent of action for Pacman.  Do not change or remove this
        method.
        """
        action = QLearningAgent.getAction(self,state)
        self.doAction(state,action)
        return action

class ApproximateQAgent(PacmanQAgent):
    """
    Approximate Q-Learning avec un réseau de neurones (MLPRegressor)
    qui approxime Q(s,a) à partir de features φ(s,a) (SimpleExtractor).

    - Entrée du réseau : vecteur de features φ(s,a) (numpy array 1D).
    - Sortie du réseau : un scalaire Q(s,a).
    """

    def __init__(self,
                 hidden_sizes=(32,),
                 nn_learning_rate=0.001,
                 **args):

        # Initialise la partie Q-Learning (epsilon, gamma, alpha, etc.)
        PacmanQAgent.__init__(self, **args)

        # Extracteur de features
        self.featExtractor = SimpleExtractor()
        self.featureKeys = None  # sera fixé au premier appel

        # Hyperparamètres du MLP
        self.hidden_sizes = hidden_sizes
        self.nn_learning_rate = float(nn_learning_rate)

        # Réseau de neurones pour approximer Q(s,a)
        self.nn = MLPRegressor(
            hidden_layer_sizes=self.hidden_sizes,
            activation="relu",
            solver="sgd",
            learning_rate_init=self.nn_learning_rate,
            warm_start=True,     # permet plusieurs fit successifs
            max_iter=1,          # un petit pas de gradient par fit
            random_state=0
        )
        self.nn_initialized = False

    # ----------------------- FEATURES φ(s,a) -----------------------

    def _stateActionToFeatures(self, state, action):
        """
        Utilise SimpleExtractor pour obtenir un dict de features, puis
        le convertit en vecteur numpy dans un ordre fixe.

        φ(s,a) = [bias, closest-food, eats-food, #-of-ghosts-1-step-away, ...]
        """
        feats = self.featExtractor.getFeatures(state, action)  # util.Counter

        # On fixe une fois pour toutes l'ordre des features
        if self.featureKeys is None:
            self.featureKeys = sorted(feats.keys())

        vec = np.array(
            [feats.get(k, 0.0) for k in self.featureKeys],
            dtype=float
        )
        return vec

    # ----------------------- Réseau Q(s,a) -----------------------

    def _ensureInitialized(self, phi_sa):
        """
        Initialise le réseau à partir d'un premier vecteur de features.
        On lui fait apprendre Q(s,a) = 0 juste pour fixer les dimensions.
        """
        if not self.nn_initialized:
            X = phi_sa.reshape(1, -1)
            y = np.array([0.0], dtype=float)
            self.nn.fit(X, y)
            self.nn_initialized = True

    def _predictQ_from_features(self, phi_sa):
        """
        Prédit Q(s,a) à partir des features φ(s,a).
        """
        if not self.nn_initialized:
            return 0.0
        X = phi_sa.reshape(1, -1)
        q = float(self.nn.predict(X)[0])
        return q

    def getQValue(self, state, action):
        """
        Q(s,a) = sortie du réseau pour les features φ(s,a).
        """
        if action is None:
            return 0.0
        phi_sa = self._stateActionToFeatures(state, action)
        return self._predictQ_from_features(phi_sa)

    def update(self, state, action, nextState, reward):
        """
        Mise à jour de Q via Q-learning + réseau de neurones :

          target = r + gamma * max_{a'} Q(nextState, a')

        On entraîne le réseau pour approximer cette cible à partir de φ(s,a).
        """
        if action is None:
            return

        # Features de (s,a)
        phi_sa = self._stateActionToFeatures(state, action)
        self._ensureInitialized(phi_sa)  # init réseau si nécessaire

        # Cible de Q-learning
        legal_next = self.getLegalActions(nextState)
        if len(legal_next) == 0:
            target = reward  # état terminal
        else:
            max_next = max(self.getQValue(nextState, a) for a in legal_next)
            target = reward + self.discount * max_next

        # Q(s,a) courant
        currentQ = self._predictQ_from_features(phi_sa)

        # Q_new = Q_old + alpha * (target - Q_old)
        td_target = currentQ + self.alpha * (target - currentQ)

        # On clippe la cible pour éviter les explosions numériques
        td_target = float(np.clip(td_target, -1000.0, 1000.0))

        # Si la valeur est non-finie, on saute la mise à jour
        if not np.isfinite(td_target):
            return

        # Entraînement du réseau sur cet exemple (φ(s,a), td_target)
        X = phi_sa.reshape(1, -1)
        y = np.array([td_target], dtype=float)
        self.nn.fit(X, y)

    def final(self, state):
        """
        Appelée à la fin de chaque partie.
        On garde le comportement de la classe parente.
        """
        PacmanQAgent.final(self, state)
        if self.episodesSoFar == self.numTraining:
            # Fin de l'entraînement
            pass
