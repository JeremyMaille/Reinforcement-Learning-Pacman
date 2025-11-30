# analysis.py
# -----------
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


######################
# ANALYSIS QUESTIONS #
######################

# Set the given parameters to obtain the specified policies through
# value iteration.

def question2():
    """on veut que l'agent ose traverser le pont.
    Par défaut (discount=0.9, noise=0.2) il évite car le risque de tomber vers
    le mauvais terminal est trop élevé.
    On ne change qu'un paramètre: ici je garde discount=0.9
    et je réduis le bruit pour rendre les actions quasi déterministes.
    Avec noise=0.0, traverser devient fiable.
    """
    answerDiscount = 0.9   # on garde le discount original
    answerNoise = 0.0      # on baisse seulement le bruit
    return answerDiscount, answerNoise

def question3a():
    """ici on préfère la sortie proche (+1) en RISQUANT la falaise (-10).
    on n'accorde pas trop d'importance au long terme en mettant un discount faible,
    et je supprime le bruit pour pouvoir couper au plus court, quitte à frôler la falaise.
    """
    answerDiscount = 0.3
    answerNoise = 0.0
    answerLivingReward = 0.0
    return answerDiscount, answerNoise, answerLivingReward
    # If not possible, return 'NOT POSSIBLE'

def question3b():
    """ on préfère la sortie proche (+1) mais en ÉVITANT la falaise.
    encore peu de poids au futur,
    mais on remet du bruit pour décourager les trajets au bord du vide.
    """
    answerDiscount = 0.3
    answerNoise = 0.2
    answerLivingReward = 0.0
    return answerDiscount, answerNoise, answerLivingReward
    # If not possible, return 'NOT POSSIBLE'

def question3c():
    """on préfère la sortie loin (+10) en risquant la falaise.
    On valorise beaucoup le futur (discount haut) et on enlève le bruit pour
    foncer tout droit, même si c'est risqué.
    """
    answerDiscount = 0.9
    answerNoise = 0.0
    answerLivingReward = 0.0
    return answerDiscount, answerNoise, answerLivingReward
    # If not possible, return 'NOT POSSIBLE'

def question3d():
    """on préfère la sortie LOINTAINE (+10) en évitant la falaise.
    On reste patient (discount haut) mais on conserve du bruit pour éviter
    les trajets trop dangereux le long de la falaise.
    """
    answerDiscount = 0.9
    answerNoise = 0.2
    answerLivingReward = 0.0
    return answerDiscount, answerNoise, answerLivingReward
    # If not possible, return 'NOT POSSIBLE'

def question3e():
    """On évite les deux sorties positives et la falaise pour ne jamais finir.
    On donne une récompense de vie positive: vivre rapporte, donc on traîne
    au lieu de sortir; discount haut pour apprécier de rester en vie.
    """
    answerDiscount = 0.9
    answerNoise = 0.2
    answerLivingReward = 1.0
    return answerDiscount, answerNoise, answerLivingReward
    # If not possible, return 'NOT POSSIBLE'

def question6():
    """BridgeGrid avec 50 épisodes: demander >99% de chances de trouver la
    politique optimale avec un simple Q-learning est trop fort.
    Avec l'aléa des transitions/exploration et un horizon si court, on ne
    peut pas garantir une proba > 99% pour des hyperparamètres fixes.
    """
    return 'NOT POSSIBLE'

if __name__ == '__main__':
    print ('Answers to analysis questions:')
    import analysis
    for q in [q for q in dir(analysis) if q.startswith('question')]:
        response = getattr(analysis, q)()
        print ('  Question %s:\t%s' % (q, str(response)))
