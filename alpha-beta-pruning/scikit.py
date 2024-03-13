from sklearn.tree import DecisionTreeClassifier
from dtreeviz.trees import dtreeviz
import matplotlib.pyplot as plt

# Example data for Naughts and Crosses (Tic-Tac-Toe)
X = [[0, 0, 0, 0, 1, 0, 0, 0, 1],  # Example board state
     [0, 0, 0, 1, 1, 0, 0, 0, 0],
     [0, 1, 0, 0, 1, 0, 0, 0, 0],
     [0, 1, 0, 0, 1, 0, 0, 0, 0],
     [1, 0, 0, 0, 1, 0, 0, 0, 0],
     [1, 0, 0, 0, 1, 0, 0, 0, 0]]

# Example target labels
y = ['O', 'X', 'O', 'X', 'O', 'X']

# Create and fit the decision tree classifier
clf = DecisionTreeClassifier()
clf = clf.fit(X, y)

# Visualize the decision tree
viz = dtreeviz(clf, X, y, feature_names=[f"cell_{i}" for i in range(9)], class_names=['O', 'X'])
viz.save("decision_tree.svg")
viz.view()
