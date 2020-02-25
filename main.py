import numpy
from sklearn import tree

attribs = [[150, 0], [170, 0], [140, 1], [130, 1]] # 0 para different y 1 para soft
tags = [0, 0, 1, 1] # 0 para pineapple y 1 para cucamelon 

clasifier = tree.DecisionTreeClassifier() #clasifica los datos
clasifier = clasifier.fit(attribs, tags) #encuentra un patron

print(clasifier.predict([[140, 0]]))

