import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import seaborn as sns

#importing the iris Dataset
iris = pd.read_csv("iris.csv")
iris.head()

#shape
iris.shape

iris.columns

iris['Species'].value_counts() #balanced dataset

#2D -scatter plot
iris.plot(kind = 'scatter',x = 'SepalLengthCm',y = 'SepalWidthCm')
plt.show()

sns.set_style("whitegrid")

sns.FacetGrid(iris, hue = "Species",height = 5).map(plt.scatter,'PetalLengthCm','PetalWidthCm').add_legend()

plt.show

#pair plots
plt.close()
sns.set_style("whitegrid")
sns.pairplot(iris, hue = "Species", height = 3)
plt.show()

#histogram

#1-D scatter plot
iris = pd.read_csv("Iris.csv")
import numpy as np
iris_setosa = iris.loc[iris["Species"]=="Iris-setosa"];
iris_versicolor = iris.loc[iris["Species"]=="Iris-versicolor"];
iris_virginica = iris.loc[iris["Species"]=="Iris-virginica"];
plt.plot(iris_setosa["PetalLengthCm"],np.zeros_like(iris_setosa['PetalLengthCm']),'o')
plt.plot(iris_versicolor["PetalLengthCm"],np.zeros_like(iris_versicolor['PetalLengthCm']),'o')
plt.plot(iris_virginica["PetalLengthCm"],np.zeros_like(iris_virginica['PetalLengthCm']),'o')

sns.set_style("whitegrid")
sns.FacetGrid(iris, hue="Species",height = 5).map(sns.distplot,"PetalLengthCm").add_legend()
plt.show

sns.set_style("whitegrid")
sns.FacetGrid(iris, hue="Species",height = 5).map(sns.distplot,"PetalWidthCm").add_legend()
plt.show
