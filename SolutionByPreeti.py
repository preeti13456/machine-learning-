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

sns.set_style("whitegrid")
sns.FacetGrid(iris, hue="Species",height = 5).map(sns.distplot,"SepalLengthCm").add_legend()
plt.show

sns.set_style("whitegrid")
sns.FacetGrid(iris, hue="Species",height = 5).map(sns.distplot,"SepalWidthCm").add_legend()
plt.show

counts, bin_edges = np.histogram(iris_setosa['PetalLengthCm'],bins =10,density = False)
print(counts) #no. of data points at particular bin
print(bin_edges)
print("***********")
#compute cdf(cumulative density function)
counts, bin_edges = np.histogram(iris_versicolor['PetalLengthCm'],bins =10,density = False)
print(counts) #no. of data points at particular bin
print(bin_edges)
print("***********")
counts, bin_edges = np.histogram(iris_virginica['PetalLengthCm'],bins =10,density = False)
print(counts) #no. of data points at particular bin
print(bin_edges)
print("***********")

#counts/bin_dufference/sum(counts)
#bin_differece = (max_no-low_no)/bins
counts, bin_edges = np.histogram(iris_setosa['PetalLengthCm'],bins =10,density = True)
pdf = counts/(sum(counts))
print(pdf)
print("counts :",counts)
print("bin_edge:", bin_edges)
#compute cdf
cdf=np.cumsum(pdf)
print("cdf",cdf)
plt.plot(bin_edges[1:],pdf, label="PDF")
plt.plot(bin_edges[1:],cdf, label="CDF")
plt.legend()
plt.show()
print("**********")
counts, bin_edges = np.histogram(iris_versicolor['PetalLengthCm'],bins =10,density = True)
pdf = counts/(sum(counts))
print(pdf)
print("counts :",counts)
print("bin_edge:", bin_edges)
#compute cdf
cdf=np.cumsum(pdf)
print("cdf",cdf)
plt.plot(bin_edges[1:],pdf, label="PDF")
plt.plot(bin_edges[1:],cdf, label="CDF")
plt.legend()
plt.show()
print("**********")
counts, bin_edges = np.histogram(iris_virginica['PetalLengthCm'],bins =10,density = True)
pdf = counts/(sum(counts))
print(pdf)
print("counts :",counts)
print("bin_edge:", bin_edges)
#compute cdf
cdf=np.cumsum(pdf)
print("cdf",cdf)
plt.plot(bin_edges[1:],pdf, label="PDF")
plt.plot(bin_edges[1:],cdf, label="CDF")
plt.legend()
plt.show()
print("**********")

iris_setosa['PetalLengthCm'].max()
iris_versicolor['PetalLengthCm'].max()
iris_virginica['PetalLengthCm'].max()

iris_setosa['PetalLengthCm'].min()
iris_virginica['PetalLengthCm'].min()
iris_versicolor['PetalLengthCm'].min()

#median - when count of num is odd[odd(n+1/2) and even(n/2)]
          #when count of num is even[mean((n/2),(n+1/2)]
print("Medians:")
print(np.median(iris_setosa["PetalLengthCm"]))
print(np.median(np.append(iris_setosa["PetalLengthCm"],50)))

#median - when count of num is odd[odd(n+1/2) and even(n/2)]
          #when count of num is even[mean((n/2),(n+1/2)]
#percentile(no. of percent of occurence of element)
import numpy as np
print("\n90th Percentiles:")
print(np.percentile(iris_setosa["PetalLengthCm"],90))
#quantile
print(np.percentile(iris_setosa["PetalLengthCm"],np.arange(0,100,50)))
#median absolute deviation = median(|(xi-median)i=1 to n|)
#IQR(inter quartile range)(The interquartile range (IQR) is a measure of variability, based on dividing a data set into quartiles. Quartiles divide a rank-ordered data set into four equal parts. The values that divide each part are called the first, second, and third quartiles; and they are denoted by Q1, Q2, and Q3, respectively.)
print("********")
print(np.percentile(iris_versicolor["PetalLengthCm"],90))
#quantile
print(np.percentile(iris_virginica["PetalLengthCm"],np.arange(0,100,50))) 

#Mean(sum(elements)/count(element)),Variance = (sum(elements)-mean)2/n,std-dev(tell me an area which have majority of data points)= sqrt(variance))
print("Means:")
print(np.mean(iris_setosa["PetalLengthCm"]))
print(np.mean(np.append(iris_setosa["PetalLengthCm"],50)))
print(np.std(iris_setosa["PetalLengthCm"]))
print("***********")
print("Means:")
print(np.mean(iris_versicolor["PetalLengthCm"]))
print(np.mean(np.append(iris_versicolor["PetalLengthCm"],50)))
print(np.std(iris_versicolor["PetalLengthCm"]))
print("***********")
print("Means:")
print(np.mean(iris_virginica["PetalLengthCm"]))
print(np.mean(np.append(iris_virginica["PetalLengthCm"],50))) #outlier
print(np.std(iris_virginica["PetalLengthCm"]))
print("***********")
from statsmodels import robust
print ("\nMedian Absolute Deviation")
print(robust.mad(iris_setosa["PetalLengthCm"]))
print(robust.mad(iris_virginica["PetalLengthCm"]))
print(robust.mad(iris_versicolor["PetalLengthCm"]))

sns.boxplot(x="Species",y = "PetalLengthCm" , data = iris)
plt.show()

sns.violinplot(x="Species",y = "PetalLengthCm" , data = iris, heighy = 5)
plt.show()

#2-D Density-plot contor-plot
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
iris = pd.read_csv("Iris.csv")
iris_setosa = iris.loc[iris["Species"]=="Iris-setosa"];
sns.jointplot(x="PetalLengthCm", y = "PetalWidthCm", data = iris_setosa, kind = "kde");
plt.show();
