import matplotlib.pyplot as plt
def plot_histogram(housing, bincount, figure):
    housing.hist(bins=bincount, figsize=figure)
    plt.show()


#HeatMap of housing data. 
#Plot the data set as a scatter plot to visualize the data. 
def heatmap(housing):
    housing.plot(kind='scatter', x='longitude', y='latitude', alpha=0.4, s=housing['population']/100, label='population',
    figsize=(12, 8), c='median_house_value', cmap=plt.get_cmap('jet'), colorbar=True)
    plt.legend()
    plt.show()