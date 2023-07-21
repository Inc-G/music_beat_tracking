import pandas as pd
import numpy as np

from sklearn.decomposition import PCA

from sklearn.cluster import KMeans
from sklearn import metrics

from matplotlib import pyplot as plt
from matplotlib.pyplot import figure
import seaborn as sns

from mpl_toolkits.mplot3d import axes3d 

def auxiliary_function_savefig(savefig, name_saved_fig):
    if savefig:
        if name_saved_fig[-4:] != '.png':
            raise ValueError('need to add .png at the end of name_saved_fig')        
        plt.savefig(name_saved_fig, facecolor='white')
    

def plot_cm(cm, my_labels, title='', savefig=False, name_saved_fig='', show=True):
    """
    cm is a numpy square 2darray
    
    Plots the confusion matrix. Title is the title of the plot (string), and my_labels is
    a list of labels for x and y axis
    """
    l = len(my_labels)
    fig = plt.figure(figsize = (7,7))
    ax = fig.add_subplot(111)
    cax = ax.matshow(cm, cmap = plt.cm.summer)
    for i in range(len(my_labels)):
        for j in range(len(my_labels)):
            c = cm[j,i].round(2)
            if c>.5:
                ax.text(i, j, str(c), va='center', ha='center', color='black')
            else:
                ax.text(i, j, str(c), va='center', ha='center', color='white')
    plt.grid(False)
    ax.title.set_text(title)
    ax.set_yticks(range(l))
    ax.set_xticks(range(l))
    ax.set_xticklabels(labels = my_labels)
    ax.set_yticklabels(labels = my_labels)
    auxiliary_function_savefig(savefig, name_saved_fig)
    if show:
        plt.show()


def plot_components_pca(np_2darray, include_2d=True, include_3d=True, labels=None,
                        plot_singular_values_=False, savefig_2d=False, name_saved_fig_2d='',
                        savefig_3d=False, name_saved_fig_3d=''):
    """
    np_2darray is a np_2darray.
    If include_3d it plots also the 3d plot.
    If labels, it plots using the labels
    """
    pca = PCA(n_components=2)
    after_pca = pca.fit_transform(np_2darray)
           
    pca_2 = pd.DataFrame(after_pca, columns=['first','second'])
    print('################')
    print('#### explained_variance_ratio_ with 2 components')
    print([round(_,3) for _ in pca.explained_variance_ratio_])
    print('################')
    
    if include_2d:
        plt.figure(figsize = (10,10))    
        if type(labels)!=type(None):
            colors = {0:'green', 1:'orange', 2:'blue', 3:'red', 4:'purple', 5:'brown', 6:'black', 7:'pink', -1:'pink'}
            fig, ax = plt.subplots(figsize = (10,10))
            labels = np.array(labels)
            for g in np.unique(labels):
                ix = np.where(labels == g)
                ax.scatter(pca_2[['first']].to_numpy()[ix],
                           pca_2[['second']].to_numpy()[ix],
                           c=colors[g],
                           label=g,
                           s=100)
                    
            ax.set_xlabel("first PC")
            ax.set_ylabel("second PC")
            ax.legend()
            ax.grid(True)
            auxiliary_function_savefig(savefig_2d, name_saved_fig_2d)
            plt.show()
    
        else:
            sns.scatterplot(data=pca_2, x='first', y='second', s=100)
            plt.grid(True)
            auxiliary_function_savefig(savefig_2d, name_saved_fig_2d)
            plt.show()

    if include_3d:
        pca = PCA(n_components=3)
        after_pca = pca.fit_transform(np_2darray)
        print('################')
        print('#### explained_variance_ratio_ with 3 components')
        print([round(_,3) for _ in pca.explained_variance_ratio_])
        print('################')
        if plot_singular_values_:
            print('#### components')
            print(pca.components_)
        
        pca_3 = pd.DataFrame(after_pca, columns=['first','second','third'])
        
        ## 3d plot     
        fig = plt.figure(figsize=(10,10))
        ax = fig.add_subplot(111, projection='3d')
        x = pca_3['first']
        y = pca_3['second']
        z = pca_3['third']
        ax.set_xlabel("first")
        ax.set_ylabel("second")
        ax.set_zlabel("third")
        colors = {0:'green', 1:'orange', 2:'blue', 3:'red', 4:'purple', 5:'brown', 6:'black', 7:'pink', -1:'pink'}
        if type(labels) == type(None):
            ax.scatter(x, y, z, s=100)
            auxiliary_function_savefig(savefig_3d, name_saved_fig_3d)
            plt.show()
        else:
            ax.scatter(x, y, z, c=pd.Series(labels).map(colors), s=100)
            auxiliary_function_savefig(savefig_3d, name_saved_fig_3d)
            plt.show()

def get_sil_scores_kmeans(data, K=[3,4,5,6,7,8,9], plot_inertia=True, savefig=False, name_saved_fig=''): 
    """
    data is a np.2darray, K is a list with the number of clusters.

    Returns: sil_scores, inertia, final_labels. If plot_inertia, it plots the graph inertia-numb clusters
    (to look for the elbow)
    """
    sil_scores = []
    inertia = []
    final_labels = []
    for k in K:
        kmeans = KMeans(n_clusters=k, random_state=0, n_init=50, max_iter=500)
        final_labels.append(kmeans.fit_predict(data))
        if len(set(final_labels[-1])) != 1:
            sil_scores.append(round(metrics.silhouette_score(data, final_labels[-1]),3))
            inertia.append(round(kmeans.inertia_,3))
        else:
            sil_scores.append('one label')
            inertia.append('one label')
    if plot_inertia:
        figure(figsize=(12,12))
        
        x_axis = []
        for index, _ in enumerate(K):
            x_axis.append(str(_) + ', sil=' + str(round(sil_scores[index],3)))
            
        print('Silhouette scores: ', sil_scores)
        print('Inertia: ', inertia)
        plt.plot(x_axis, inertia, 'bx-')
        plt.xlabel('Values of K and sil scores')
        plt.ylabel('Inertia')
        plt.suptitle('Inertia Score Elbow for K-Means')
        plt.grid(True)
        auxiliary_function_savefig(savefig, name_saved_fig)
        plt.show()

    return sil_scores, inertia, final_labels
