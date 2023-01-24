import numpy as np
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from open_data import get_probs
from open_data import get_params


def visualize(gmm ,iteration):
    # choose dimension to create 2-dimensional image
    dim1, dim2 = 1, 6
    slicing = slice(dim1, dim2+1, (dim2-dim1))
    # create meshgrid
    gran = 100
    mg1 = np.linspace(gmm.maxdims[dim1,0], gmm.maxdims[dim1,1], gran)
    mg2 = np.linspace(gmm.maxdims[dim2,0], gmm.maxdims[dim2,1], gran)
    x, y = np.meshgrid(mg1, mg2)
    pos = np.dstack((x, y))
    # create plot
    fig = plt.figure()
    # calculate probability distribution
    z = np.zeros((gran,gran))
    for k in range(gmm.k):
        print(f'Sigma for cluster {k} in iteration {iteration}\n{gmm.sigma[k]}')
        rv = multivariate_normal(gmm.mu[k,slicing], gmm.sigma[k,slicing,slicing])
        z += gmm.cl_probs[k] * rv.pdf(pos)
    # plot heat map and data points
    plt.contourf(x, y, z)
    plt.plot(gmm.data[:, dim1], gmm.data[:, dim2], 'k.')
    # save figure
    plt.savefig(f'img/gmm_iter{iteration}.png')
    plt.close()

    # plot contour lines
    # reuse x, y, pos, mg1, mg2
    fig = plt.figure()
    colours = mcolors.BASE_COLORS
    col_iter = iter(colours)
    for k in range(gmm.k):
        rv = multivariate_normal(gmm.mu[k,slicing], gmm.sigma[k,slicing,slicing], allow_singular=True)
        z = rv.pdf(pos)
        plt.contour(x, y, z, colors=next(col_iter))
    plt.plot(gmm.data[:,dim1], gmm.data[:,dim2], 'k.')
    # save figure
    plt.savefig(f'img/gmm_cont_iter{iteration}.png')
    plt.close()


def visualize_cont(gmm, iteration, probs=True):
    # choose dimension to create 2-dimensional image
    dim1, dim2 = 1, 6
    slicing = slice(dim1, dim2 + 1, (dim2 - dim1))
    # create meshgrid
    gran = 100
    mg1 = np.linspace(gmm.maxdims[dim1, 0], gmm.maxdims[dim1, 1], gran)
    mg2 = np.linspace(gmm.maxdims[dim2, 0], gmm.maxdims[dim2, 1], gran)
    x, y = np.meshgrid(mg1, mg2)
    pos = np.dstack((x, y))
    # plot contour lines
    # reuse x, y, pos, mg1, mg2
    fig = plt.figure()
    plt.xlabel(gmm.labels[dim1])
    plt.ylabel(gmm.labels[dim2])
    colours = mcolors.BASE_COLORS
    col_iter = iter(colours)
    # meteorological seasons: spring: 1.3., summer: 1.6., autumn: 1.9., winter1: 1.12., winter2: 1.1.
    seasons = np.array([31 + 28, 31 + 30 + 31, 30 + 31 + 31, 30 + 31 + 30, 31 ,31 + 28])
    seasons = [np.sum(seasons[:k]) for k in range(len(seasons))]
    season_cols = [mcolors.CSS4_COLORS['lightgreen'], mcolors.CSS4_COLORS['plum'],
                   mcolors.CSS4_COLORS['sandybrown'], mcolors.CSS4_COLORS['lightskyblue'],
                   mcolors.CSS4_COLORS['lightskyblue']]

    # plotting seasons with different colors
    for s in range(5):  # spring summer autumn winter winter
        startdate = seasons[s ] % seasons[3]
        enddate = seasons[s+1]
        plt.plot(gmm.data[startdate:enddate, dim1],
                 gmm.data[startdate:enddate, dim2],
                 color=season_cols[s], marker='.', linestyle='')

    for k in range(gmm.k * int(probs)):
        rv = multivariate_normal(gmm.mu[k, slicing], gmm.sigma[k, slicing, slicing], allow_singular=True)
        z = rv.pdf(pos)
        plt.contour(x, y, z, colors=next(col_iter))
    # save figure
    plt.savefig(f'img/gmm_cont_iter{iteration}.png')
    plt.close()


def visualize_dims(k, kmeans=False, first_setup=True):
    inittype = 'kmeans_init' if kmeans else 'random_init'
    setup = 'first_setup' if first_setup else 'second_setup'

    data, labels, dim, mu, sigma, maxdims = get_params(k=k, kmeans=kmeans, first_setup=first_setup)

    seasons = np.array([31 + 28, 31 + 30 + 31, 30 + 31 + 31, 30 + 31 + 30, 31, 31 + 28 -365])
    seasons = [np.sum(seasons[:i]) for i in range(1,len(seasons)+1)]
    season_cols = [mcolors.CSS4_COLORS['lightgreen'], mcolors.CSS4_COLORS['plum'],
                   mcolors.CSS4_COLORS['sandybrown'], mcolors.CSS4_COLORS['lightskyblue'],
                   mcolors.CSS4_COLORS['lightskyblue']]
    colours = mcolors.BASE_COLORS
    gran = 100
    for dim1 in range(dim):
        mg1 = np.linspace(maxdims[dim1, 0], maxdims[dim1, 1], gran)
        for dim2 in range(dim1+1,dim):
            print(f'k={k}, dim1={dim1}, dim2={dim2}')
            fig2,ax = plt.subplots()
            ax.set(xlabel=labels[dim1], ylabel=labels[dim2])
            slicing = slice(dim1, dim2+1, (dim2-dim1))
            mg2 = np.linspace(maxdims[dim2, 0], maxdims[dim2, 1], gran)
            x, y = np.meshgrid(mg1, mg2)
            pos = np.dstack((x, y))
            # plot contour lines
            col_iter = iter(colours)
            for i in range(k):
                rv = multivariate_normal(mu[i, slicing], sigma[i, slicing, slicing], allow_singular=True)
                z = rv.pdf(pos)
                ax.contour(x, y, z, colors=next(col_iter))
            # plotting seasons with different colors
            for s in range(5):  # spring summer autumn winter winter
                startdate = seasons[s] % seasons[4]
                enddate = seasons[s + 1]
                ax.plot(data[startdate:enddate, dim1],
                         data[startdate:enddate, dim2],
                         color=season_cols[s], marker='.', linestyle='')
            # save figure
            plt.savefig(f'{setup}/{inittype}/img/gmm_k={k}_{dim1}{dim2}.png')
            plt.close()


def vis_calendar(k):
    pis = get_probs(k)
    k = pis.shape[1]-2
    days_p_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    months = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
    beg_month = [int(np.sum(days_p_month[:i])) for i in range(len(days_p_month)+1)]
    fig, axs = plt.subplots(nrows=12,ncols=1)
    for row in range(12):
        pos = np.arange(31) + 1
        axs[row].set(ylabel=months[row])
        axs[row].tick_params(left=False, labelleft=False)
        for pk in range(k):
            height = pis[beg_month[row]:beg_month[row+1],pk]
            height = np.concatenate([height, np.zeros(31-days_p_month[row])])
            bottom = np.sum(pis[beg_month[row]:beg_month[row+1],:pk],axis=1)
            bottom = np.concatenate([bottom, np.zeros(31-days_p_month[row])])
            axs[row].bar(x=pos, height=height, width=0.9, bottom=bottom, label='Cluster '+str(pk))
    # save figure
    plt.savefig(f'img/calendar{k}.png')
    plt.close()


def main():
    for i in range(2,8):
        #vis_calendar(i)
        # visualize final distributions
        visualize_dims(k=i,kmeans=False)

if __name__ == '__main__':
    main()