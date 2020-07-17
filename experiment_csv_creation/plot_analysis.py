import warnings
import random
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.mlab as mlab
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib
from matplotlib import pyplot as plt
from math import exp, fabs, sqrt, ceil, isnan
from scipy.stats import pearsonr, f_oneway
from statsmodels.stats.multicomp import MultiComparison
from scipy.stats import pearsonr, spearmanr, gaussian_kde, probplot, ks_2samp
from matplotlib.colors import LogNorm
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
from scipy import stats
import seaborn as sns
from matplotlib.patches import Polygon

roles = ['all', 'Goalkeeper', 'Defender', 'Midfielder', 'Forward']

lw = 2.5
label_size = 30
title_size = 30
figsize = (6, 6)
newspaper2style = {'gazzetta': {'color': '#d95f02'},
                   'tuttosport': {'color': '#7570b3'},
                   'corriere': {'color': '#1b9e77'},
                   'fantagazzetta': {'color': '#48D3FB'}}

newspaper2name = {'gazzetta': '$NP_1$',
                  'tuttosport': '$NP_3$',
                  'corriere': '$NP_2$',
                  'fantagazzetta': '$NP_4$',
                 'artificial': '$NP_5$',
                 'human' : '$NP_6$'}

newspaper2name = {'gazzetta': '$G$',
                  'tuttosport': '$T$',
                  'corriere': '$C$',
                  'fantagazzetta': '$F$',
                 'human' : '$Human$',
                 'artificial': '$AJ$'}

mpl.figure.figsize = (6, 6)


def plotDistribution(ratings_g, name):
    '''

    :param ratings_g: Gazzetta dello sport list ratings
    :param ratings_c: Corriere dello sport list ratings
    :param ratings_t: Tuttosport list ratings
    :param ratings_f: Fantacalcio List ratingd
    :return:
            Visualization of the distribution of Gazzetta Rating and its computation
    '''
    matplotlib.style.use('seaborn-ticks')
    # mean and stf
    mean, std = round(ratings_g.mean(), 2), round(ratings_g.std(), 2)

    fig = plt.figure(figsize=figsize)
    ax = plt.axes()

    bins = list(np.arange(0, 10.5, 0.5))
    plt.hist(ratings_g, bins=bins, normed=True, rwidth=0.85,
             linewidth=0, color=newspaper2style[name]['color'], label=newspaper2name[name],
             zorder=2, align='left')

    ## LABELS
    plt.xlabel('%s ratings' %name, fontsize=label_size)
    plt.ylabel('p(ratings)', fontsize=label_size, labelpad=15)
    plt.xticks(list(np.arange(0, 10.5, 1.0)), fontsize=10)
    plt.xlim(0, 10)

    # take the min and max mark values, and compute avg and std
    #min_rating = min([min(ratings_g), min(ratings_c), min(ratings_t), min(ratings_f)])
    #max_rating = min([max(ratings_g), max(ratings_c), max(ratings_t), min(ratings_f)])
    #avg = ceil(np.mean(ratings_g + ratings_c + ratings_t + ratings_f))
    #std = ceil(np.std(ratings_g + ratings_c + ratings_t + ratings_f))

    ## SUFFICIENCY LINE
    plt.vlines(6, 0, 0.7, linewidth=2.5, color='k', linestyle='--', alpha=0.75, label='sufficiency', zorder=3)

    ## SET THE TICKS
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=15)
    for i in range(4, 9):
        ax.xaxis.get_major_ticks()[i].label1.set_fontweight('bold')
        if i == 6:
            ax.xaxis.get_major_ticks()[i].label1.set_fontsize('25')

    ## FILL OUR OF MEAN +- STD
    ax.fill_between([min(ratings_g), max(ratings_g)], 0, 0.8, facecolor='k', alpha=0.10)
    ax.set_ylim(0,0.7)

    ## LEGEND
    legend = plt.legend(loc=2, fontsize=15, frameon=True, shadow=True, handlelength=1.6, handletextpad=0.4)
    # print legend.get_texts()[0].set_fontsize(20)

    ## ANNOTATE INFO
    plt.annotate('$\mu=%s$\n$\sigma=%s$' % (mean, std), xy=(0.8, 0.31), fontsize=25)
    plt.grid(alpha=0.60)
    #plt.title('%s distribution' %name, fontsize=title_size)
    
    fig.tight_layout()
    plt.savefig('img/ratings_distr_%s.pdf' %name)
    plt.savefig('img/ratings_distr_%s.svg' %name)
    plt.savefig('img/ratings_distr_%s.png' %name)
    plt.show()


def plotBoxplotNewspaperDistribution(ratings_g, ratings_c, ratings_t, ratings_f):
    '''

    :param ratings_g: Gazzetta dello sport list ratings
    :param ratings_c: Corriere dello sport list ratings
    :param ratings_t: Tuttosport list ratings
    :param ratings_f: Fantacalcio list ratingd
    :return:
            A boxplot of newspaper marks disrtribution
    '''
    # mean and std for gazzetta scores
    mean, std = ratings_g.mean(), ratings_g.std()

    fig = plt.figure(figsize=figsize)
    ax = plt.axes()
    bp = plt.boxplot([ratings_g, ratings_c, ratings_t, ratings_f], sym="o")

    ## LABELS
    plt.ylabel('ratings', fontsize=label_size)

    ## BOXPLOTS STYLE
    bp['boxes'][0].set(color=newspaper2style['gazzetta']['color'], linewidth=2)
    bp['boxes'][1].set(color=newspaper2style['corriere']['color'], linewidth=2)
    bp['boxes'][2].set(color=newspaper2style['tuttosport']['color'], linewidth=2)
    bp['boxes'][3].set(color=newspaper2style['fantagazzetta']['color'], linewidth=2)

    bp['medians'][0].set(color=newspaper2style['gazzetta']['color'], linewidth=2)
    bp['medians'][1].set(color=newspaper2style['corriere']['color'], linewidth=2)
    bp['medians'][2].set(color=newspaper2style['tuttosport']['color'], linewidth=2)
    bp['medians'][3].set(color=newspaper2style['fantagazzetta']['color'], linewidth=2)

    bp['whiskers'][0].set(color=newspaper2style['gazzetta']['color'], linewidth=2)
    bp['whiskers'][1].set(color=newspaper2style['gazzetta']['color'], linewidth=2)
    bp['whiskers'][2].set(color=newspaper2style['corriere']['color'], linewidth=2)
    bp['whiskers'][3].set(color=newspaper2style['corriere']['color'], linewidth=2)
    bp['whiskers'][4].set(color=newspaper2style['tuttosport']['color'], linewidth=2)
    bp['whiskers'][5].set(color=newspaper2style['tuttosport']['color'], linewidth=2)
    bp['whiskers'][6].set(color=newspaper2style['fantagazzetta']['color'], linewidth=2)
    bp['whiskers'][7].set(color=newspaper2style['fantagazzetta']['color'], linewidth=2)

    bp['caps'][0].set(color=newspaper2style['gazzetta']['color'], linewidth=2)
    bp['caps'][1].set(color=newspaper2style['gazzetta']['color'], linewidth=2)
    bp['caps'][2].set(color=newspaper2style['corriere']['color'], linewidth=2)
    bp['caps'][3].set(color=newspaper2style['corriere']['color'], linewidth=2)
    bp['caps'][4].set(color=newspaper2style['tuttosport']['color'], linewidth=2)
    bp['caps'][5].set(color=newspaper2style['tuttosport']['color'], linewidth=2)
    bp['caps'][6].set(color=newspaper2style['fantagazzetta']['color'], linewidth=2)
    bp['caps'][7].set(color=newspaper2style['fantagazzetta']['color'], linewidth=2)

    flier = bp['fliers'][0]
    flier.set(markeredgecolor=newspaper2style['gazzetta']['color'],
              markerfacecolor=newspaper2style['gazzetta']['color'],
              alpha=0.3)
    flier = bp['fliers'][1]
    flier.set(markeredgecolor=newspaper2style['corriere']['color'],
              markerfacecolor=newspaper2style['corriere']['color'],
              alpha=0.3)
    flier = bp['fliers'][2]
    flier.set(markeredgecolor=newspaper2style['tuttosport']['color'],
              markerfacecolor=newspaper2style['tuttosport']['color'],
              alpha=0.3)
    flier = bp['fliers'][3]
    flier.set(markeredgecolor=newspaper2style['fantagazzetta']['color'],
              markerfacecolor=newspaper2style['fantagazzetta']['color'],
              alpha=0.3)

    ## FILL OUR OF MEAN +- STD
    ax.fill_between([0, 5], 4, 8, facecolor='k', alpha=0.05)

    ## TICKS
    plt.xticks([1, 2, 3, 4], [newspaper2name['gazzetta'],
                              newspaper2name['corriere'],
                              newspaper2name['tuttosport'],
                              newspaper2name['fantagazzetta']],
               fontsize=20)
    plt.yticks(fontsize=15)
    ax.yaxis.get_major_ticks()[3].label1.set_fontweight('bold')
    ax.yaxis.get_major_ticks()[3].label1.set_fontsize('25')

    plt.grid(alpha=0.60)
    plt.ylim(0, 10.5)
    fig.tight_layout()
    plt.savefig('img/boxplots_ratings.pdf')
    plt.show()

    # The one-way ANOVA compares the means between the groups
    # you are interested in and determines whether any of those means are statistically significantly different from each other.
    # works in steps
    # 1. Divide our osservation into groups
    # 2. Compute mean and variance of all the groups and the whole dataset
    # 3. Compute deviance between groups
    # 4. Compute deviance between means of groups
    # 5. Compute population variance
    # 6. Compute F statistics
    F, p = f_oneway(ratings_g, ratings_c, ratings_t, ratings_f)
    '''
    P value
    The P value tests the null hypothesis that data from all groups are drawn from populations with identical means. Therefore,
    the P value answers this question: If all the populations really have the same mean (the treatments are ineffective),
    what is the chance that random sampling would result in means as far apart (or more so) as observed in this experiment?

    If the overall P value is large, the data do not give you any reason to conclude that the means differ. Even if the 
    population means were equal, you would not be surprised to find sample means this far apart just by chance. 
    This is not the same as saying that the true means are the same. You just don't have compelling evidence that they differ.

    If the overall P value is small, then it is unlikely that the differences you observed are due to random sampling. 
    You can reject the idea that all the populations have identical means. This doesn't mean that every mean differs 
    from every other mean, only that at least one differs from the rest. Look at the results of post tests to identify 
    where the differences are.



    F value

    ANOVA partitions the variability among all the values into one component that is due to variability among group means
    (due to the treatment) and another component that is due to variability within the groups (also called residual variation).
    Variability within groups (within the columns) is quantified as the sum of squares of the differences between each value 
    and its group mean. This is the residual sum-of-squares. Variation among groups (due to treatment) is quantified as the sum 
    of the squares of the differences between the group means and the grand mean (the mean of all values in all groups). 
    Adjusted for the size of each group, this becomes the treatment sum-of-squares.

    Each sum-of-squares is associated with a certain number of degrees of freedom 
    (df, computed from number of subjects and number of groups), and the mean square (MS) is computed by dividing the 
    sum-of-squares by the appropriate number of degrees of freedom. These can be thought of as variances. 
    The square root of the mean square residual can be thought of as the pooled standard deviation.

    The F ratio is the ratio of two mean square values. 
    If the null hypothesis is true, you expect F to have a value close to 1.0 most of the time.
    A large F ratio means that the variation among group means is more than you'd expect to see by chance. 
    You'll see a large F ratio both when the null hypothesis is wrong (the data are not sampled from populations with the same mean)
    and when random sampling happened to end up with large values in some groups and small values in others.

    '''
    print('one_way ANOVA: ', F, p)

    ll = list(ratings_g) + list(ratings_c) + list(ratings_t) + list(ratings_f)
    ll2 = list(['1'] * len(ratings_g)) + list(['2'] * len(ratings_g)) + list(['3'] * len(ratings_g)) + list(
        ['4'] * len(ratings_f))
    print(len(ll), len(ll2))
    mc = MultiComparison(np.array(ll), np.array(ll2), )
    '''
    Turkey hsd used to know if two groups of value has large difference
    Use a concept of ordering and rank in order to found out if the groups are unbalanced
    '''
    result = mc.tukeyhsd()
    print(result)


def plotViolinPlotNewspaperDistribution(ratings_g, ratings_c, ratings_t, ratings_f):
    '''

    :param ratings_g: Gazzetta dello sport list ratings
    :param ratings_c: Corriere dello sport list ratings
    :param ratings_t: Tuttosport list ratings
    :param ratings_f: Fantacalcio list ratings
    :return:
            Violin plot of marks distribution of different newspapers
    '''
    mean, std = ratings_g.mean(), ratings_g.std()

    fig = plt.figure(figsize=figsize)
    ax = plt.axes()

    vert = False

    bp = plt.violinplot(ratings_g, showextrema=False, vert=vert)
    for pc in bp['bodies']:
        pc.set_facecolor('w')
        pc.set_edgecolor(newspaper2style['gazzetta']['color'])
        pc.set_alpha(0.95)
        pc.set_linewidth(2)

    bp = plt.violinplot(ratings_c, showextrema=False, vert=vert)
    for pc in bp['bodies']:
        pc.set_facecolor('w')
        pc.set_edgecolor(newspaper2style['corriere']['color'])
        pc.set_alpha(0.5)
        pc.set_linewidth(2)

    bp = plt.violinplot(ratings_t, showextrema=False, vert=vert)
    for pc in bp['bodies']:
        pc.set_facecolor('w')
        pc.set_edgecolor(newspaper2style['tuttosport']['color'])
        pc.set_alpha(0.6)
        pc.set_linewidth(2)

    bp = plt.violinplot(ratings_f, showextrema=False, vert=vert)
    for pc in bp['bodies']:
        pc.set_facecolor('w')
        pc.set_edgecolor(newspaper2style['fantagazzetta']['color'])
        pc.set_alpha(0.7)
        pc.set_linewidth(2)

    ks_gc = ks_2samp(ratings_g, ratings_c)[0]
    ks_gt = ks_2samp(ratings_g, ratings_t)[0]
    ks_gf = ks_2samp(ratings_g, ratings_f)[0]
    ks_ct = ks_2samp(ratings_c, ratings_t)[0]
    ks_cf = ks_2samp(ratings_c, ratings_f)[0]
    ks_tf = ks_2samp(ratings_t, ratings_f)[0]

    ks = np.mean([ks_gc, ks_gt, ks_gf, ks_ct, ks_cf, ks_tf])

    print(ks)

    plt.annotate('$\overline{KS}=%s$' % round(ks, 2), xy=(0.6, 0.1), xycoords='axes fraction',
                 fontsize=30)

    to_show = ['2', '', '3', '', '4', '', '5', '', '6', '', '7', '', '8', '', '9', '', '10']
    plt.xticks(np.arange(2, 10.5, 0.5), to_show, fontsize=15)
    plt.yticks([])

    ax.xaxis.get_major_ticks()[8].label1.set_fontsize(30)
    ax.xaxis.get_major_ticks()[8].label1.set_fontweight('bold')

    for i in [6, 10]:
        ax.xaxis.get_major_ticks()[i].label1.set_fontsize(25)
        ax.xaxis.get_major_ticks()[i].label1.set_fontweight('bold')

    for i in [4, 12]:
        ax.xaxis.get_major_ticks()[i].label1.set_fontsize(20)

    # Create custom artists
    gazz_artist = plt.Line2D((0, 1), (0, 0),
                             color=newspaper2style['gazzetta']['color'], linewidth=2)
    corr_artist = plt.Line2D((0, 1), (0, 0),
                             color=newspaper2style['corriere']['color'], linewidth=2)
    tutt_artist = plt.Line2D((0, 1), (0, 0),
                             color=newspaper2style['tuttosport']['color'], linewidth=2)
    fanta_artist = plt.Line2D((0, 1), (0, 0),
                              color=newspaper2style['fantagazzetta']['color'], linewidth=2)

    # Create legend from custom artist/label lists
    plt.legend([gazz_artist, corr_artist, tutt_artist, fanta_artist],
               ['$G$', '$C$', '$T$', '$F$'], fontsize=25, loc=2, shadow=True, frameon=True,
               handletextpad=0.1, handlelength=1)

    plt.xlabel('ratings', fontsize=label_size)

    plt.grid(alpha=0.60, which='major', )
    fig.tight_layout()
    plt.savefig('img/violinplots_ratings_all_in_one.pdf')
    plt.savefig('img/violinplots_ratings_all_in_one.svg')
    plt.savefig('img/violinplots_ratings_all_in_one.png')
    plt.show()


def correlation_newspapers(orig_x, orig_y, name1, name2,plot_type='scatter', concordance=None):
    '''

    :param orig_x: list of ratings of a newspaper
    :param orig_y: list of ratings of a comparing newspaper
    :param name1: name of the first newspaper
    :param name2: name of the second newspaper
    :param plot_type: always scatter plot
    :param concordance: always set to none
    :return:
            A scatter plot that put in correlation the two newspaper analyzing concordances and discordances
    '''
    x, y = [], []
    for a, b in zip(orig_x, orig_y):
        if concordance == True:  # we take just the concordant ratings
            if (a >= 6 and b >= 6) or (a < 6 and b < 6):
                x.append(a)
                y.append(b)
        elif concordance == False:
            if (a >= 6 and b < 6) or (a < 6 and b >= 6):
                x.append(a)
                y.append(b)
        else:
            x.append(a)
            y.append(b)
            
    number_of_agreements = 0
    tott = len(orig_x)
    for a, b in zip(orig_x, orig_y):
        if (a >= 6 and b >= 6) or (a < 6 and b < 6):
            number_of_agreements += 1
    print(number_of_agreements/tott)
    # compute max error
    diffs = []
    for a, b in zip(x, y):
        diffs.append(fabs(a - b))
    max_diff = max(diffs)

    fig = plt.figure(figsize=(6, 6))
    ax = plt.axes()
    if plot_type == 'scatter':
        xy = np.vstack([x, y])
        z = gaussian_kde(xy)(xy)
        print(z)
        # Sort the points by density, so that the densest points are plotted last
        sc = plt.scatter(x, y, s=(z * 150) + 20, linewidth=0.1, zorder=1, marker='o', alpha=0.2, norm=LogNorm())
    else:
        bb = [len(np.arange(min(x), max(x) + 0.5, 0.5)), len(np.arange(min(y), max(y) + 0.5, 0.5))]
        plt.hist2d(x, y, bins=bb, normed=False, norm=LogNorm(), alpha=0.7)
        # plt.colorbar(pad=0.01)

    plt.plot([0, 10], [0, 10], color='k', alpha=0.5)
    lr = LinearRegression(normalize=True)
    new_x, new_y = [[xx] for xx in x], [[yy] for yy in y]
    score = cross_val_score(lr, new_x, new_y, cv=10, scoring='r2')
    avg_score = np.mean(score)

    pearson, pvalue = pearsonr(x, y)
    rmse_val = round(sqrt(mean_squared_error(x, y)), 2)
    ks = ks_2samp(x, y)[0]
    plt.xlim(-0., 10.)
    plt.ylim(-0., 10.)
    plt.annotate("$r=%s$\n$KS=%s$\n$RMSE=%s$" % (round(pearson, 2), round(ks, 2), rmse_val), xy=(0.05, 0.75),
                 xycoords='axes fraction', fontsize=21)
    # plt.annotate('$diff_{max}=%s$' %max_diff, xy=(0.635, 0.2),
    #             xycoords='axes fraction', fontsize=18)
    plt.xlabel("%s ratings" % newspaper2name[name1], fontsize=25)
    plt.ylabel("%s ratings" % newspaper2name[name2], fontsize=25)

    plt.vlines(6, 0, 10)
    plt.hlines(6, 0, 10)

    plt.xticks(range(0, 11), fontsize=15)
    plt.yticks(range(0, 11), fontsize=15)
    ax.xaxis.get_major_ticks()[6].label1.set_fontweight('bold')
    ax.xaxis.get_major_ticks()[6].label1.set_fontsize(20)
    ax.yaxis.get_major_ticks()[6].label1.set_fontweight('bold')
    ax.yaxis.get_major_ticks()[6].label1.set_fontsize(20)

    plt.fill_between([6, 10], 6, color='k', alpha=0.05, label='disagreement')
    plt.fill_between([0, 6], 6, 10, color='k', alpha=0.05)
    plt.legend(loc=3, fontsize=18, frameon=True, shadow=True, handletextpad=0.2)
    #plt.title('%s' %tit, fontsize=25)
    plt.grid(alpha=0.2)
    fig.tight_layout()
    plt.savefig('img/corr_ratings_%s_%s.png' % (name1, name2))
    plt.savefig('img/corr_ratings_%s_%s.svg' % (name1, name2))
    plt.savefig('img/corr_ratings_%s_%s.pdf' % (name1, name2))
    plt.show()


def plotHistogramRoleMarkDistribution(midfielder, forward, defence, goalkeeper, newspaperName):
    '''

    :param midfielder: midfielder marks of a newspaper
    :param forward: forward marks of a newspaper
    :param defence: defence marks of a newspaper
    :param goalkeeper: goalkeeeper marks of a newspaper
    :return:
            Histogram Plot of distribution of mark in each separate role
    '''
    # Assign colors for each role and the names
    colors = ['#E69F00', '#56B4E9', '#D55E00', '#009E73']
    names = ['Mid', 'For', 'Def', 'Gk']

    fig = plt.figure(figsize=figsize)
    ax = plt.axes()
    # Make the histogram using a list of lists
    # Normalize the flights and assign colors and names
    plt.hist([midfielder, forward, defence, goalkeeper], bins=int(180 / 15), normed=True,
             color=colors, label=names, width=0.15)  
    
    # Plot formatting
    plt.legend(fontsize=18, frameon=True, shadow=True, handletextpad=0.2, loc='upper left')
    plt.grid(alpha=0.2)
    plt.xlabel(newspaperName, fontsize=label_size)
    plt.ylabel('P(%s)' %newspaperName, fontsize=label_size,labelpad = 15)
    to_show = ['3', '', '4', '', '5', '', '6', '', '7', '', '8','','9']
    #positions = [3,3.3,3.9,4.47,5.05,5.65,6.2,6.8,7.4,7.95,8.55]
    positions = [3.25,3.75,4.25,4.75,5.25,5.75,6.25,6.75,7.25,7.75,8.25,8.75,9.3]
    plt.xticks(positions, to_show, fontsize=20)
    #plt.xticks(range(3, 9), fontsize=20)
    plt.yticks(fontsize=15)
    
    ax.xaxis.get_major_ticks()[7].label1.set_fontsize(30)
    ax.xaxis.get_major_ticks()[7].label1.set_fontweight('bold')

    for i in [5, 9]:
        ax.xaxis.get_major_ticks()[i].label1.set_fontsize(25)
        ax.xaxis.get_major_ticks()[i].label1.set_fontweight('bold')

    for i in [4, 12]:
        ax.xaxis.get_major_ticks()[i].label1.set_fontsize(20)
        
    ax.xaxis.get_major_ticks()[6].label1.set_fontweight('bold')
    ax.xaxis.get_major_ticks()[6].label1.set_fontsize(30)
    ax.set_xlim(3,9)
    #plt.title('Histogram with Multiple Roles', fontsize=title_size - 5)
    plt.tight_layout()
    plt.savefig('img/roles_mark_distribution.pdf')
    plt.show()


def plotViolinPlotRolesMarkDistribution(midfielder, forward, defence, goalkeeper, newspaperName):
    '''

    :param midfielder: midfielder marks of a newspaper
    :param forward: forward marks of a newspaper
    :param defence: defence marks of a newspaper
    :param goalkeeper: goalkeeeper marks of a newspaper
    :param newspaperName: name of the newspaper where marks are taken
    :return:
            Violin Plot of distribution of mark in each separate role
    '''
    fig = plt.figure(figsize=figsize)
    ax = plt.axes()

    vert = False

    # plot violin plot

    bp = plt.violinplot(midfielder, showextrema=False, vert=vert)
    for pc in bp['bodies']:
        pc.set_facecolor('w')
        pc.set_edgecolor('#E69F00')
        #pc.set_alpha(1)
        pc.set_linewidth(3)

    bp = plt.violinplot(forward, showextrema=False, vert=vert)
    for pc in bp['bodies']:
        pc.set_facecolor('w')
        pc.set_edgecolor('#56B4E9')
        #pc.set_alpha(1)
        pc.set_linewidth(3)

    bp = plt.violinplot(defence, showextrema=False, vert=vert)
    for pc in bp['bodies']:
        pc.set_facecolor('w')
        pc.set_edgecolor('#D55E00')
        #pc.set_alpha(1)
        pc.set_linewidth(3)

    bp = plt.violinplot(goalkeeper, showextrema=False, vert=vert)
    for pc in bp['bodies']:
        pc.set_facecolor('w')
        pc.set_edgecolor('#009E73')
        #pc.set_alpha(0.5)
        pc.set_linewidth(3)

    to_show = ['2', '', '3', '', '4', '', '5', '', '6', '', '7', '', '8', '', '9', '', '10']
    plt.xticks(np.arange(2, 10.5, 0.5), to_show, fontsize=20)
    plt.yticks([])

    ax.xaxis.get_major_ticks()[8].label1.set_fontsize(30)
    ax.xaxis.get_major_ticks()[8].label1.set_fontweight('bold')

    for i in [6, 10]:
        ax.xaxis.get_major_ticks()[i].label1.set_fontsize(25)
        ax.xaxis.get_major_ticks()[i].label1.set_fontweight('bold')

    for i in [4, 12]:
        ax.xaxis.get_major_ticks()[i].label1.set_fontsize(20)

    # Create custom artists
    midfielder_artist = plt.Line2D((0, 1), (0, 0),
                                   color='#E69F00', linewidth=2)
    forward_artist = plt.Line2D((0, 1), (0, 0),
                                color='#56B4E9', linewidth=2)
    defence_artist = plt.Line2D((0, 1), (0, 0),
                                color='#D55E00', linewidth=2)
    gk_artist = plt.Line2D((0, 1), (0, 0),
                           color='#009E73', linewidth=2)

    # Create legend from custom artist/label lists
    plt.legend([midfielder_artist, forward_artist, defence_artist, gk_artist],
               ['$Mid$', '$For$', '$Def$', '$Gk$'], fontsize=18, loc=2, shadow=True, frameon=True,
               handletextpad=0.1, handlelength=1)

    plt.xlabel('%s Ratings' %newspaperName, fontsize=label_size)
    #plt.title('Ratings Distribution by Role for %s' % newspaperName, fontsize=20)

    plt.grid(alpha=0.60, which='major', )
    fig.tight_layout()
    plt.savefig('img/violinplots_ratings_by_role.pdf')
    plt.savefig('img/violinplots_ratings_by_role.svg')
    plt.savefig('img/violinplots_ratings_by_role.png')
    plt.show()


def boxplotCorrelationWinFinalMark(wingazzetta, notwingazzetta, winfanta, notwinfanta):
    '''

    :param wingazzetta: list of votes of player that win the match of a newspaper
    :param notwingazzetta: list of votes of player that didn't win the match of a newspaper
    :param winfanta: same as above
    :param notwinfanta: same as above
    :return:

            Boxplot containing distirbution of votes in each of the four described situations
    '''

    random_dists = ['Winner G', 'Not Winner G', 'Winner F', 'Not Winner F']

    # Generate some random indices that we'll use to resample the original data
    # arrays. For code brevity, just use the same random indices for each array
    data = [
        wingazzetta, notwingazzetta, winfanta, notwinfanta
    ]
    fig, ax1 = plt.subplots(figsize=figsize)
    fig.canvas.set_window_title('A Boxplot Example')
    fig.subplots_adjust(left=0.075, right=0.95, top=0.9, bottom=0.25)

    bp = ax1.boxplot(data, notch=0, sym='gD', vert=1, whis=1.5)
    plt.setp(bp['boxes'], color='black')
    plt.setp(bp['whiskers'], color='black')
    plt.setp(bp['fliers'], color='red', marker='+')

    # Add a horizontal grid to the plot, but make it very light in color
    # so we can use it for reading data values but not be distracting
    ax1.yaxis.grid(True, linestyle='-', which='major', color='lightgrey',
                   alpha=0.5)

    # Hide these grid behind plot objects
    ax1.set_axisbelow(True)
    #ax1.set_title('Comparison of Winning Distribution and Not Winning Ones', fontsize=14)
    ax1.set_xlabel('Final Result', fontsize=label_size)
    ax1.set_ylabel('Scores', fontsize=label_size, labelpad=15)

    # Now fill the boxes with desired colors
    box_colors = ['darkkhaki', 'royalblue']
    num_boxes = len(data)
    medians = np.empty(num_boxes)
    for i in range(num_boxes):
        box = bp['boxes'][i]
        boxX = []
        boxY = []
        for j in range(5):
            boxX.append(box.get_xdata()[j])
            boxY.append(box.get_ydata()[j])
        box_coords = np.column_stack([boxX, boxY])
        # Alternate between Dark Khaki and Royal Blue
        ax1.add_patch(Polygon(box_coords, facecolor=box_colors[i % 2]))
        # Now draw the median lines back over what we just filled in
        med = bp['medians'][i]
        medianX = []
        medianY = []
        for j in range(2):
            medianX.append(med.get_xdata()[j])
            medianY.append(med.get_ydata()[j])
            ax1.plot(medianX, medianY, 'k')
        medians[i] = medianY[0]
        # Overplot the sample averages, with horizontal alignment
        # in the center of each box
        ax1.plot(np.average(med.get_xdata()), np.average(data[i]),
                 color='w', marker='*', markeredgecolor='k')

    # Set the axes ranges and axes labels
    ax1.set_xlim(0.5, num_boxes + 0.5)
    top = 10
    bottom = 0
    ax1.set_ylim(bottom, top)
    ax1.set_xticklabels(np.repeat(random_dists, 1),
                        rotation=35, fontsize=17, ha='right')
    # y management
    pos = np.arange(num_boxes) + 1
    upper_labels = [str(np.round(s, 2)) for s in medians]
    weights = ['bold', 'semibold']
    for tick, label in zip(range(num_boxes), ax1.get_xticklabels()):
        k = tick % 2
        ax1.text(pos[tick], .95, upper_labels[tick],
                 transform=ax1.get_xaxis_transform(),
                 horizontalalignment='center', size='x-large',
                 weight=weights[k], color=box_colors[k])

    # Finally, add a basic legend
    fig.text(0.75, 0.48, 'Winner',
             backgroundcolor=box_colors[0], color='black', weight='roman',
             size='medium')
    fig.text(0.75, 0.445, 'Not Winner',
             backgroundcolor=box_colors[1],
             color='white', weight='roman', size='medium')
    fig.text(0.75, 0.410, '*', color='white', backgroundcolor='silver',
             weight='roman', size='medium')
    fig.text(0.765, 0.413, ' Average Value', color='black', weight='roman',
             size='medium')
    plt.xlabel('Match Outcome', fontsize=label_size)
    plt.ylabel('Ratings (G, F)', fontsize=label_size, labelpad=15)
    plt.yticks(fontsize=15)
    plt.tight_layout()
    plt.savefig('img/relation_win_mark.pdf')
    plt.show()


def computeChiSquareTest(cat1, cat2, name1, name2):
    '''

    :param cat1: list of categorical value to compare with cat2
    :param cat2: list of categorical value to compare with cat1
    :param name1: name of the categorical value refers to cat1
    :param name2: name of the categorical value refers to cat2

    Print all the information regarding chi square test computation
    '''
    df_chi = pd.DataFrame()
    df_chi[name1] = cat1
    df_chi[name2] = cat2
    contingency_table = pd.crosstab(df_chi[name1], df_chi[name2])
    print('contingency_table :-\n', contingency_table)

    # Observed Values
    Observed_Values = contingency_table.values
    print("Observed Values :-\n", Observed_Values)

    b = stats.chi2_contingency(contingency_table)
    Expected_Values = b[3]
    print("Expected Values :-\n", Expected_Values)

    no_of_rows = len(contingency_table.iloc[0:2, 0])
    no_of_columns = len(contingency_table.iloc[0, 0:2])
    ddof = (no_of_rows - 1) * (no_of_columns - 1)
    print("Degree of Freedom:-", ddof)
    alpha = 0.05

    from scipy.stats import chi2
    chi_square = sum([(o - e) ** 2. / e for o, e in zip(Observed_Values, Expected_Values)])
    chi_square_statistic = chi_square[0] + chi_square[1]
    print("chi-square statistic:-", chi_square_statistic)

    critical_value = chi2.ppf(q=1 - alpha, df=ddof)
    print('critical_value:', critical_value)

    # p-value
    p_value = 1 - chi2.cdf(x=chi_square_statistic, df=ddof)
    print('p-value:', p_value)

    print('Significance level: ', alpha)
    print('Degree of Freedom: ', ddof)
    print('chi-square statistic:', chi_square_statistic)
    print('critical_value:', critical_value)
    print('p-value:', p_value)
    if chi_square_statistic >= critical_value:
        print("Reject H0,There is a relationship between {} and {}".format(name1, name2))
    else:
        print("Retain H0,There is no relationship between {} and {}".format(name1, name2))

    if p_value <= alpha:
        print("Reject H0,There is a relationship between {} and {}".format(name1, name2))
    else:
        print("Retain H0,There is no relationship between {} and {}".format(name1, name2))

    return None


def createDataframeForMarkEvolution(df, metrics):
    '''

    :param df: Datframe of Player id match id, all the technical ffeatures and ecc..
    :param metrics: the metrics we want to extract in all the gamweeeks
    :return:

            A pandas dataframe representing a group by player id, gameweek. In columns there will be the player id;
            in rows there will be gameweek and each cell contain the metrics request
    '''

    metrics = metrics.upper()
    result = 0
    if metrics == 'GAZZETTA':
        result = df.groupby([df.player_id, df.match_day, df.gazzetta_score])
    if metrics == 'CORRIERE':
        result = df.groupby([df.player_id, df.match_day, df.corriere_score])
    if metrics == 'TUTTOSPORT':
        result = df.groupby([df.player_id, df.match_day, df.tuttosport_score])
    if metrics == 'FANTACALCIO':
        result = df.groupby([df.player_id, df.match_day, df.fantacalcio_score])
    if metrics == 'PLAYERRANK':
        result = df.groupby([df.player_id, df.match_day, df.ratings_total_alpha])
    # list of player id, index
    ids_player_for_time_series = set()
    # list of gameweek value
    gameweek = set()
    # list of marks during the gamesweek
    player_history = {}
    # for each row in the result group by
    for name, group in result:
        ids_player_for_time_series.add(name[0])
    for el in ids_player_for_time_series:
        player_history[el] = {}
    for name, group in result:
        player_history[name[0]][name[1]] = name[2]

    for el in player_history:
        for i in range(1, 39):
            if (i not in player_history[el].keys()):
                player_history[el][i] = 0

    new_player_history = {}
    for el in player_history:
        new_player_history[el] = {}
        for el1 in sorted(player_history[el].keys()):
            new_player_history[el][el1] = player_history[el][el1]

    series = pd.DataFrame.from_dict({(i): new_player_history[i]
                                     for i in new_player_history.keys()
                                     },
                                    orient='index')
    series = series.transpose()
    return series


def scatterPlayerEvolution(series, ids,name,season):
    '''
    Plot in a fancy way the player evolution for the gameweek in a season
    :param series: the pandas dataframe structured as row the gaemweek, columns the player ids and in the cells
                    the measures we want to evaluate
    :param ids: the id of the player we want to analyze
    :return:
            Scatter Plot and Stem Plot of a Player Performance
    '''

    x = series.index.values
    x = list(map(int, x))
    y = series[ids]
    y = list(map(float, y))
    plt.figure(figsize=(12, 6))

    ev = plt.stem(x, y, markerfmt=' ', linefmt='grey')
    ev = plt.scatter(x, y, color='red')

    ymax = max(y)
    indices = find(y, ymax)
    i = 0
    for el in indices:
        plt.scatter(el, ymax, s=400, color='orange', marker='*', label='Best Perfomance' if i == 0 else '_nolegend_')
        i = 1
    plt.xticks(x, fontsize=15)
    plt.yticks([0,2,4,6,8,10], fontsize=15)
    plt.ylim(0, 10)
    plt.annotate('0 values indicate matches not player', xy=(0, 9.5), fontsize=15)
    plt.ylabel('G Ratings of %s' %name, fontsize=label_size, labelpad=15)
    plt.xlabel('Gameweek season %s' %season, fontsize=label_size)
    plt.legend()
    plt.grid(alpha=0.25)    
    plt.tight_layout()
    plt.savefig('img/performance_evolution.pdf')
    plt.show()


def find(lst, a):
    '''

    :param lst: list of values
    :param a: max value to find in a list
    :return:
            all the index of the list that contains the maximum value
    '''
    return [i + 1 for i, x in enumerate(lst) if x == a]


def plotDifferentPerformanceLevelOfPlayer(series, numzeros, upperValue, downVal, metrics):
    '''

    :param series: the pandas dataframe structured as row the gaemweek, columns the player ids and in the cells
                    the measures we want to evaluate
    :param numzeros: maximum number of game that player didn't play
    :param upperValue: upper bound to define high skill players
    :param downVal: down bound to define low skill players
    :param downVal: down bound to define low skill players
    :return:
    '''
    # create transpose
    randomicity = series.transpose()
    lessSet = set()
    less = 0
    midSet = set()
    mid = 0
    highSet = set()
    high = 0
    for index, el in randomicity.iterrows():
        numberofzeros = 0
        maxVal = np.max(el)
        mini = sorted(el)
        minVal = 0
        for val in mini:
            if val == 0:
                numberofzeros += 1
            if val != 0:
                minVal = val
                break
        if numberofzeros < numzeros:
            if maxVal > upperValue:
                high += 1
                highSet.add(int(index))
            elif minVal < downVal:
                less += 1
                lessSet.add(int(index))
            else:
                mid += 1
                midSet.add(int(index))
    total = less + mid + high
    print('Percentage of Less than 4.5 : \t \t' + str(less / total))
    print('Percentage of Betw 4.5 and 7.5 : \t ' + str(mid / total))
    print('Percentage of Over 7.5 : \t \t' + str(high / total))

    print('Lenght of high ' + str(len(highSet)))
    print('Lenght of mid ' + str(len(midSet)))
    print('Lenght of less ' + str(len(lessSet)))

    # take first sample of less, mid and high
    sampleLess = random.choice(list(lessSet))
    sampleMid = random.choice(list(midSet))
    sampleHigh = random.choice(list(highSet))
    toFilter = [sampleLess, sampleMid, sampleHigh]
    Filter_df = randomicity[randomicity.index.isin(toFilter)]
    Filter_df = Filter_df.transpose()
    positions = []
    for el in Filter_df.columns:
        contatore = 0
        for el1 in toFilter:
            if (int(el) == int(el1)):
                if (contatore == 0):
                    positions.append('Low Performance')
                elif (contatore == 1):
                    positions.append('Mid Performance')
                elif (contatore == 2):
                    positions.append('High Performance')
            contatore += 1

    Filter_df.columns = positions
    Filter_df.plot(figsize=(10, 6), xticks=range(0, 39), legend=False,
                   title='Differences Between High, Mid and Low Player')
    plt.ylabel('%s Evaluation' % metrics)
    plt.xlabel('Gameweek')
    plt.legend()
    plt.annotate('0 values means not played', xy=(0, 0), fontsize=14)
    plotAveragePerfomanceForPlayerSets(randomicity, lessSet, midSet, highSet, metrics)


def plotAveragePerfomanceForPlayerSets(randomicity, lessSet, midSet, highSet, metrics):
    '''
    Auto Called function from the previous one
    :param randomicity: randomicity set from series
    :param lessSet: set of id considered as low level players
    :param midSet: set of id considered as mid level players
    :param highSet: set of id considered as high level players
    :return:
            Scatter Plot containing the different levels of players
    '''
    lessDict = {}
    midDict = {}
    highDict = {}

    for index, el in randomicity.iterrows():
        if (int(index) in lessSet):
            for i in range(1, 39):
                if (i in lessDict):
                    # avoid to count 0 in the mean
                    if (el[i] != 0):
                        lessDict[i]['marks'] += float(el[i])
                        lessDict[i]['count'] += 1
                else:
                    lessDict[i] = {}
                    lessDict[i]['marks'] = 0
                    lessDict[i]['count'] = 0
        if (int(index) in midSet):
            for i in range(1, 39):
                if (i in midDict):
                    # avoid to count 0 in the mean
                    if (el[i] != 0):
                        midDict[i]['marks'] += float(el[i])
                        midDict[i]['count'] += 1
                else:
                    midDict[i] = {}
                    midDict[i]['marks'] = 0
                    midDict[i]['count'] = 0
        if (int(index) in highSet):
            for i in range(1, 39):
                if (i in highDict):
                    # avoid to count 0 in the mean
                    if (el[i] != 0):
                        highDict[i]['marks'] += float(el[i])
                        highDict[i]['count'] += 1
                else:
                    highDict[i] = {}
                    highDict[i]['marks'] = 0
                    highDict[i]['count'] = 0
    lessValues = []
    days = []
    highValues = []
    midValues = []

    for key in lessDict:
        days.append(key)
        lessValues.append(float(lessDict[key]['marks'] / lessDict[key]['count']))
    for key in midDict:
        midValues.append(float(midDict[key]['marks'] / midDict[key]['count']))
    for key in highDict:
        highValues.append(float(highDict[key]['marks'] / highDict[key]['count']))
    plt.figure(figsize=(6, 6), dpi=80)

    plt.axhline(y=min(lessValues), color='lightblue', linestyle='-')
    plt.scatter(days, lessValues, color='lightblue', label='Low')
    plt.axhline(y=max(lessValues), color='lightblue', linestyle='-')

    plt.axhline(y=min(midValues), color='green', linestyle='-')
    plt.scatter(days, midValues, color='green', label='Medium')
    plt.axhline(y=max(midValues), color='green', linestyle='-')

    plt.axhline(y=min(highValues), color='red', linestyle='-')
    plt.scatter(days, highValues, color='red', label='High')
    plt.axhline(y=max(highValues), color='red', linestyle='-')

    plt.xlabel('Gameweek', fontsize= label_size)
    plt.ylabel('Mean %s Rating' % metrics[0].upper(), fontsize = label_size, labelpad=15)
    plt.xticks([0,5,10,15,20,25,30,35,40], fontsize=20)
    plt.yticks([5.6,5.8,6.0,6.2,6.4,6.6],fontsize=15)
    #plt.title('Mean Performance Per Player Category')
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.10), fontsize=12,frameon=True, shadow=True, handlelength=1.6, handletextpad=0.4, ncol=3)
    plt.tight_layout()
    plt.savefig('img/category_performance.pdf')
    plt.show()


def distributionRandomEvolutionVsOriginalDistribution(series, metrics, run):
    '''

    :param series: the pandas dataframe structured as row the gaemweek, columns the player ids and in the cells
                    the measures we want to evaluate
    :param metrics: string of the metrics used to consider the evolution of the player in the gameweek
    :param run: number of times the randomization need to be applyied
    :return:
            Plot that compare random distribution against original distribution of first best performances
    '''
    if (run == 1):
        # computation to retrive first best performance for player for each gameweek
        forDistribution = pd.DataFrame()
        for column in series:
            booleanArrays = []
            maximum = max(series[column])
            for el in series[column]:
                if (el == maximum):
                    booleanArrays.append(1)
                    break
                else:
                    booleanArrays.append(0)
            for i in range(len(booleanArrays), 38):
                booleanArrays.append(0)
            forDistribution[column] = booleanArrays
        forDistribution = forDistribution.transpose()

        # shuffle the gameweek performances of each player
        # mantain distribution of marks
        randomShuffle = series.copy(deep=True)
        randomShuffle = randomShuffle.transpose()
        columns = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27,
                   28,
                   29, 30, 31, 32, 33, 34, 35, 36, 37]
        randomShuffle.columns = columns

        for index, row in randomShuffle.iterrows():
            random.shuffle(row)
            for i in range(0, 38):
                randomShuffle.at[index, i] = float(row[i])
        randomShuffle.columns = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
                                 25,
                                 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38]
        randomShuffle = randomShuffle.transpose()

        # computation for retrieve the first best performance in shuffled order for each player
        forDistributionShuffled = pd.DataFrame()
        for column in randomShuffle:
            booleanArrays = []
            maximum = max(randomShuffle[column])
            for el in randomShuffle[column]:
                if (el == maximum):
                    booleanArrays.append(1)
                    break
                else:
                    booleanArrays.append(0)
            for i in range(len(booleanArrays), 38):
                booleanArrays.append(0)
            forDistributionShuffled[column] = booleanArrays
        forDistributionShuffled = forDistributionShuffled.transpose()

        # aggregate for all players
        occurences = []
        occurencesShuffled = []
        for column in forDistribution:
            mathAccumulator = 0
            mathAccumulatorShuffled = 0
            for el in forDistribution[column]:
                mathAccumulator += el
            for el in forDistributionShuffled[column]:
                mathAccumulatorShuffled += el
            occurences.append(mathAccumulator)
            occurencesShuffled.append(mathAccumulatorShuffled)
        days = []
        for el in range(1, 39):
            days.append(el)

        # NORMALIZATION PART, IF NEEDED
        # compact everything in a single list
        occurences.extend(occurencesShuffled)
        # reshape to aplly min max scaler
        X_occurences = np.array(occurences).reshape(-1, 1)
        # scale xy value to reduce distances in order to find out a clear difference
        scaler = MinMaxScaler()
        # scale
        occurences = scaler.fit_transform(X_occurences)

        # divide again values between random and original one
        rescaledOccurences = []
        for i in range(0, 38):
            rescaledOccurences.append(occurences[i][0])

        rescaledOccurencesShuffled = []
        for i in range(38, len(occurences)):
            rescaledOccurencesShuffled.append(occurences[i][0])
        occurences = rescaledOccurences
        occurencesShuffled = rescaledOccurencesShuffled

        '''
        #COMPACT X AXIS
        meanVal = []
        first = True
        for i in range(0, 37):
            if(first):
                first = False
                meanValue = (occurences[i:i+2][0] + occurences[i:i+2][1])/2
                meanVal.append(meanValue)
            else:
                meanValue = (occurences[i:i+2][0] + occurences[i:i+2][1])/2
                meanVal.append(meanValue)
        meanVal.append(occurences[37])
        occurences = meanVal
    
        meanVal = []
        first = True
        for i in range(0, 37):
            if(first):
                first = False
                meanValue = (occurencesShuffled[i:i+2][0] + occurencesShuffled[i:i+2][1])/2
                meanVal.append(meanValue)
            else:
                meanValue = (occurencesShuffled[i:i+2][0] + occurencesShuffled[i:i+2][1])/2
                meanVal.append(meanValue)
        meanVal.append(occurencesShuffled[37])
        occurencesShuffled = meanVal
        '''
        # Draw Plot
        plt.figure(figsize=(12, 6))
        plt.plot(days, occurences, color='tab:blue', label='Fisrt Best Perfromances')
        plt.plot(days, occurencesShuffled, color='grey', label='Randomized First Best Perfomance', linestyle='dashed')
        ymax = max(occurences)
        ymaxShuffled = max(occurencesShuffled)
        indices = find(occurences, ymax)
        indicesShuffled = find(occurencesShuffled, ymaxShuffled)
        for el in indices:
            plt.scatter(el, ymax - 0.02, marker=mpl.markers.CARETUPBASE, color='tab:green', s=100, label='Peaks')
        for el in indicesShuffled:
            plt.scatter(el, ymaxShuffled - 0.02, marker=mpl.markers.CARETUPBASE, color='grey', s=100,
                        label='Random Peaks')
        # Decoration
        #plt.title("Distribution of First Best Performances of %s score in Gameweeks" % metrics, fontsize=22)
        plt.yticks(fontsize=15)
        plt.xticks(fontsize = 20)
        plt.ylim(0, 1)
        plt.xlabel('Gameweek', fontsize=label_size)
        plt.ylabel('Normalized Best Performances', fontsize = label_size)

        # Lighten borders
        plt.gca().spines["top"].set_alpha(.0)
        plt.gca().spines["bottom"].set_alpha(.3)
        plt.gca().spines["right"].set_alpha(.0)
        plt.gca().spines["left"].set_alpha(.3)

        plt.legend(fontsize=15, frameon=True, shadow=True, handlelength=1.6, handletextpad=0.4, loc='upper right')
        plt.grid(axis='y', alpha=.3)
        plt.tight_layout()
        plt.savefig('img/comparison_real_random.pdf')
        plt.show()

        print('DTW computation between the two time series..')
        print('Time series 1 regarding number of first best performances')
        print('Time series 2 ragarding shuffled perfomance w.r.t. number of first best perfomances')
        from dtw import dtw

        x = np.array(occurences).reshape(-1, 1)
        y = np.array(occurencesShuffled).reshape(-1, 1)

        manhattan_distance = lambda x, y: np.abs(x - y)

        d, cost_matrix, acc_cost_matrix, path = dtw(x, y, dist=manhattan_distance)

        print('Only insertion Cost is keep..')
        print('Reallignment Cost :' + str(d))

        plt.imshow(acc_cost_matrix.T, origin='lower', cmap='inferno', interpolation='nearest')
        plt.plot(path[0], path[1], 'w')
        plt.tight_layout()
        plt.savefig('img/comparison_real_random_dtw.pdf')
        plt.show()
    else:
        # computation to retrive first best performance for player for each gameweek
        forDistribution = pd.DataFrame()
        for column in series:
            booleanArrays = []
            maximum = max(series[column])
            for el in series[column]:
                if (el == maximum):
                    booleanArrays.append(1)
                    break
                else:
                    booleanArrays.append(0)
            for i in range(len(booleanArrays), 38):
                booleanArrays.append(0)
            forDistribution[column] = booleanArrays
        forDistribution = forDistribution.transpose()

        # shuffle the gameweek performances of each player
        # mantain distribution of marks
        # for each run
        occurencesShuffled = []
        first = True
        for i in range(0, run):
            randomShuffle = series.copy(deep=True)
            randomShuffle = randomShuffle.transpose()
            columns = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26,
                       27,
                       28,
                       29, 30, 31, 32, 33, 34, 35, 36, 37]
            randomShuffle.columns = columns
            # make a random shuffle
            for index, row in randomShuffle.iterrows():
                random.shuffle(row)
                for i in range(0, 38):
                    randomShuffle.at[index, i] = float(row[i])
            randomShuffle.columns = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
                                     24,
                                     25,
                                     26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38]
            randomShuffle = randomShuffle.transpose()

            # computation for retrieve the first best performance in shuffled order for each player
            forDistributionShuffled = pd.DataFrame()
            for column in randomShuffle:
                booleanArrays = []
                maximum = max(randomShuffle[column])
                for el in randomShuffle[column]:
                    if (el == maximum):
                        booleanArrays.append(1)
                        break
                    else:
                        booleanArrays.append(0)
                for i in range(len(booleanArrays), 38):
                    booleanArrays.append(0)
                forDistributionShuffled[column] = booleanArrays
            forDistributionShuffled = forDistributionShuffled.transpose()

            for column in forDistributionShuffled:
                mathAccumulatorShuffled = 0
                for el in forDistributionShuffled[column]:
                    mathAccumulatorShuffled += el
                if (first):
                    occurencesShuffled.append(mathAccumulatorShuffled)
                else:
                    occurencesShuffled[column] = (occurencesShuffled[column] + mathAccumulatorShuffled) / 2
            first = False

        # aggregate for all players
        occurences = []
        for column in forDistribution:
            mathAccumulator = 0
            for el in forDistribution[column]:
                mathAccumulator += el
            occurences.append(mathAccumulator)
        days = []
        for el in range(1, 39):
            days.append(el)

        # NORMALIZATION PART, IF NEEDED
        # compact everything in a single list
        occurences.extend(occurencesShuffled)
        # reshape to aplly min max scaler
        X_occurences = np.array(occurences).reshape(-1, 1)
        # scale xy value to reduce distances in order to find out a clear difference
        scaler = MinMaxScaler()
        # scale
        occurences = scaler.fit_transform(X_occurences)

        # divide again values between random and original one
        rescaledOccurences = []
        for i in range(0, 38):
            rescaledOccurences.append(occurences[i][0])

        rescaledOccurencesShuffled = []
        for i in range(38, len(occurences)):
            rescaledOccurencesShuffled.append(occurences[i][0])
        occurences = rescaledOccurences
        occurencesShuffled = rescaledOccurencesShuffled

        '''
        #COMPACT X AXIS
        meanVal = []
        first = True
        for i in range(0, 37):
            if(first):
                first = False
                meanValue = (occurences[i:i+2][0] + occurences[i:i+2][1])/2
                meanVal.append(meanValue)
            else:
                meanValue = (occurences[i:i+2][0] + occurences[i:i+2][1])/2
                meanVal.append(meanValue)
        meanVal.append(occurences[37])
        occurences = meanVal

        meanVal = []
        first = True
        for i in range(0, 37):
            if(first):
                first = False
                meanValue = (occurencesShuffled[i:i+2][0] + occurencesShuffled[i:i+2][1])/2
                meanVal.append(meanValue)
            else:
                meanValue = (occurencesShuffled[i:i+2][0] + occurencesShuffled[i:i+2][1])/2
                meanVal.append(meanValue)
        meanVal.append(occurencesShuffled[37])
        occurencesShuffled = meanVal
        '''
        # Draw Plot
        plt.figure(figsize=(12, 6), dpi=80)
        plt.plot(days, occurences, color='tab:blue', label='Fisrt Best Perfromances')
        plt.plot(days, occurencesShuffled, color='grey', label='Randomized First Best Perfomance', linestyle='dashed')
        ymax = max(occurences)
        ymaxShuffled = max(occurencesShuffled)
        indices = find(occurences, ymax)
        indicesShuffled = find(occurencesShuffled, ymaxShuffled)
        for el in indices:
            plt.scatter(el, ymax - 0.02, marker=mpl.markers.CARETUPBASE, color='tab:green', s=100, label='Peaks')
        for el in indicesShuffled:
            plt.scatter(el, ymaxShuffled - 0.02, marker=mpl.markers.CARETUPBASE, color='grey', s=100,
                        label='Random Peaks')
       # Decoration
        #plt.title("Distribution of First Best Performances of %s score in Gameweeks" % metrics, fontsize=22)
        plt.yticks(fontsize=15)
        plt.xticks(fontsize = 20)
        plt.ylim(0, 1)
        plt.xlabel('Gameweek', fontsize=label_size)
        plt.ylabel('N. Best Performances', fontsize = label_size)

        # Lighten borders
        plt.gca().spines["top"].set_alpha(.0)
        plt.gca().spines["bottom"].set_alpha(.3)
        plt.gca().spines["right"].set_alpha(.0)
        plt.gca().spines["left"].set_alpha(.3)

        plt.legend(fontsize=15, frameon=True, shadow=True, handlelength=1.6, handletextpad=0.4, loc='upper right')
        plt.grid(axis='y', alpha=.3)
        plt.tight_layout()
        plt.savefig('img/comparison_real_random.pdf')
        plt.show()

        print('DTW computation between the two time series..')
        print('Time series 1 regarding number of first best performances')
        print(
            'Time series 2 ragarding %s run of shuffled perfomance w.r.t. number of first best perfomances' % str(run))
        from dtw import dtw

        x = np.array(occurences).reshape(-1, 1)
        y = np.array(occurencesShuffled).reshape(-1, 1)

        manhattan_distance = lambda x, y: np.abs(x - y)

        d, cost_matrix, acc_cost_matrix, path = dtw(x, y, dist=manhattan_distance)

        print('Only insertion Cost is keep..')
        print('Reallignment Cost :' + str(d))
        
        plt.figure(figsize=(8,8))
        plt.imshow(acc_cost_matrix.T, origin='lower', cmap='inferno', interpolation='nearest')
        plt.plot(path[0], path[1], 'w')
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=15)
        plt.ylabel('Best Performance Shuffled', fontsize=label_size)
        plt.xlabel('Best Performance Original', fontsize=label_size)
        plt.tight_layout()
        plt.savefig('img/comparison_real_random_dtw.pdf')
        plt.show()


def plotTeamVersusPlayerPerformance(df, listPlayerId, metrics):
    '''

    :param df: Datframe of Player id match id, all the technical ffeatures and ecc..
    :param listPlayerId: List of id of player we want to plot
    :param metrics: possibile options Gazzetta or Playerrank metrics
    :return:
            Scatter plot for each player insert with the relative performance against teams
    '''
    # Draw Plot
    plt.figure(figsize=(17, 6), dpi=80)
    plt.ylim(0, 10)
    for ids in listPlayerId:
        player = df[df['player_id'] == ids]
        player = player.sort_values(by=['match_day'])
        nameplayer = list(player['player_name_fantacalcio'])[0]
        against = player['contextual_against_club_name']
        if (metrics.upper() == 'GAZZETTA'):
            marks = player['gazzetta_score']
        if (metrics.upper() == 'PLAYERRANK'):
            marks = player['ratings_total_alpha']
        plt.scatter(against, marks, label=nameplayer)
    plt.legend(loc='upper left')
    plt.ylabel('%s Score' % metrics)
    plt.xlabel('Team Against')
    plt.xticks(fontsize=9)
    plt.show()


def bestPerformanceAgainstTeamName(df, series, classpath, metrics):
    '''

    :param df:
    :param series:
    :param classpath: path where there is the matches json
    :param metrics:
    :return:
    '''
    # computation to retrive first best performance for player for each gameweek
    forDistribution = pd.DataFrame()
    for column in series:
        booleanArrays = []
        maximum = max(series[column])
        for el in series[column]:
            if (el == maximum):
                booleanArrays.append(1)
                break
            else:
                booleanArrays.append(0)
        for i in range(len(booleanArrays), 38):
            booleanArrays.append(0)
        forDistribution[column] = booleanArrays
    forDistribution = forDistribution.transpose()
    # compute for each player the dates where he outperformed
    playerBestDays = {}
    for index, player in forDistribution.iterrows():
        contator = 1
        listBestDays = []
        for mark in player:
            if (mark == 1):
                listBestDays.append(contator)
            contator = contator + 1
        playerBestDays[index] = listBestDays
        # for each record of the dataset we want to extract the team against
    # for best performance analysis
    teamAgainstBestName = {}
    relationTeamWYID = {}
    for row in df.values:
        # l'id nel dizionario
        if (row[3] in playerBestDays):
            # la giornata e nella chiave del dizionario
            if (row[1] in playerBestDays[row[3]]):
                if (row[149] in teamAgainstBestName):
                    teamAgainstBestName[row[149]] += 1
                else:
                    teamAgainstBestName[row[149]] = 1
    # extract also the relation name of the team and wyids for extract
    # the podium
    for row in df.values:
        if (row[148] in teamAgainstBestName):
            relationTeamWYID[row[148]] = row[2]

    with open('%s' % classpath, encoding='utf-8-sig') as f_input:
        matchesItaly = pd.read_json(f_input)

    # classifica computation from matches
    classifica = {}
    for team in relationTeamWYID:
        classifica[team] = 0
        for el in matchesItaly.values:
            for teamsData in el[3]:
                # the team is inside the match
                if (int(relationTeamWYID[team]) == int(teamsData)):
                    # winner
                    if (relationTeamWYID[team] == el[6]):
                        classifica[team] += 3
                    if (el[6] == 0):
                        classifica[team] += 1

    # order the classifica based on points
    from collections import OrderedDict
    from operator import itemgetter
    classificaOrdinata = OrderedDict(sorted(classifica.items(), key=itemgetter(1), reverse=True))

    # pass through all the classifica in order to create sorted arrays
    teams = []
    numberOfBestPerformances = []
    for el in classificaOrdinata:
        if (el in teamAgainstBestName):
            teams.append(el)
            numberOfBestPerformances.append(teamAgainstBestName[el])
    # Draw Plot
    plt.figure(figsize=(20, 6), dpi=80)
    plt.plot(teams, numberOfBestPerformances, color='tab:blue', label='Best Perfromances')
    ymax = max(numberOfBestPerformances)
    indices = find(numberOfBestPerformances, ymax)
    for el in indices:
        plt.scatter(teams[el - 1], ymax, marker=mpl.markers.CARETUPBASE, color='tab:green', s=100, label='Peaks')

    # Decoration
    plt.title("Distribution of Best Performances based on %s Against Team" % metrics, fontsize=22)
    plt.yticks(fontsize=12, alpha=.7)

    # Lighten borders
    plt.gca().spines["top"].set_alpha(.0)
    plt.gca().spines["bottom"].set_alpha(.3)
    plt.gca().spines["right"].set_alpha(.0)
    plt.gca().spines["left"].set_alpha(.3)

    plt.xlabel('Team Against')
    plt.ylabel('Number of Best Performances of Player')

    plt.legend(loc='upper left')
    plt.grid(axis='y', alpha=.3)
    plt.show()
    meanScoreAgainstTeam(df, classificaOrdinata, metrics)


def meanScoreAgainstTeam(df, classificaOrdinata, metrics):
    meanEvaluation = []
    teams = []
    for el in classificaOrdinata:
        teams.append(el)
        # take the dataset of the against player
        subAgainst = df[df['contextual_against_club_name'] == el]
        numberOfPlayer = 0
        totalEvaluations = 0
        for el in subAgainst.values:
            numberOfPlayer += 1
            totalEvaluations += el[155]
        meanEvaluation.append(totalEvaluations / numberOfPlayer)

    # Draw Plot
    plt.figure(figsize=(20, 6), dpi=80)
    plt.plot(teams, meanEvaluation, color='tab:blue', label='Best Perfromances')
    plt.scatter(teams, meanEvaluation, color='tab:blue', label='Best Perfromances')
    ymax = max(meanEvaluation)
    indices = find(meanEvaluation, ymax)
    for el in indices:
        plt.scatter(teams[el - 1], ymax, marker=mpl.markers.CARETUPBASE, color='tab:green', s=100, label='Peaks')

    # Decoration
    plt.title("Distribution of Mean %s Score Against Team" % metrics, fontsize=22)
    plt.yticks(fontsize=12, alpha=.7)

    # Lighten borders
    plt.gca().spines["top"].set_alpha(.0)
    plt.gca().spines["bottom"].set_alpha(.3)
    plt.gca().spines["right"].set_alpha(.0)
    plt.gca().spines["left"].set_alpha(.3)

    plt.xlabel('Team Against')
    plt.ylabel('Average performace %s score of against player' % metrics)

    plt.legend(loc='upper left')
    plt.grid(axis='y', alpha=.3)
    plt.show()


def plotBeforeandAfterHighestImpactInEachPlayerCategory(series, n0, upperValue, downValue, metrics):
    '''

    :param series:
    :param n0: number of games the  player need to at least play
    :param upperValue: maximum value to consider player as high performance
    :param downValue: minimum value to consider player as high performance
    :param metrics: the name of the metrics used to verify the player category level
    :return:
    '''
    #create a copy
    nStarDistribution = series.copy()
    nStarDistribution = nStarDistribution.transpose()
    lessSet = set()
    less = 0
    midSet = set()
    mid = 0
    highSet = set()
    high = 0
    #iterate each player in order to distinct bewteen mid less or high player
    for index, el in nStarDistribution.iterrows():
        numberofzeros = 0
        maxVal = np.max(el)
        mini = sorted(el)
        minVal = 0
        for val in mini:
            if val == 0:
                numberofzeros += 1
            if val != 0:
                minVal = val
                break
        if numberofzeros < n0:
            if maxVal > upperValue:
                high += 1
                highSet.add(int(index))
            elif minVal < downValue:
                less += 1
                lessSet.add(int(index))
            else:
                mid += 1
                midSet.add(int(index))
    #all values needed
    nStarHigh = 0
    nStarMid = 0
    nStarLess = 0
    minusTwoHigh = 0
    minusTwoMid = 0
    minusTwoLess = 0
    plusTwoHigh = 0
    plusTwoMid = 0
    plusTwoLess = 0
    minusFourHigh = 0
    minusFourMid = 0
    minusFourLess = 0
    plusFourHigh = 0
    plusFourMid = 0
    plusFourLess = 0
    minusSixHigh = 0
    minusSixMid = 0
    minusSixLess = 0
    plusSixHigh = 0
    plusSixMid = 0
    plusSixLess = 0
    minusEightHigh = 0
    minusEightMid = 0
    minusEightLess = 0
    plusEightHigh = 0
    plusEightMid = 0
    plusEightLess = 0

    # for each player
    for index, el in nStarDistribution.iterrows():
        # check id appartenance in less set
        if (int(index) in lessSet):
            if (nStarLess == 0):
                nStarLess = max(el)
                indices = find(el, nStarLess)
            else:
                nStarLess = (nStarLess + max(el)) / 2
                indices = find(el, nStarLess)
            if (minusEightLess == 0):
                try:
                    if (el[indices[0] - 8] != 0):
                        minusEightLess = el[indices[0] - 8]
                except KeyError as e:
                    doNothing = ''
                except IndexError as e:
                    doNothing = ''
            else:
                try:
                    if (el[indices[0] - 8] != 0):
                        minusEightLess = (minusEightLess + el[indices[0] - 8]) / 2
                except KeyError as e:
                    doNothing = ''
                except IndexError as e:
                    doNothing = ''
            if (minusSixLess == 0):
                try:
                    if (el[indices[0] - 6] != 0):
                        minusSixLess = el[indices[0] - 6]
                except KeyError as e:
                    doNothing = ''
                except IndexError as e:
                    doNothing = ''
            else:
                try:
                    if (el[indices[0] - 6] != 0):
                        minusSixLess = (minusSixLess + el[indices[0] - 6]) / 2
                except KeyError as e:
                    doNothing = ''
                except IndexError as e:
                    doNothing = ''
            if (minusTwoLess == 0):
                try:
                    if (el[indices[0] - 2] != 0):
                        minusTwoLess = el[indices[0] - 2]
                except KeyError as e:
                    doNothing = ''
                except IndexError as e:
                    doNothing = ''
            else:
                try:
                    if (el[indices[0] - 2] != 0):
                        minusTwoLess = (minusTwoLess + el[indices[0] - 2]) / 2
                except KeyError as e:
                    doNothing = ''
                except IndexError as e:
                    doNothing = ''
            if (minusFourLess == 0):
                try:
                    if (el[indices[0] - 4] != 0):
                        minusFourLess = el[indices[0] - 4]
                except KeyError as e:
                    doNothing = ''
                except IndexError as e:
                    doNothing = ''
            else:
                try:
                    if (el[indices[0] - 4] != 0):
                        minusFourLess = (minusFourLess + el[indices[0] - 4]) / 2
                except KeyError as e:
                    doNothing = ''
                except IndexError as e:
                    doNothing = ''
            if (plusEightLess == 0):
                try:
                    if (el[indices[0] + 8] != 0):
                        plusEightLess = el[indices[0] + 8]
                except KeyError as e:
                    doNothing = ''
                except IndexError as e:
                    doNothing = ''
            else:
                try:
                    if (el[indices[0] + 8] != 0):
                        plusEightLess = (plusEightLess + el[indices[0] + 8]) / 2
                except KeyError as e:
                    doNothing = ''
                except IndexError as e:
                    doNothing = ''
            if (plusSixLess == 0):
                try:
                    if (el[indices[0] + 6] != 0):
                        plusSixLess = el[indices[0] + 6]
                except KeyError as e:
                    doNothing = ''
                except IndexError as e:
                    doNothing = ''
            else:
                try:
                    if (el[indices[0] + 6] != 0):
                        plusSixLess = (plusSixLess + el[indices[0] + 6]) / 2
                except KeyError as e:
                    doNothing = ''
                except IndexError as e:
                    doNothing = ''
            if (plusTwoLess == 0):
                try:
                    if (el[indices[0] + 2] != 0):
                        plusTwoLess = el[indices[0] + 2]
                except KeyError as e:
                    doNothing = ''
                except IndexError as e:
                    doNothing = ''
            else:
                try:
                    if (el[indices[0] + 2] != 0):
                        plusTwoLess = (plusTwoLess + el[indices[0] + 2]) / 2
                except KeyError as e:
                    doNothing = ''
                except IndexError as e:
                    doNothing = ''
            if (plusFourLess == 0):
                try:
                    if (el[indices[0] + 4] != 0):
                        plusFourLess = el[indices[0] + 4]
                except KeyError as e:
                    doNothing = ''
                except IndexError as e:
                    doNothing = ''
            else:
                try:
                    if (el[indices[0] + 4] != 0):
                        plusFourLess = (plusFourLess + el[indices[0] + 4]) / 2
                except KeyError as e:
                    doNothing = ''
                except IndexError as e:
                    doNothing = ''
        # check if the player is in the mid set
        if (int(index) in midSet):
            if (nStarMid == 0):
                nStarMid = max(el)
                indices = find(el, nStarMid)
            else:
                nStarMid = (nStarMid + max(el)) / 2
                indices = find(el, nStarMid)
            if (minusEightMid == 0):
                try:
                    if (el[indices[0] - 8] != 0):
                        minusEightMid = el[indices[0] - 8]
                except KeyError as e:
                    doNothing = ''
                except IndexError as e:
                    doNothing = ''
            else:
                try:
                    if (el[indices[0] - 8] != 0):
                        minusEightMid = (minusEightMid + el[indices[0] - 8]) / 2
                except KeyError as e:
                    doNothing = ''
                except IndexError as e:
                    doNothing = ''
            if (minusSixMid == 0):
                try:
                    if (el[indices[0] - 6] != 0):
                        minusSixMid = el[indices[0] - 6]
                except KeyError as e:
                    doNothing = ''
                except IndexError as e:
                    doNothing = ''
            else:
                try:
                    if (el[indices[0] - 6] != 0):
                        minusSixMid = (minusSixMid + el[indices[0] - 6]) / 2
                except KeyError as e:
                    doNothing = ''
                except IndexError as e:
                    doNothing = ''
            if (minusTwoMid == 0):
                try:
                    if (el[indices[0] - 2] != 0):
                        minusTwoMid = el[indices[0] - 2]
                except KeyError as e:
                    doNothing = ''
                except IndexError as e:
                    doNothing = ''
            else:
                try:
                    if (el[indices[0] - 2] != 0):
                        minusTwoMid = (minusTwoMid + el[indices[0] - 2]) / 2
                except KeyError as e:
                    doNothing = ''
                except IndexError as e:
                    doNothing = ''
            if (minusFourMid == 0):
                try:
                    if (el[indices[0] - 4] != 0):
                        minusFourMid = el[indices[0] - 4]
                except KeyError as e:
                    doNothing = ''
                except IndexError as e:
                    doNothing = ''
            else:
                try:
                    if (el[indices[0] - 4] != 0):
                        minusFourMid = (minusFourMid + el[indices[0] - 4]) / 2
                except KeyError as e:
                    doNothing = ''
                except IndexError as e:
                    doNothing = ''
            if (plusEightMid == 0):
                try:
                    if (el[indices[0] + 8] != 0):
                        plusEightMid = el[indices[0] + 8]
                except KeyError as e:
                    doNothing = ''
                except IndexError as e:
                    doNothing = ''
            else:
                try:
                    if (el[indices[0] + 8] != 0):
                        plusEightMid = (plusEightMid + el[indices[0] + 8]) / 2
                except KeyError as e:
                    doNothing = ''
                except IndexError as e:
                    doNothing = ''
            if (plusSixMid == 0):
                try:
                    if (el[indices[0] + 6] != 0):
                        plusSixMid = el[indices[0] + 6]
                except KeyError as e:
                    doNothing = ''
                except IndexError as e:
                    doNothing = ''
            else:
                try:
                    if (el[indices[0] + 6] != 0):
                        plusSixMid = (plusSixMid + el[indices[0] + 6]) / 2
                except KeyError as e:
                    doNothing = ''
                except IndexError as e:
                    doNothing = ''
            if (plusTwoMid == 0):
                try:
                    if (el[indices[0] + 2] != 0):
                        plusTwoMid = el[indices[0] + 2]
                except KeyError as e:
                    doNothing = ''
                except IndexError as e:
                    doNothing = ''
            else:
                try:
                    if (el[indices[0] + 2] != 0):
                        plusTwoMid = (plusTwoMid + el[indices[0] + 2]) / 2
                except KeyError as e:
                    doNothing = ''
                except IndexError as e:
                    doNothing = ''
            if (plusFourMid == 0):
                try:
                    if (el[indices[0] + 4] != 0):
                        plusFourMid = el[indices[0] + 4]
                except KeyError as e:
                    doNothing = ''
                except IndexError as e:
                    doNothing = ''
            else:
                try:
                    if (el[indices[0] + 4] != 0):
                        plusFourMid = (plusFourMid + el[indices[0] + 4]) / 2
                except KeyError as e:
                    doNothing = ''
                except IndexError as e:
                    doNothing = ''

        # check if the player is in the high  set
        if (int(index) in highSet):
            if (nStarHigh == 0):
                nStarHigh = max(el)
                indices = find(el, nStarHigh)
            else:
                nStarHigh = (nStarHigh + max(el)) / 2
                indices = find(el, nStarHigh)
            if (minusEightHigh == 0):
                try:
                    if (el[indices[0] - 8] != 0):
                        minusEightHigh = el[indices[0] - 8]
                except KeyError as e:
                    doNothing = ''
                except IndexError as e:
                    doNothing = ''
            else:
                try:
                    if (el[indices[0] - 8] != 0):
                        minusEightHigh = (minusEightHigh + el[indices[0] - 8]) / 2
                except KeyError as e:
                    doNothing = ''
                except IndexError as e:
                    doNothing = ''
            if (minusSixHigh == 0):
                try:
                    if (el[indices[0] - 6] != 0):
                        minusSixHigh = el[indices[0] - 6]
                except KeyError as e:
                    doNothing = ''
                except IndexError as e:
                    doNothing = ''
            else:
                try:
                    if (el[indices[0] - 6] != 0):
                        minusSixHigh = (minusSixHigh + el[indices[0] - 6]) / 2
                except KeyError as e:
                    doNothing = ''
                except IndexError as e:
                    doNothing = ''
            if (minusTwoHigh == 0):
                try:
                    if (el[indices[0] - 2] != 0):
                        minusTwoHigh = el[indices[0] - 2]
                except KeyError as e:
                    doNothing = ''
                except IndexError as e:
                    doNothing = ''
            else:
                try:
                    if (el[indices[0] - 2] != 0):
                        minusTwoHigh = (minusTwoHigh + el[indices[0] - 2]) / 2
                except KeyError as e:
                    doNothing = ''
                except IndexError as e:
                    doNothing = ''
            if (minusFourHigh == 0):
                try:
                    if (el[indices[0] - 4] != 0):
                        minusFourHigh = el[indices[0] - 4]
                except KeyError as e:
                    doNothing = ''
                except IndexError as e:
                    doNothing = ''
            else:
                try:
                    if (el[indices[0] - 4] != 0):
                        minusFourHigh = (minusFourHigh + el[indices[0] - 4]) / 2
                except KeyError as e:
                    doNothing = ''
                except IndexError as e:
                    doNothing = ''
            if (plusEightHigh == 0):
                try:
                    if (el[indices[0] + 8] != 0):
                        plusEightHigh = el[indices[0] + 8]
                except KeyError as e:
                    doNothing = ''
                except IndexError as e:
                    doNothing = ''
            else:
                try:
                    if (el[indices[0] + 8] != 0):
                        plusEightHigh = (plusEightHigh + el[indices[0] + 8]) / 2
                except KeyError as e:
                    doNothing = ''
                except IndexError as e:
                    doNothing = ''
            if (plusSixHigh == 0):
                try:
                    if (el[indices[0] + 6] != 0):
                        plusSixHigh = el[indices[0] + 6]
                except KeyError as e:
                    doNothing = ''
                except IndexError as e:
                    doNothing = ''
            else:
                try:
                    if (el[indices[0] + 6] != 0):
                        plusSixHigh = (plusSixHigh + el[indices[0] + 6]) / 2
                except KeyError as e:
                    doNothing = ''
                except IndexError as e:
                    doNothing = ''
            if (plusTwoHigh == 0):
                try:
                    if (el[indices[0] + 2] != 0):
                        plusTwoHigh = el[indices[0] + 2]
                except KeyError as e:
                    doNothing = ''
                except IndexError as e:
                    doNothing = ''
            else:
                try:
                    if (el[indices[0] + 2] != 0):
                        plusTwoHigh = (plusTwoHigh + el[indices[0] + 2]) / 2
                except KeyError as e:
                    doNothing = ''
                except IndexError as e:
                    doNothing = ''
            if (plusFourHigh == 0):
                try:
                    if (el[indices[0] + 4] != 0):
                        plusFourHigh = el[indices[0] + 4]
                except KeyError as e:
                    doNothing = ''
                except IndexError as e:
                    doNothing = ''
            else:
                try:
                    if (el[indices[0] + 4] != 0):
                        plusFourHigh = (plusFourHigh + el[indices[0] + 4]) / 2
                except KeyError as e:
                    doNothing = ''
                except IndexError as e:
                    doNothing = ''
    analyze = ['-8', '-6', '-4', '-2', 'N*', '2', '4', '6', '8']
    mid = [minusEightMid, minusSixMid, minusFourMid, minusTwoMid, nStarMid, plusTwoMid, plusFourMid, plusSixMid,
           plusEightMid]
    less = [minusEightLess, minusSixLess, minusFourLess, minusTwoLess, nStarLess, plusTwoLess, plusFourLess,
            plusSixLess, plusEightLess]
    high = [minusEightHigh, minusSixHigh, minusFourHigh, minusTwoHigh, nStarHigh, plusTwoHigh, plusFourHigh,
            plusSixHigh, plusEightHigh]
    # Draw Plot
    plt.figure(figsize=figsize, dpi=80)
    plt.plot(analyze, less, color='blue')
    plt.scatter(analyze, less, color='blue', label='Less Performance')
    plt.plot(analyze, mid, color='yellow')
    plt.scatter(analyze, mid, color='yellow', label='Mid Performance')
    plt.plot(analyze, high, color='red')
    plt.scatter(analyze, high, color='red', label='Best Performance')

    # Decoration
    plt.title("Average impact of performances before and after the highest impact (%s)" %metrics, fontsize=22)
    plt.yticks(fontsize=12, alpha=.7)

    plt.xlabel('Timelapse')
    plt.ylabel('Mean impact of players performances')

    plt.legend(loc='upper left')
    plt.grid(axis='y', alpha=.3)
    plt.show()

def table_statistics(df):
    '''
    compute r, ks and RMSE for role for newspaper couples
    
    Param:
        df
    Return:
        dictionary with results
    '''
    data = {}
    for roles in ['A', 'C', 'D', 'P']:
        data[roles] = {}
        for new1 in ['fantacalcio_score', 'corriere_score', 'tuttosport_score','gazzetta_score']:
            data[roles][new1] = {}
            for new2 in ['fantacalcio_score', 'corriere_score', 'tuttosport_score','gazzetta_score']:
                if(new1 != new2):
                    x = df[df['player_role_newspaper'] == roles][new1]
                    y = df[df['player_role_newspaper'] == roles][new2]
                    pearson, pvalue = pearsonr(x, y)
                    rmse_val = round(sqrt(mean_squared_error(x, y)), 3)
                    ks = round(ks_2samp(x, y)[0],3)
                    
                    number_of_agreements = 0
                    tott = len(x)
                    for a, b in zip(x, y):
                        if (a >= 6 and b >= 6) or (a < 6 and b < 6):
                            number_of_agreements += 1
                    perc_agr = number_of_agreements/tott
                    perc_dis = 1 - perc_agr
                    
                    data[roles][new1][new2] = {}
                    data[roles][new1][new2]['r'] = round(pearson,3)
                    data[roles][new1][new2]['ks'] = ks
                    data[roles][new1][new2]['RMSE'] = rmse_val
                    data[roles][new1][new2]['disagreement'] = round(perc_dis,3)
                    
    return data
                    
                    