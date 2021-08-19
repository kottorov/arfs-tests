import numpy as np
from sklearn.inspection import permutation_importance
from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
import shap
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from lightgbm import LGBMRegressor, LGBMClassifier
from xgboost import XGBRegressor, XGBClassifier
import arfs
import pandas as pd
import arfs.featselect as arfsfs
import arfs.allrelevant as arfsgroot
from matplotlib import pyplot as plt
from arfs.utils import highlight_tick
import timeit
import math

#-------------------------------------------------------------------------------

def plot_y_vs_X(X, y, ncols=2, figsize=(10, 10)):
    """
    Plot target vs relevant and non-relevant predictors

    :param X: pd.DataFrame
        the pd DF of the predictors
    :param y: np.array
        the target
    :param ncols: int, default=2
        the number of columns in the facet plot
    :param figsize: 2-uple of float, default=(10, 10)
        the figure size
    :return:f, matplotlib objects
        the univariate plots y vs pred_i
    """

    X = pd.DataFrame(X)
    ncols_to_plot = X.shape[1]
    n_rows = int(np.ceil(ncols_to_plot / ncols))

    # Create figure and axes (this time it's 9, arranged 3 by 3)
    f, axs = plt.subplots(nrows=n_rows, ncols=ncols, figsize=figsize)

    # delete non-used axes
    n_charts = ncols_to_plot
    n_subplots = n_rows * ncols
    cols_to_enum = X.columns

    # Make the axes accessible with single indexing
    if n_charts > 1:
        axs = axs.flatten()

    for i, col in enumerate(cols_to_enum):
        # select the axis where the map will go
        if n_charts > 1:
            ax = axs[i]
        else:
            ax = axs

        ax.scatter(X[col], y, alpha=0.1)
        ax.set_title(col)

    if n_subplots > n_charts > 1:
        for i in range(n_charts, n_subplots):
            ax = axs[i]
            ax.set_axis_off()

    # Display the figure
    plt.tight_layout()
    return f

#-------------------------------------------------------------------------------

# repeat the experiment 5 times
def grootRepeatCor(size,w1,w2,w3,w4,w5,target,case,N=5,C=[0]):
    c = 0
    relVar = set(['var0','var1','var2','var3','var4','var5'])
    np.random.seed(42)
    N = N
    p = []
    C = C
    for cor in C:
        c = 0
        for i in range(N):
            # create data set
            X,y = generate_corr_dataset_regr(w1=w1,w2=w2,w3=w3,w4=w4,w5=w5,target=target,
                                             size=size,cor=cor,case=case)
            #groot
            feat_selector = arfsgroot.GrootCV(objective='rmse', cutoff = 1, n_folds=5, n_iter=5)
            feat_selector.fit(X, y, sample_weight=None)
            if relVar.intersection(set(feat_selector.support_names_)) == relVar:
                c += 1
        print("----------- correlation = ", cor,"-------------")
        print("number of times that all relevant features were chosen: ",c, "out of", N)
        print("error rate = ", 1-c/N)
        p.append(1-c/N)
    return(p)
#-------------------------------------------------------------------------------

class groothelper:

    def __init__(self, X, y):

        # data sets
        self.X = X
        self.y = y

        # spread of scores
        self.s0 = None
        self.s1 = None
        self.s2 = None
        self.s3 = None
        self.s4 = None
        self.s5 = None

    def Plot(self):

        #groot fit
        feat_selector = arfsgroot.GrootCV(objective='rmse', cutoff = 1, n_folds=5, n_iter=5)
        feat_selector.fit(self.X, self.y, sample_weight=None)

        print("Relevant:")
        print(feat_selector.support_names_)
        fig = feat_selector.plot_importance(n_feat_per_inch=5)
        print("")
        # print median
        print("Median Values:")
        print(feat_selector.cv_df[['feature','Med']][:6])
        # get standard deviation of scores
        self.s0 = np.std(np.array(feat_selector.cv_df.iloc[:1,2:26]))
        self.s1 = np.std(np.array(feat_selector.cv_df.iloc[1:2,2:26]))
        self.s2 = np.std(np.array(feat_selector.cv_df.iloc[2:3,2:26]))
        self.s3 = np.std(np.array(feat_selector.cv_df.iloc[3:4,2:26]))
        self.s4 = np.std(np.array(feat_selector.cv_df.iloc[4:5,2:26]))
        self.s5 = np.std(np.array(feat_selector.cv_df.iloc[5:6,2:26]))


        # highlight synthetic random variable
        fig = highlight_tick(figure=fig, str_match='random')
        fig = highlight_tick(figure=fig, str_match='genuine', color='green')
        plt.show()
#-------------------------------------------------------------------------------

"""CORRELATED AND STRONGLY RELEVANT: LINEAR + WEAK NON LINEARITY
MODEL + Strong NON linearity  AND (NO) NOISE"""

# This function generates different data sets. Specifically it generates 8
# different cases:
# case1 : 1 pair of correlated and strongly relevant
# case2 : 2 pairs of cor. and s. r.
# case3 : 3 pairs of c. and s.r.
# case4 : 3 c. and s.r.
# case5: 3 and 3 ..
# case6 : 4 c. and s.r.
# case7 : 5 ...
# case8 : 6

def generate_corr_dataset_regr(size, cor, w1, w2, w3, w4, w5, case=1, target='linear', noise=False, sd=0):


    """ This function returns a data set X of predictors in data frame format and a numpy
    array that contains the target y.
    Parameters:
    1. The relationship between the target and the features is set with
    the 'target' parameter: 'linear', 'weak non-linear' and 'strong non-linear'.
    2. The 'cor' parameter defines the correlation strength between variables.
    3. The weights 'w1', 'w2', 'w3', 'w4', 'w5' are the coefficients of the features in
    the input-output relation.
    4. The 'size' is the number of rows of the data sets.
    5. The  'case' parameter distinguishes between the different cases of correlation.
    6. The 'noise' is a Boolean that determines the presense/absense of noise
    7. 'sd' is the standard deviation of noise. """

    # error term
    w = np.random.normal(loc=0,size=size,scale=sd)
    # GENERATION ----------------------------------------------------
    if case <= 3:
        # define mean
        mean = [0, 1]
        # covariance matrix
        m = np.array([[1, cor], [cor, 1]])
        cov = (1/(1+cor**2))*np.dot(m, m.transpose())
        # first pair
        x0, x1 = np.random.multivariate_normal(mean, cov, size=size).T
        #x0 = np.random.normal(0,1,size)
        #x1 = np.random.normal(1,1,size)
        if case == 1:
            # no correlation
            x2 = np.random.gamma(1, 1, size)
            x3 = np.random.normal(1, 1, size)
            x4 = np.random.normal(-1, 1, size)
            x5 = np.random.gamma(1, 1, size)
        else:
            # second pair
            x2, x3 = np.random.multivariate_normal(mean, cov, size=size).T
            if case == 2:
                x4 = np.random.normal(-1, 1, size)
                x5 = np.random.gamma(1, 1, size)
            else:
                # third pair
                x4, x5 = np.random.multivariate_normal(mean, cov, size=size).T

    if case == 4 or case == 5:
        # mean
        mean = [0, 1, 2]
        # covariance
        m = np.array([[1, cor, cor], [cor, 1, cor], [cor, cor, 1]])
        cov = (1/(1+2*cor**2))*np.dot(m, m.transpose())
        # three correlated
        x0, x1, x2 = np.random.multivariate_normal(mean, cov, size=size).T
        if case == 4:
            x3 = np.random.normal(1, 1, size)
            x4 = np.random.normal(-1, 1, size)
            x5 = np.random.gamma(1, 1, size)
        else:
            # three more correlated
            x3, x4, x5 = np.random.multivariate_normal(mean, cov, size=size).T

    if case == 6:
        # mean
        mean = [0,1,2,3]
        # covariance matrix
        m = np.array([[1, cor, cor, cor], [cor, 1, cor, cor],
                    [cor, cor, 1, cor], [cor, cor, cor, 1]])
        cov =(1/(1+3*cor**2))*np.dot(m, m.transpose())
        # four correlated
        x0, x1, x2, x3 = np.random.multivariate_normal(mean, cov, size=size).T
        x4 = np.random.normal(-1, 1, size)
        x5 = np.random.gamma(1, 1, size)

    if case == 7:
        # mean
        mean = [0,1,2,3,4]
        # covariance
        m = np.array([[1,cor,cor,cor,cor],[cor,1,cor,cor,cor],[cor,cor,1,cor,cor], [cor,cor,cor,1,cor],[cor,cor,cor,cor,1]])
        cov = (1/(1+4*cor**2))*np.dot(m, m.transpose())
        # 5 correlated
        x0, x1, x2, x3, x4 = np.random.multivariate_normal(mean, cov, size=size).T
        x5 = np.random.gamma(1,1,size)

    if case == 8:
        # mean
        mean = [0,1,2,3,4,5]
        # covariance
        m = np.array([[1,cor,cor,cor,cor,cor],
                   [cor,1,cor,cor,cor,cor],[cor,cor,1,cor,cor,cor],
                   [cor,cor,cor,1,cor,cor],[cor,cor,cor,cor,1,cor],
                   [cor,cor,cor,cor,cor,1]])
        cov = (1/(1+5*cor**2))*np.dot(m, m.transpose())
        # 6 correlated
        x0, x1, x2, x3, x4, x5 = np.random.multivariate_normal(mean, cov, size=size).T


    # DATA FRAME CONSTRUCTION -----------------------------------------------------------
    # initiate empty matrix
    X = np.zeros((size,11))


    # 6 columns = relevant features
    X[:, 0] = x0
    X[:,1] = x1
    X[:,2] = x2
    X[:,3] = x3
    X[:,4] = x4
    X[:,5] = x5

    # 5 columns =  irrelevant ones
    X[:,6] = np.random.normal(-10,1,size)
    X[:,7] = np.random.gamma(4,1,size)
    X[:,8] = np.random.uniform(-12,1,size)
    X[:,9] = np.random.uniform(5,1,size)
    X[:,10] = np.random.normal(4,1,size)


    # make  it a pandas DF
    column_names = ['var' + str(i) for i in range(11)]
    X = pd.DataFrame(X)
    X.columns = column_names



    # CONSTRUCTION OF TARGET ----------------------------------------------------------------------
    #
    if target == 'linear':
        # target lin model - no noise
        y = w1*x0 + w2*x1 + w3*x2 + w4*x3 + w5*x4 + (1-w1-w2-w3-w4-w5)*x5
    elif target == 'weak non-linear':
        # weak linearity
        y = w1*np.sqrt(abs(x0)) + w2*x1 + w3*x2 + w4*x3 + w5*x4 + (1-w1-w2-w3-w4-w5)*x5
    elif target == 'strong non-linear':
        # strong non linearity
        y = w1*np.sin(x0) + w2*x1 + w3*x2 + w4*x3 + w5*x4 + (1-w1-w2-w3-w4-w5)*x5
    else:
        # just return zero
        y = np.zeros(size)

    # add noise if true
    if noise == True:
        y += w


    return X, y

#-------------------------------------------------------------------------------

# repeat the experiment 10 times
def grootRepeatNoise(size,w1,w2,w3,w4,w5,target,case,cor,N=5,
                            SD=[i*0.4 for i in range(0,24)]):
    c = 0
    relVar = set(['var0','var1','var2','var3','var4','var5'])
    np.random.seed(42)
    N = N
    p = []
    SD = SD
    for sd in SD:
        c = 0
        for i in range(N):
            # create data set
            X,y = generate_corr_dataset_regr(w1=w1,w2=w2,w3=w3,w4=w4,w5=w5,target=target,noise=True,
                                             size=size,cor=cor,case=case,sd=sd)
            #groot
            feat_selector = arfsgroot.GrootCV(objective='rmse', cutoff = 1, n_folds=5, n_iter=5)
            feat_selector.fit(X, y, sample_weight=None)
            if relVar.intersection(set(feat_selector.support_names_)) == relVar:
                c += 1
        print("----------- standard deviation of noise = ", sd,"-------------")
        print("number of times that all relevant features were chosen: ",c, "out of 10")
        print("error rate = ", 1-c/N)
        p.append(1-c/N)
    return(p)


#-------------------------------------------------------------------------------
def generate_more_var(size, n, cor):

    """This function generates a data set with n relevant and
    n irrelevant variables. All the relevant variables have a
    correlation. Parameters:
    size : number of rows
    n : number of columns/2
    cor : correlation strength"""

    # first generate mean
    mean1 = [i for i in range(n)]
    # covariance matrix
    a = np.ones(n) - cor
    m = np.diag(a,0) + np.ones((n,n))*cor
    cov1 = (1/(1+(n-1)*cor**2))*np.dot(m, m.transpose())

    # initiate data matrix
    X1 = np.zeros((size,n))
    # fill its contents:
    y = 0
    #relevant and correlated
    for i in range(n):
        # set seed so that the columns are the same
        np.random.seed(2)
        # fill matrix
        X1[:,i] =  np.random.multivariate_normal(mean = mean1,
            cov = cov1, size=size).T[i]

        y += (1/n)*X1[:,i]

    # make  it a pandas DF
    column_names = ['var' + str(i) for i in range(n)]
    X = pd.DataFrame(X1)
    X.columns = column_names
    #
    # irrelevant variables
    X2 = np.zeros((size,n))
    mean2 = np.ones(n)
    cov2 = np.diag(a,0)
    for i in range(n):
        # set seed so that the columns are the same
        np.random.seed(42)
        # fill matrix
        X2[:,i] =  np.random.multivariate_normal(mean = mean2,
        cov = cov2, size=size).T[i]

    # make  it a pandas DF
    column_names = ['var' + str(i) for i in range(n,2*n)]
    X2 = pd.DataFrame(X2)
    X2.columns = column_names

    X = pd.concat([X, X2], axis = 1)

    return X, y

# ------------------------------------------------------------------------------

def generate_outliers(size, out=True):

    # relevant variables
    mean = [0, 1]
    m = np.array([[1, 0], [0, 1]])
    #cov = np.array([[1,cor],[cor,1]])
    x0, x1 = np.random.multivariate_normal(mean, cov, size=size).T
    x2 = np.random.gamma(1, 1, size)
    x3 = np.random.normal(1,1,size)
    x4 = np.random.normal(-1,1,size)
    x5 = np.random.gamma(1,1,size)
    X = np.zeros((size,11))


    # 6 relevant features
    X[:, 0] = x0
    X[:,1] = x1
    X[:,2] = x2
    X[:,3] = x3
    X[:,4] = x4
    X[:,5] = x5

    # 5 irrelevant ones
    X[:,6] = np.random.normal(-10,1,size)
    X[:,7] = np.random.gamma(4,1,size)
    X[:,8] = np.random.uniform(-12,1,size)
    X[:,9] = np.random.uniform(5,1,size)
    X[:,10] = np.random.normal(4,1,size)


    # make  it a pandas DF
    column_names = ['var' + str(i) for i in range(11)]
    X = pd.DataFrame(X)
    X.columns = column_names

    if out==True:
        # outliers
        y = 0.4*np.sqrt(abs(x0)) + 0.2*np.sin(x1) + 0.1*x2 + 0.1*np.cos(x3) + 0.05*x4 + 0.05*np.tan(x5)
    else:
        y = 0.4*np.sqrt(abs(x0)) + 0.2*np.sin(x1) + 0.1*x2 + 0.1*np.cos(x3) + 0.05*x4 + 0.05*x5


    return X, y
