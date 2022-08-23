#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from sklearn.model_selection import cross_validate, cross_val_score, KFold,GridSearchCV,RandomizedSearchCV
import numpy as np
import pandas as pd
import pylab
import matplotlib.pyplot as plt
import seaborn as sns
#%matplotlib inline
from pandas_profiling import ProfileReport
from tqdm import tqdm
import pylab
import matplotlib.pyplot as plt
import seaborn as sns
#%matplotlib inline
from pandas_profiling import ProfileReport
from sklearn.cluster import KMeans
from kmodes.kprototypes import KPrototypes


# In[ ]:


class Judge:
    
    def __init__(self, innercycle = 3, outercycle = 3):
        self.metrics = ()
        self.params = {}
        self.hyper_param = {}
        self.innercycle = innercycle
        self.outercycle = outercycle
        return None
    
    def __str__(self):
        return f" Judging {self.nome_dataframe}"
    
    def set_research(self, refit = "ROC_AUC", grid = "Grid"):
        self.refit = refit
        self.grid  = grid
    
    def self_innercylce(self,ncycle):
        self.innercycle = ncycle
        return self
    
    def self_outercycle(self,ncycle):
        self.outercycle = ncycle
        return self
    
    def set_data(self, X, y):
        self.X = X
        self.y = y
        return self
    
    def set_algorithms(self, pipelines):
        self.algorithms = pipelines
        return self
    
    def set_params(self, params):
        self.params = params
        return self
    
    def set_metrics(self, metrics):
        self.metrics = metrics
        return self

    def set_diz_hyper_param(self,dizionario):
        self.diz_hyper_param = dizionario
    
    def set_hyper_param(self,algorithm):
        self.hyper_param = self.diz_hyper_param[algorithm]

    def get_performance_from_algorithm(self, name, algorithm):
        cv_inner = KFold(n_splits=self.innercycle, shuffle=True, random_state=1)
        if bool(self.hyper_param):
            if self.grid.lower() == "grid":
                search = GridSearchCV(algorithm, self.hyper_param, scoring=self.metrics, n_jobs=-1, cv=cv_inner, refit=self.refit)
            else:
                search = RandomizedSearchCV(algorithm, self.hyper_param, scoring=self.metrics, n_jobs=-1, cv=cv_inner, refit=self.refit)
        else:
            search = algorithm
        cv_outer = KFold(n_splits=self.outercycle, shuffle=True, random_state=1)
        cv_results = cross_validate(search, self.X, self.y, scoring = self.metrics, cv=cv_outer,return_train_score=True,n_jobs=-1)
        risultati = []
        for i in self.metrics.keys():
            score = round(np.mean(cv_results['test_'+i]) * 100, 2)
            risultati.append(score)
        return risultati
    
    def get_comparison_table(self):
        diz = {}
        for name, algorithm in self.algorithms.items():
            self.set_hyper_param(name)
            diz[name] = self.get_performance_from_algorithm(name, algorithm)
        table = pd.DataFrame.from_dict(diz).T
        table.columns = self.metrics.keys()
        return table


# In[1]:


class EDA(object):
    def __init__(self,df = None):
        self.df = df
        return None
    
    def plot_correlation_matrix(self,columns = None,figsize=(10,30)):
        df_num   = self.df
        if not columns:
            columns = df_num.columns
        df_num   = df_num[columns]
        fig,ax   = plt.subplots(figsize=figsize)
        correlated_matrix = df_num.corr().abs()
        sns.heatmap(correlated_matrix, annot = True, cmap= 'Greens')
        ax.set_title('Correlation Matrix')
        return plt.show()
        
    def plot_dist_boxplot(self,columns=None, dist=False, kde=False, figsize=(10,30),title="I'm plotting the Features BoxPlot"):
        df_local    = self.df
        if not columns:
            columns = df_local.columns        
        df_local    = df_local[columns]

        lista = columns
       
        if not dist:
            fig = plt.figure(figsize=figsize)
            fig.suptitle(title, fontsize=16, x=0.5, y=1.01) 
            rows  = (len(lista) // 4 ) + 1
            for i, nf in enumerate(lista):
                ax = fig.add_subplot(rows, 4, i+1)
                sns.boxplot(df_local[nf], ax=ax)
                plt.title(f'{nf}')
                plt.xlabel(None)
        else:
            fig = plt.figure(figsize=figsize)
            fig.suptitle(title, fontsize=16, x=0.5, y=1.01)
            rows  = len(lista)
            j     = 0
            for i, nf in enumerate(lista):
                ax = fig.add_subplot(rows, 2, j+1)
                if kde:
                    sns.histplot(df_local[nf], kde=True, ax=ax)
                else:
                    sns.histplot(df_local[nf], kde=False, ax=ax)
                plt.title(f'{nf}- Skew: {round(df_local[nf].skew(), 2)}', size=10)
                plt.xlabel(None) 
                j += 1
                ax = fig.add_subplot(rows, 2, j+1)
                sns.boxplot(df_local[nf], ax=ax, color='#99befd', fliersize=5)
                plt.title(f'{nf}')
                plt.xlabel(None)
                j += 1
            
            #plt.xlabel(f'{nf}- Skew: {round(df_local[nf].skew(), 2)}', size=20)
        fig.tight_layout()
        return plt.show()
    
    # # Checking the distribution of continuous features
    def plot_distribution(self, columns, title="I'm plotting the Features Distributions", kde=True, figsize=(10,30)):
        
        df_local = self.df
        fig = plt.figure(figsize=figsize)
        fig.suptitle(title, fontsize=16, x=0.5, y=1.01)

        lista = columns
        rows  = (len(lista) // 4 ) + 1        
        for i, nf in enumerate(lista):
            ax = fig.add_subplot(rows, 4, i+1)
            if kde:
                sns.histplot(df_local[nf], kde=True, ax=ax)
            else:
                sns.histplot(df_local[nf], ax=ax)
            plt.title(f'{nf}- Skew: {round(df_local[nf].skew(), 2)}', size=10)
            plt.xlabel(None)
            #plt.xlabel(f'{nf}- Skew: {round(df_local[nf].skew(), 2)}', size=20)
        fig.tight_layout()
           
        return plt.show()
        
#        df_local = self.df
#        i = 1
#        fig, ax = plt.subplots(7,6, figsize=(40,30))
        
#        for feature in tqdm(self.numerical_columns):
#            plt.subplot(7,6, i)
#            sns.kdeplot(data = df_local, y = feature, vertical=True, palette = 'coolwarm_r', shade = True)
#            plt.xlabel(f'{feature}- Skew: {round(df_local[feature].skew(), 2)}', size=20)
#            i += 1

#        fig.tight_layout()
#        return plt.show()
    
    def plot_missing_values(self):
        df_local = self.df
        cols_with_missing = [col for col in df_local.columns if df_local[col].isnull().any()]
        if len(cols_with_missing) == 0:
            return print("No Missing Values are present in the Dataset")
        """ For each column with missing values plot proportion that is missing."""
        data = [(col, (df_local[col].isnull().sum() / len(df_local))*100) for col in df_local.columns if df_local[col].isnull().sum() > 0] 
        col_names = ['column', 'percent_missing']
        missing_df = pd.DataFrame(data, columns=col_names).sort_values('percent_missing')
        pylab.rcParams['figure.figsize'] = (15, 8)
        missing_df.plot(kind='barh', x='column', y='percent_missing')
        plt.legend(loc='lower right')
        plt.title('Percent of missing values in columns')
        return plt.show()
    
    def plot_pie(self, column):
        df_local    = self.df
        tipi        = df_local[column].value_counts()
        colors      = sns.color_palette('bright')
        labels      = tipi.index.to_list()
        plt.pie(tipi,colors=colors,labels = labels, autopct = '%0.0f%%')
        return plt.show()
    
    def profile(self, only_numeric = True, message = "Numerical Datatype Profiling Report"):
        self.profile_message = message
        df_local             = self.df
        if only_numeric:
            profile_num          = ProfileReport(df_local[self.numerical_columns], self.profile_message)
        else:
            profile_num          = ProfileReport(df_local, self.profile_message)
        return profile_num.to_widgets()
    
    def get_numerical_column(self):
        df_local = self.df
        numerical_columns = df_local.select_dtypes(np.number).columns
        self.numerical_columns = numerical_columns
        return self

    def get_numerical_categorical(self, feature_list = None, n_unique=False):
        df_local = self.df
        if not feature_list:
            feature_list = df_local.columns
            
        int_features         = []
        float_features       = []
        categorical_features = []
        for column in feature_list:
            if n_unique:
                if df_local[column].nunique() > n_unique:
                    numerical_features.append(column)
                else:
                    categorical_features.append(column)
            else:
                if df_local[column].dtypes == 'object': 
                    categorical_features.append(column)
                elif ( (df_local[column].dtypes == 'int64') or (df_local[column].dtypes == 'int16') ):
                    int_features.append(column)
                else:
                    float_features.append(column)
        self.int_columns         = int_features
        self.float_columns       = float_features
        self.numerical_columns   = float_features + int_features
        self.categorical_columns = categorical_features
        return self


# In[ ]:


def Func_Kmeans(X, y, kcluster = [10,20,30,40,50,60,70,80], thresh = 0.3, plot = False):
    
    if len(kcluster) < 3:
        raise Exception("len of kcluster list < 3")
    
    y_name = y.name
    df     = pd.concat([X,y], axis=1)
    
    sse    = {}
    for k in kcluster:
        kmeans = KMeans(init="k-means++", n_clusters=k).fit(df)
    #print(data["clusters"])
        sse[k] = kmeans.inertia_
    rms_values = list(sse.values())
    cluster    = list(sse.keys())    
    for i in range(1,len(rms_values)-1):
        m1 = np.abs(rms_values[i-1]-rms_values[i])
        m2 = np.abs(rms_values[i]-rms_values[i+1]) 
        if (np.abs((m1-m2)/m1) < thresh):
            goodK = cluster[i-1]
            break 
        else:
            goodK = None
            
    if plot:
        plt.figure()
        plt.plot(list(sse.keys()), list(sse.values()),marker='o')
        plt.xlabel("Numero di clusters", fontsize=16)
        plt.ylabel("Somma delle distanza al quadrato", fontsize=16)
        plt.savefig("number_of_k.png")
        plt.show()
        
    if not goodK:
        raise Exception("Good K not found")
        
    kmeans  = KMeans(init="k-means++", n_clusters=goodK).fit(df)
    centers = kmeans.cluster_centers_
    columns = df.columns
    new_df  = pd.DataFrame(centers, columns = columns) 
    y_new   = new_df[y.name]
    X_new   = new_df.drop(y.name,axis=1)
    return X_new,y_new


# In[ ]:


def get_PR_AUC(y_true,y_proba):
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    return auc(recall,precision)

