#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from feature_engine.creation import CyclicalFeatures
from sklearn.decomposition import PCA
from geopy.geocoders import Nominatim
from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
import numpy as np
import pandas as pd
import re
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import Normalizer
from sklearn import metrics
from scipy.stats import boxcox, yeojohnson
from sklearn.cluster import KMeans
from kmodes.kprototypes import KPrototypes


# In[ ]:


class CalcLatLong(BaseEstimator, TransformerMixin):
    def __init__(self,column,country,missing_value):
        self.column        = column
        self.country       = country
        self.missing_value = missing_value
    
    def recodifica_city_latitudine(self,city):
        try:
            return self.diz_lat_long[city][0]
        except:
            return self.missing_value

    def recodifica_city_longitudine(self,city):
        try:
            return self.diz_lat_long[city][1]   
        except:
            return self.missing_value

    def get_diz_lat_long(self,data):
        diz_lat_long = {}
        for i in np.unique(data[self.column]):
            geolocator = Nominatim(user_agent="http")
            city       = i
            country    = self.country
            try:
                loc        = geolocator.geocode(city+','+ country) 
#            print("city: ",city,"; latitude is :-" ,loc.latitude,"; longtitude is:-" ,loc.longitude)
                diz_lat_long[i] = [loc.latitude, loc.longitude] 
            except:
                r = re.findall('([A-Z])', city)
                splitting = city.split(r[1])
                city = splitting[0] + " " +r[1]+splitting[1]
                loc        = geolocator.geocode(city+','+ country)
#            print("city: ",city,"; latitude is :-" ,loc.latitude,"; longtitude is:-" ,loc.longitude)
                diz_lat_long[i] = [loc.latitude, loc.longitude] 
        return diz_lat_long        
        
    def fit(self,X,y=None):
        self.diz_lat_long = self.get_diz_lat_long(X)
        return self
            
    def transform(self,X):
        df = X.copy()
        df['Latitudine'] = df[self.column].apply(self.recodifica_city_latitudine)
        df['Longitudine'] = df[self.column].apply(self.recodifica_city_longitudine)
        return df.drop(self.column, axis=1)


# In[ ]:


class CastType(BaseEstimator, TransformerMixin):
    
    def __init__(self, column, datatype = int, substring = "_cat",verbose = False):
        self.column      = column
        self.datatype    = datatype 
        self.sub         = substring
        self.verbose     = verbose

    def my_isnan(self,value):
        try:
            return np.isnan(value)
        except:
            return False     
        
    def fit(self,X,y=None):
        if self.verbose:
            print("### CastType (fit) of ",self.column,", Shape of X : ", X.shape )
        return self
    
    def transform(self,X):
        df              = X.copy()
        df[self.column+self.sub] = df[self.column].apply(lambda x: self.datatype(x) if not self.my_isnan(x) else x)
        if self.verbose:
            print("### CastType (transform) of ",self.column,", Shape of X : ", df.shape )
        return df


# In[ ]:


class ChangeType(BaseEstimator, TransformerMixin):
    
    def __init__(self, columns, tipi,verbose = False):
        self.columns  = columns
        self.tipi     = tipi
        self.verbose  = verbose
        
    def fit(self,X,y=None):
        if self.verbose:
            print("### ChangeType (fit), Shape of X : ", X.shape )
        return self
    
    def transform(self,X):
        df         = X.copy()
        diz_map    = dict(zip(self.columns, self.tipi))
        df         = df.astype(diz_map)
        if self.verbose:
            print("### ChangeType (transform) of ",self.columns,", Shape of X : ", df.shape )
        return df


# In[ ]:


class DropCol(BaseEstimator, TransformerMixin):
    
    def __init__(self, columns, verbose = False):
        self.columns  = columns
        self.verbose  = verbose 
        
    def fit(self,X,y=None):
        if self.verbose:
            print("### DropCol (fit) of ",self.columns,", Shape of X : ", X.shape )
        return self
    
    def transform(self,X):
        df         = X.copy()
        df         = df.drop(self.columns,axis=1) 
        if self.verbose:
            print("### DropCol (transform) of ",self.columns,", Shape of X : ", df.shape )
        return df


# In[ ]:


class DropColNull(BaseEstimator, TransformerMixin):
    
    def __init__(self, thresh=40, verbose = False):
        self.thresh   = thresh
        self.verbose  = verbose
        
    def fit(self,X,y=None):
        self.n_min = int(len(X) * (self.thresh/100) )
        if self.verbose:
            print("### DropColNull (fit), Shape of X : ", X.shape )
        return self
    
    def transform(self,X):
        df         = X.copy()
        columns    = []
        for i in df.columns:
            if (df[i].isnull().sum() > self.n_min):
                df = df.drop(i,axis=1)
                columns.append(i)
        if self.verbose:
            print("### DropColNull (transform) of ",columns,", Shape of X : ", df.shape )
        return df


# In[ ]:


class FSCorrMatryxByValue(BaseEstimator, TransformerMixin):
    
    def __init__(self,soglia=20,verbose=False):
        self.soglia  = soglia
        self.verbose = verbose
    
    def fit(self,X,y=None):
        df_num                 = pd.concat([y, X], axis=1)
        correlation_matrix_abs = df_num.corr().abs()
        correlation_y          = correlation_matrix_abs[y.name]        
        correlation_y_sorted   = correlation_y.sort_values(ascending = False)
        correlation_y_sorted   = correlation_y_sorted[1:len(correlation_y_sorted)] # 1 to skip y itself
        cut_off                = self.soglia / 100
        fs                     = correlation_y_sorted[correlation_y_sorted > cut_off] 
        self.column = fs.index
        if self.verbose:
            print("### FSCorrMatryxByValue (fit), Shape of X : ", X.shape)
#            print(correlation_y)
        return self
    
    def transform(self,X):
        df = X.copy()
        df = df[self.column]
        if self.verbose:
            print("### FSCorrMatryxByValue (transform), Shape of X : ", df.shape)
        return df


# In[ ]:


class FSCorrMatryxByValue_NoY(BaseEstimator, TransformerMixin):
    
    def __init__(self,soglia = 50, verbose = False):
        self.soglia      = soglia
        self.verbose     = verbose
        
    def fit(self,X,y=None):
        df = X.copy()
        cor_matrix   = df.corr().abs()
        upper_tri    = cor_matrix.where(np.triu(np.ones(cor_matrix.shape),k=1).astype(np.bool))
        cut_off      = self.soglia / 100
        to_drop      = [column for column in upper_tri.columns if any(upper_tri[column] > cut_off)]
        self.to_drop = to_drop 
        if self.verbose:
            print("### FSCorrMatryxByValue_NoY (fit), Shape of X : ", df.shape)
        return self
    
    def transform(self,X):
        df = X.copy()
        df = df.drop(self.to_drop, axis=1)
        if self.verbose:
            print("### FSCorrMatryxByValue_NoY (transform), Shape of X : ", df.shape)
        return df


# In[ ]:


class FS_PCA(BaseEstimator, TransformerMixin):
    def __init__(self,soglia = 80 , verbose=False):
        self.soglia   = soglia
        self.verbose  = verbose
    
    def fit(self,X,y=None):
        self.pca    = PCA(n_components=len(X.columns), random_state=2020)
        self.pca.fit(X)
        cut_off     = self.soglia
        self.indici = np.where(np.cumsum(self.pca.explained_variance_ratio_ * 100)>cut_off)[0]
        if self.verbose:
            print(print("The Explained variance is :", np.cumsum(self.pca.explained_variance_ratio_ * 100)))
            print("The indexes of the satisfied condition are: ",self.indici)
            print("### FS_PCA (fit), Shape of X : ", X.shape)
        return self
    
    def transform(self,X):
        df = X.copy()
        df = self.pca.transform(df)
        if (len(self.indici) > 1):
            df    = df[:,0:self.indici[1]]
            n_col = self.indici[1]
        else:
            df    = df[:,0:self.indici[0]+1]
            n_col = 1
        
        columns = ["pca_comp_"+str(i+1) for i in range(n_col)]
        df      = pd.DataFrame(df, columns=columns)
        if self.verbose:
            print("### FS_PCA (transform), Shape of X : ", df.shape )
        return df


# In[ ]:


def Func_Kmeans_pipe(X, y, list_columns, target_col, kcluster = [10,20,30,40,50,60,70,80], thresh = 0.3, plot = False):
    
    if len(kcluster) < 3:
        raise Exception("len of kcluster list < 3")
    
    y      = np.reshape(y, (len(y),1))
    df     = np.concatenate((X, y), axis=1)
    
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
    new_df  = pd.DataFrame(centers, columns = list_columns) 
    y_new   = new_df[target_col]
    X_new   = new_df.drop(target_col,axis=1)
    return X_new,y_new


# In[ ]:


class ImputatorMeanValue(BaseEstimator, TransformerMixin):
    
    def __init__(self, column,verbose=False):
        self.column  = column
        self.verbose = verbose
    
    def fit(self, X, y=None):
        self.mean_value = X[self.column].mean()
        if self.verbose:
            print("### ImputatorMeanValue (fit) of ",self.column,", Shape of X : ", X.shape)
        return self
    
    def transform(self, X):
        df = X.copy()
        df[self.column] = df[self.column].fillna(self.mean_value)
        if self.verbose:
            print("### ImputatorMeanValue (transform) of ",self.column,", Shape of X : ", df.shape )
        return df


# In[ ]:


class ImputatorMostFreqValue(BaseEstimator, TransformerMixin):
    
    def __init__(self, columns,verbose=False):
        self.columns = columns
        self.verbose = verbose
    
    def fit(self, X, y=None):
        self.most_freq_value = []
        for i in self.columns:
            lista = list(X[i])
            self.most_freq_value.append(max(set(lista), key = lista.count))
        if self.verbose:
            print("### ImputerMeanValue (fit) of ",self.columns,", Shape of X : ", X.shape )
        return self
    
    def transform(self, X):
        df = X.copy()
        for i, col in enumerate(self.columns):
            df[col] = df[col].fillna(self.most_freq_value[i])
        if self.verbose:
            print("### ImputerMeanValue (transform) of ",self.columns,", Shape of X : ", df.shape)
        return df


# In[ ]:


class ImputerMeanValue(BaseEstimator, TransformerMixin):
    
    def __init__(self, column, verbose=False):
        self.column = column
        self.verbose = verbose
    
    def fit(self, X, y=None):
        self.mean_value = X[self.column].mean()
        if self.verbose:
            print("### ImputerMeanValue (fit) of ",self.column,", Shape of X : ", X.shape )
        return self
    
    def transform(self, X):
        df = X.copy()
        df.loc[np.isnan(df[self.column]), self.column] = self.mean_value
        #X[self.column] = X[self.column].fillna(self.mean_value)
        if self.verbose:
            print("### ImputerMeanValue (transform) of ",self.column,", Shape of X : ", df.shape)
        return df


# In[ ]:


class ImputerNewCategory(BaseEstimator, TransformerMixin):
    
    def __init__(self, column, category, verbose=False):
        self.column   = column
        self.category = category
        self.verbose  = verbose
    
    def fit(self, X, y=None):
        if self.verbose:
            print("### ImputerNewCategory (fit) of ",self.column,", Shape of X : ", X.shape )
        return self
    
    def transform(self, X):
        df = X.copy()
        df.loc[df[self.column].isna(), self.column] = self.category
        # X[self.column] = X[self.column].fillna(self.category)
        if self.verbose:
            print("### ImputerNewCategory (transform) of ",self.column,", Shape of X : ", df.shape )
        return df


# In[ ]:


class ImputerNullWithModel(BaseEstimator, TransformerMixin):
    def __init__(self, estimator, target_name, feature_names, verbose = False):
        self.estimator     = estimator
        self.target_name   = target_name
        self.feature_names = feature_names
        self.verbose       = verbose
        
    def fit(self,X,y=None):
        dc = X.copy()
        bool_target_not_null = dc[self.target_name].notnull()
        row_indexes_where_target_notnull = dc.index[np.where(bool_target_not_null)]
        X_train = dc.loc[row_indexes_where_target_notnull, self.feature_names]
        y_train = dc.loc[row_indexes_where_target_notnull, self.target_name]
        self.estimator.fit(X_train, y_train)
        if self.verbose:
            print("### ImputerNullWithModel (fit) of ",self.feature_names,", Shape of X : ", X.shape)
            
        return self
    
    def transform(self,X):
        dc = X.copy()
        bool_target_is_null             = dc[self.target_name].isnull()
        row_indexes_where_target_isnull = dc.index[np.where(bool_target_is_null)]
        if (len(row_indexes_where_target_isnull) == 0):
            return X
        X_test                          = dc.loc[row_indexes_where_target_isnull, self.feature_names]
        preds                           = self.estimator.predict(X_test)
        X.loc[row_indexes_where_target_isnull, self.target_name] = preds
        if self.verbose:
            print("### ImputerNullWithModel (transform) of ",self.feature_names,", Shape of X : ", X.shape )
            
        return X


# In[ ]:


class LabelOneHotEncoder(BaseEstimator, TransformerMixin):
    
    def __init__(self, categorical_features,onehot = True, verbose=False):
        self.categorical_features = categorical_features
        self.onehot               = onehot
        self.verbose              = verbose
        
    def fit(self, X, y=None):

        if not isinstance(X, pd.DataFrame):
            raise ValueError("Pass a pandas.DataFrame")
            
        if not isinstance(self.categorical_features, list):
            raise ValueError(
                "Pass categorical_features as a list of column names")
                    
        self.encoding = {}
        self.lenght   = {}
        for c in self.categorical_features:

            _, int_id = X[c].factorize()
            self.encoding[c] = dict(zip(list(int_id), range(0,len(int_id))))
            self.lenght[c]   = len(int_id)
            
        if self.verbose:
            print("### LabelOneHotEncoder (fit) of ",self.categorical_features, ", Shape of X : ", X.shape)            
        return self

    def transform(self, X):

        if not isinstance(X, pd.DataFrame):
            raise ValueError("Pass a pandas.DataFrame")

        if not hasattr(self, 'encoding'):
            raise AttributeError("FeatureTransformer must be fitted")
            
        df = X.drop(self.categorical_features, axis=1)
        
        if self.onehot:  # one-hot encoding
            for c in sorted(self.categorical_features):            
                categories = X[c].map(self.encoding[c]).values
                for val in self.encoding[c].values():
                    df["{}_{}".format(c,val)] = (categories == val).astype('int64')
        else:       # label encoding
            for c in sorted(self.categorical_features):
                df[c] = X[c].map(self.encoding[c]).fillna(self.lenght[c]).astype('int64')
                
        if self.verbose:
            print("### LabelOneHotEncoder (tranform) of ",self.categorical_features, ", Shape of X : ", df.shape)
                
        return df


# In[ ]:


class Logger(BaseEstimator, TransformerMixin):
    
    def __init__(self, message):
        self.message = message
    
    def fit(self, X, y):
        print("### Logger (fit): " + self.message)
        return self
    
    def transform(self, X):
        print("### Logger (transform): " + self.message + " X shape: ",X.shape)
        return X


# In[ ]:


class LSA(BaseEstimator, TransformerMixin):
    
    def __init__(self, column, components = 2, verbose = False):
        self.column      = column
        self.components  = components 
        self.verbose     = verbose
        
    def fit(self,X,y=None):
        if self.verbose:
            print("### LSA (fit) of ",self.column,", Shape of X : ", X.shape )
        return self
    
    def transform(self,X):
        df         = X.copy()
        old_index  = df.index
        df         = df.reset_index(drop=True)
        vectorizer = CountVectorizer(min_df = 1, stop_words = None)
        dtm        = vectorizer.fit_transform(df[self.column]) 
        lsa        = TruncatedSVD(self.components)
        dtm_lsa    = lsa.fit_transform(dtm)
        dtm_lsa    = Normalizer(copy=False).fit_transform(dtm_lsa)
        columns    = [self.column+"_"+str(x+1) for x in range(self.components)]
        new_df     = pd.DataFrame(dtm_lsa, columns = columns)
        df         = pd.concat([df, new_df], axis=1)
        df         = df.drop(self.column,axis=1) 
        df         = df.set_index(old_index)
        if self.verbose:
            print("### LSA (transform) of ",self.column,", Shape of X : ", df.shape )
        return df


# In[ ]:


class OneHotEncoderColumn(BaseEstimator, TransformerMixin):
    
    def __init__(self, column, verbose=False):
        self.column = column
        self.verbose = verbose
    
    def fit(self, X, y):
        self.ohe = OneHotEncoder(categories = "auto", handle_unknown = "ignore")
        self.ohe.fit(X[[self.column]].values.reshape(-1, 1))
        if self.verbose:
            print("### OneHotEncoderColumn (fit) of ",self.column, ", Shape of X : ", X.shape)
        return self
    
    def transform(self, X):
        # from np array to dataframe
        ohe_data_pd = pd.DataFrame(
            self.ohe.transform(X[[self.column]].values.reshape(-1, 1)).toarray(),
            columns = [self.column + '_' + x for x in self.ohe.categories_[0]]
        )
        res = pd.concat([X, ohe_data_pd], axis=1).drop(self.column, axis=1)
        if self.verbose:
            print("### OneHotEncoderColumn (tranform) of ",self.column, ", Shape of X : ", res.shape)
            
        return res


# In[ ]:


class OutlierImputation_IQR(BaseEstimator, TransformerMixin):
    
    def __init__(self, columns, factor = 1.5, method = "mean"):
        self.method  = method
        self.columns = columns 
        self.factor  = factor 
        
    def fit(self,X,y=None):
        df = X.copy()
        self.lower           = []
        self.upper           = []
        self.min             = []
        self.max             = []
        self.most_freq_value = []
        for col in self.columns:
            q25, q75      = np.percentile(df[col], 25), np.percentile(df[col], 75)
            iqr           = q75 - q25
            # calculate the outlier cutoff and most frequent value
            cut_off       = iqr * self.factor
            lower, upper  = q25 - cut_off, q75 + cut_off
            most_frequent = max(set(list(df[col])), key = list(df[col]).count)
            self.lower.append(lower)
            self.upper.append(upper)
            self.max.append(q75)
            self.min.append(q25)
            self.most_freq_value.append( most_frequent )
        self.median, self.mean = df[self.columns].median(), df[self.columns].mean()
        
        return self
    
    def transform(self,X):
        df = X.copy()
        for i, col in enumerate(self.columns):
            
            if self.method == "mean":
                df.loc[df[col] > self.upper[i], col] = self.mean[i]
                df.loc[df[col] < self.lower[i], col] = self.mean[i]
            elif self.method == "median":
                df.loc[df[col] > self.upper[i], col] = self.median[i]
                df.loc[df[col] < self.lower[i], col] = self.median[i] 
            elif self.method == "most_freq":
                df.loc[df[col] > self.upper[i], col] = self.most_freq_value[i]
                df.loc[df[col] < self.lower[i], col] = self.most_freq_value[i] 
            elif self.method == "up_low":
                df.loc[df[col] > self.upper[i], col] = self.upper[i]
                df.loc[df[col] < self.lower[i], col] = self.lower[i]  
            else:
                df.loc[df[col] > self.upper[i], col] = self.max[i]
                df.loc[df[col] < self.lower[i], col] = self.min[i] 
        return df


# In[ ]:


class OutlierImputation_STD(BaseEstimator, TransformerMixin):
    
    def __init__(self, columns, factor = 3, method = "mean"):
        self.method  = method
        self.columns = columns 
        self.factor  = factor 
        
    def fit(self,X,y=None):
        df = X.copy()
        self.median, self.mean, self.std = df[self.columns].median(), df[self.columns].mean(), df[self.columns].std()
        self.cutoff = self.std * self.factor
        self.lower, self.upper = self.mean - self.cutoff, self.mean + self.cutoff
        return self
    
    def transform(self,X):
        df = X.copy()
        for i, col in enumerate(self.columns):
            
            if self.method == "mean":
                df.loc[df[col] > self.upper[i], col] = self.mean[i]
                df.loc[df[col] < self.lower[i], col] = self.mean[i]
            else:
                df.loc[df[col] > self.upper[i], col] = self.median[i]
                df.loc[df[col] < self.lower[i], col] = self.median[i]                
        return df


# In[ ]:


class ReplaceValues(BaseEstimator, TransformerMixin):
    
    def __init__(self, columns, start_values=["No","Yes"], end_values=[0,1] , substring = "_sub", method = "map",verbose = False):
        self.columns        = columns
        self.start_values   = start_values
        self.end_values     = end_values
        self.sub            = substring
        self.method         = method
        self.verbose        = verbose
        
    def fit(self,X,y=None):
        if self.verbose:
            print("### ReplaceValues (fit) of ",self.columns,", Shape of X : ", X.shape )
        return self
    
    def transform(self,X):
        df                       = X.copy()
        dizionario = dict(zip(self.start_values, self.end_values))
        for i in self.columns:
            if self.method == "map":
                df[i+self.sub] = df[i].map(dizionario, na_action = 'ignore')
            else:
                df[i+self.sub] = df[i].replace(self.start_values, self.end_values)
        if self.verbose:
            print("### ReplaceValues (transform) of ",self.columns,", Shape of X : ", df.shape )
        return df


# In[ ]:


class SelectSubSTRING(BaseEstimator, TransformerMixin):
    
    def __init__(self, column, sep = "-", index_list = [0,1,2], verbose = False):
        self.column      = column
        self.sep         = sep 
        self.verbose     = verbose
        self.index_list  = index_list

    def my_isnan(self,value):
        try:
            return np.isnan(float(value))
        except:
            return False     
        
    def fit(self,X,y=None):
        if self.verbose:
            print("### SelectSubSTRING (fit) of ",self.column,", Shape of X : ", X.shape )
        return self
    
    def transform(self,X):
        df              = X.copy()
        for i in self.index_list:
            df[self.column+"_"+str(i)] = df[self.column].apply(lambda x: x.split("/")[i] if not self.my_isnan(x) else x)
        if self.verbose:
            print("### SelectSubSTRING (transform) of ",self.column,", Shape of X : ", df.shape )
        return df


# In[ ]:


class SkewColumns(BaseEstimator, TransformerMixin):
    
    def __init__(self, columns, tranformation = 'log', thresh = 2.0):
        self.columns       = columns
        self.thresh        = thresh
        self.tranformation = tranformation
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        df = X.copy()
        for col in self.columns:
            if np.abs(df[col].skew()) > self.thresh:
                print(col)
                if (self.tranformation == "yeoj"):
                    df[col]        = np.exp(df[col])
                    df[col], lmbda = yeojohnson(df[col])
                elif(self.tranformation == "boxcox"):
                    df[col]        = np.exp(df[col])
                    df[col], lmbda = boxcox(df[col])                    
                else:
                    df[col] = np.log(df[col])                
        return df


# In[ ]:


class SplitSTRING(BaseEstimator, TransformerMixin):
    
    def __init__(self, column, indici = [3,6], starting_index = 0,verbose = False):
        self.column      = column
        self.indici      = indici 
        self.verbose     = verbose
        self.start_index = starting_index
    
    def espandi_stringa(self, stringa):
        lista_end = []
        j         = self.start_index
        if len(self.indici) > 0:
            for i in self.indici:
                lista_end.append(stringa[j:i])
                j = i
            lista_end.append(stringa[j:])
            return " ".join(lista_end)
        else:
            return stringa[self.start_index:]

        
    def fit(self,X,y=None):
        if self.verbose:
            print("### SplitSTRING (fit) of ",self.column,", Shape of X : ", X.shape )
        return self
    
    def transform(self,X):
        df              = X.copy()
        df[self.column] = df[self.column].apply(self.espandi_stringa)
        if self.verbose:
            print("### SplitSTRING (transform) of ",self.column,", Shape of X : ", df.shape )
        return df


# In[ ]:


class Standardize(BaseEstimator, TransformerMixin):
        
    def fit(self,X,y=None):
        self.mean = X.mean()
        self.std  = X.std() 
        return self
    
    def transform(self,X):
        df = X.copy()
        df = (df - self.mean)/self.std
        return df


# In[ ]:


class TransformDate(BaseEstimator, TransformerMixin):
    
    def __init__(self,column, date_format='%Y-%m-%d', verbose=False):
        self.column      = column
        self.date_format = date_format
        self.verbose     = verbose
        
    def fit(self,X,y=None):
        if self.verbose:
            print("### TransformDate (fit) of ",self.column,", Shape of X : ", X.shape )
        return self
    
    def transform(self,X):
        df = X.copy()
        df[self.column]          = pd.to_datetime(df[self.column], format = self.date_format, errors = 'coerce')
        df[self.column+'_year']  = df[self.column].dt.year
        df[self.column+'_month'] = df[self.column].dt.month
        df[self.column+'_day']   = df[self.column].dt.day
        cyclical                 = CyclicalFeatures(variables=None, drop_original=True)
        X                        = cyclical.fit_transform(df[[self.column+'_month',self.column+'_day']])
        df                       = pd.concat([df,X],axis=1).drop([self.column+'_month',self.column+'_day',self.column], axis=1)
        if self.verbose:
            print("### TransformDate (transform) of ",self.column,", Shape of X : ", df.shape )
        return df

