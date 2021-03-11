from sklearn.decomposition import PCA
import gc
import pandas as pd
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
import itertools

def scaling(data_to_train,data,column):
    """
    Fits and transforms min-max scaling to train data (data_to_train).
    Afterwards predicts this tranformation test data (data).
    """
    minik=data_to_train[column].min()
    maxik=data_to_train[column].max()
    data_to_train[column+'_scaled']=(data_to_train[column]-minik)/(maxik-minik)
    data[column+'_scaled']=(data[column]-minik)/(maxik-minik)
    gc.collect()
    return data_to_train,data

    
def pca_fit(data,sloupce,pocet):
    """
    Fits PCA model from data
    """
    pca = PCA(n_components=pocet)
    pca.fit(data[sloupce])
    return pca

def pca_transform(pca,data,sloupce):
    """
    predicts already fited PCA model
    """

    names=[]
    for i in range(len(sloupce)):
        names.append('pc{}'.format(i+1))
    df_pca = pd.DataFrame(data = pca.transform(data[sloupce]), columns = names)
    return df_pca

def findsubsets(s, n): 
    return list(itertools.combinations(s, n)) 

def knn_grid_search(cols_sure,cols_to_test,num_nn_to_test,X_train,X_test,y_train,y_test):
    """
    Looks for optimal knn predictors and number of neighbors
    """

    df_vysledku=pd.DataFrame(columns=['preds','num_nn','perf_train','perf_test'])

    for i in num_nn_to_test:
        for j in range(0,len(cols_to_test)+1):
            preds=findsubsets(cols_to_test,j)
            for x in list(preds):
                
                cols=cols_sure+list(x)
                print(i)
                print(cols)
        
                neigh = KNeighborsClassifier(n_neighbors=i,weights='uniform')
                neigh.fit(X_train[cols], y_train)

                predictions_train=neigh.predict_proba(X_train[cols])[:,1]
                predictions_test=neigh.predict_proba(X_test[cols])[:,1]

                print('Performance train')
                a=2*roc_auc_score(y_train, predictions_train)-1
                print(a)
                print('Performance test')
                b=2*roc_auc_score(y_test, predictions_test)-1
                print(b)
            
                df_vysledku=df_vysledku.append({'preds':x,'num_nn':i,'perf_train':a,'perf_test':b},ignore_index=True)
            
                print('XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX')
    return df_vysledku.sort_values(by='perf_test',ascending=False)


def knn_fit(cols,num_nn,X_train,y_train,X_predict):
    """
    Fits knn with givel columns and number of neighbours
    """
    neigh = KNeighborsClassifier(n_neighbors=num_nn,weights='uniform')
    neigh.fit(X_train[cols], y_train)

    predictions_train=neigh.predict_proba(X_predict[cols])[:,1]

    return predictions_train