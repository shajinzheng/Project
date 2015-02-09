__author__ = 'Miroslaw Horbal'
__email__ = 'miroslaw@gmail.com'
__date__ = '14-06-2013'

from numpy import array, hstack
from sklearn import metrics, cross_validation, linear_model
from scipy import sparse
from itertools import combinations

import numpy as np
import pandas as pd
import time

SEED = 25

def group_data(data, degree=3, hash=hash):
    """ 
    numpy.array -> numpy.array
    
    Groups all columns of data into all combinations of triples
    """
    new_data = []
    m,n = data.shape

    # JaySha - Start
    # 1. In every for-loop, we construct a new feature for each training example
    # from original features indexed by indices
    # 2. Then append the new features of all training examples as a list to new_data
    # 3. After the for-loop, len(new_data) is degree-combination of {1, 2, ..., n}
    # len(new_data[0]) is m
    # 4. We need to transpose array(new_data) s.t. every row corresponds to a training example
    # JaySha - End

    for indicies in combinations(range(n), degree):
        new_data.append([hash(tuple(v)) for v in data[:,indicies]])
    return array(new_data).T

def OneHotEncoder(data, keymap=None):
     """
     OneHotEncoder takes data matrix with categorical columns and
     converts it to a sparse binary matrix.
     
     Returns sparse binary matrix and keymap mapping categories to indicies.
     If a keymap is supplied on input it will be used instead of creating one
     and any categories appearing in the data that are not in the keymap are
     ignored
     """
     if keymap is None:
          keymap = []

          # JaySha - Start
          # 1. If A is of type numpy.ndarray, for v in A, v will be each row of A
          # 2. keymap is a list, len(keymap) == data.shape[1]
          # JaySha - End
          for col in data.T:
               uniques = set(list(col))
               keymap.append(dict((key, i) for i, key in enumerate(uniques)))
     total_pts = data.shape[0]
     outdat = []
     for i, col in enumerate(data.T):
          km = keymap[i]
          num_labels = len(km)
          spmat = sparse.lil_matrix((total_pts, num_labels))
          for j, val in enumerate(col):
               if val in km:
                    spmat[j, km[val]] = 1
          outdat.append(spmat)

     # JaySha - Start
     # function: sparse.hstack()
     # Input: sequence of sparse matrices with compatible shapes
     # Output: sparse matrix
     # lil_matrix.tocsr(): Return Compressed Sparse Row format arrays for this matrix
     # JaySha - End
     outdat = sparse.hstack(outdat).tocsr()
     return outdat, keymap

def create_test_submission(filename, prediction):
    content = ['id,ACTION']
    for i, p in enumerate(prediction):
        content.append('%i,%f' %(i+1,p))
    f = open(filename, 'w')
    f.write('\n'.join(content))
    f.close()
    print 'Saved'

# This loop essentially from Paul's starter code
def cv_loop(X, y, model, N):
    mean_auc = 0.
    for i in range(N):
        X_train, X_cv, y_train, y_cv = cross_validation.train_test_split(
                                       X, y, test_size=.20, 
                                       random_state = i*SEED)
        model.fit(X_train, y_train)
        preds = model.predict_proba(X_cv)[:,1]
        auc = metrics.roc_auc_score(y_cv, preds)
        print "AUC (fold %d/%d): %f" % (i + 1, N, auc)
        mean_auc += auc
    return mean_auc/N
    
def main(train='train.csv', test='test.csv', submit='logistic_pred.csv'):
    # JaySha - Start
    # train_data and test_data are of same type - "pandas.core.frame.DataFrame"
    # JaySha - End
    print "Reading dataset..."
    train_data = pd.read_csv(train)
    test_data = pd.read_csv(test)

    # JaySha - Start
    # Ref1:  http://www.gregreda.com/2013/10/26/working-with-pandas-dataframes/
    # Ref2:  http://pandas.pydata.org/pandas-docs/stable/indexing.html
    # 1. .ix supports mixed integer and label-based access
    # 2. In this case, we leave out the first column, since it stands for label; we leave out
    # the last column, since it somehow contains the similar information with some column before
    # 3. np.vstack: stack arrays vertically
    # JaySha - End
    all_data = np.vstack((train_data.ix[:,1:-1], test_data.ix[:,1:-1]))
    num_train = np.shape(train_data)[0]
    
    # Transform data
    print "Transforming data..."
    dp = group_data(all_data, degree=2) 
    dt = group_data(all_data, degree=3)

    # JaySha - Start
    # 1. X/X_test contains training/test examples with original features
    # 2. X_2/X_test_2 and X_3/X_test_3 contain training/test examples with grouped features
    # JaySha - End
    y = array(train_data.ACTION)
    X = all_data[:num_train]
    X_2 = dp[:num_train]
    X_3 = dt[:num_train]

    X_test = all_data[num_train:]
    X_test_2 = dp[num_train:]
    X_test_3 = dt[num_train:]

    # JaySha - Start
    # Note: It is hstack, not vstack
    # JaySha - End
    X_train_all = np.hstack((X, X_2, X_3))
    X_test_all = np.hstack((X_test, X_test_2, X_test_3))
    num_features = X_train_all.shape[1]
    
    model = linear_model.LogisticRegression()
    
    # Xts holds one hot encodings for each individual feature in memory
    # speeding up feature selection

    # JaySha - Start
    # 1. X_train_all is of type numpy.ndarray
    # 2. X_train_all[:, [i]] return the values as in a column
    # 3. Xts is a list, each element is OneHotEncoding for each individual feature
    # JaySha - End
    Xts = [OneHotEncoder(X_train_all[:,[i]])[0] for i in range(num_features)]
    
    print "Performing greedy feature selection..."
    score_hist = []
    N = 10
    good_features = set([])

    # Greedy feature selection loop
    # JaySha - Start
    # 1. Greedy Lemma: Inside each while-loop, we add a feature not in good_features with
    # largest cv_loop score to good_features
    # JaySha - End
    while len(score_hist) < 2 or score_hist[-1][0] > score_hist[-2][0]:
        scores = []
        for f in range(len(Xts)):
            if f not in good_features:
                feats = list(good_features) + [f]
                Xt = sparse.hstack([Xts[j] for j in feats]).tocsr()
                t0 = time.clock()
                score = cv_loop(Xt, y, model, N)
                print "Time for one cv_loop is: %f" % (time.clock() - t0)
                scores.append((score, f))
                print "Feature: %i Mean AUC: %f" % (f, score)

        # JaySha - Start
        # add feature with largest score to good_features and score_hist
        # JaySha - End
        good_features.add(sorted(scores)[-1][1])
        score_hist.append(sorted(scores)[-1])
        print "Current features: %s" % sorted(list(good_features))
    
    # Remove last added feature from good_features
    good_features.remove(score_hist[-1][1])
    good_features = sorted(list(good_features))
    print "Selected features %s" % good_features
    
    print "Performing hyperparameter selection..."
    # Hyperparameter selection loop, i.e. hyperparameter means the parameter for regularizationwww.
    score_hist = []
    Xt = sparse.hstack([Xts[j] for j in good_features]).tocsr()
    Cvals = np.logspace(-4, 4, 15, base=2)
    for C in Cvals:
        model.C = C
        score = cv_loop(Xt, y, model, N)
        score_hist.append((score,C))
        print "C: %f Mean AUC: %f" %(C, score)
    bestC = sorted(score_hist)[-1][1]
    print "Best C value: %f" % (bestC)
    
    print "Performing One Hot Encoding on entire dataset..."
    Xt = np.vstack((X_train_all[:,good_features], X_test_all[:,good_features]))
    Xt, keymap = OneHotEncoder(Xt)
    X_train = Xt[:num_train]
    X_test = Xt[num_train:]
    
    print "Training full model..."
    model.fit(X_train, y)
    
    print "Making prediction and saving results..."
    preds = model.predict_proba(X_test)[:,1]
    create_test_submission(submit, preds)
    
if __name__ == "__main__":
    args = { 'train':  'train.csv',
             'test':   'test.csv',
             'submit': 'logistic_regression_pred.csv' }
    main(**args)
    
