import os
import sys


import pandas  as pd
import numpy   as np
from itea.regression import ITEA_regressor

from sklearn.model_selection import train_test_split
from sklearn.metrics         import mean_squared_error
from filelock                import FileLock

import warnings; warnings.filterwarnings('ignore')


filelock_name      = './regression_benckmark.lock'
datasets_folder    = './data sets'
results_fname      = './regression_benchmark_res.csv'
n_repetitions      = 30
itea_configuration = {
    'gens'            : 250,
    'popsize'         : 250,
    'max_terms'       : 10,
    'expolim'         : (-2, 2),
    'verbose'         : 25,
    'random_state'    : None,
    'simplify_method' : None,
    'tfuncs' : {
        'log'      : np.log,
        'sqrt.abs' : lambda x: np.sqrt(np.abs(x)),
        'id'       : lambda x: x,
        'sin'      : np.sin,
        'cos'      : np.cos,
        'exp'      : np.exp
    },
    'tfuncs_dx' : {
        'log'      : lambda x: 1/x,
        'sqrt.abs' : lambda x: x/( 2*(np.abs(x)**(3/2)) ),
        'id'       : lambda x: np.ones_like(x),
        'sin'      : np.cos,
        'cos'      : lambda x: -np.sin(x),
        'exp'      : np.exp,
    }
}


if __name__ == '__main__':

    datasets = ['airfoil',
                'concrete',
                'energyCooling',
                'energyHeating',
                'grossOutput',
                'qsarAquaticToxicity',
                'wineRed',
                'wineWhite',
                'yachtHydrodynamics']

    if len(sys.argv)== 1:
        print('Call the benchmark specifying as argument the data set to '
             f'execute. The algorithm will perform {n_repetitions} runs (with '
              'random partitions of train-test) on the data set. '
             f'Possible data sets are: {datasets}.')
             
        sys.exit()

    if str(sys.argv[1]) not in datasets:
        print(f'Data set {str(sys.argv[1])} not found.')
        sys.exit()

    ds = str(sys.argv[1])

    columns   = [
        'Dataset', 'Rep', 'RMSE_train', 'RMSE_test', 'Exectime', 'Expr']

    results   = {c:[] for c in columns}
    resultsDF = pd.DataFrame(columns=columns)

    if os.path.isfile(results_fname):
        resultsDF = pd.read_csv(results_fname)
        results   = resultsDF.to_dict('list')

    try:
        ds_data = pd.read_csv(f'{datasets_folder}/{ds}.csv', delimiter=',')
    except Exception as e:
        print(f'Could not load {ds} data set. Got exception {e}')
        sys.exit()
    
    for rep in range(n_repetitions):
        if len(resultsDF[
            (resultsDF['Dataset']==ds) & (resultsDF['Rep']==rep)])==1:

            print(f'already evaluated {ds}-{rep}')

            continue

        # Random train and test split
        X_train, X_test, y_train, y_test = train_test_split(
            ds_data.iloc[:, :-1], ds_data.iloc[:, -1],
            test_size=0.33, random_state=None
        )

        reg = ITEA_regressor(
            labell=ds_data.columns[:-1], **itea_configuration)
            
        reg.fit(X_train, y_train)
        itexpr = reg.bestsol_

        # Locking to avoid parallel writing if multiple datasets are being
        # executed    
        with FileLock(filelock_name):

            # Retrieving the latest results 
            if os.path.isfile(results_fname):
                resultsDF = pd.read_csv(results_fname)
                results   = resultsDF.to_dict('list')

            results['Dataset'].append(ds)
            
            results['RMSE_train'].append(mean_squared_error(
                itexpr.predict(X_train), y_train, squared=False))
            
            results['RMSE_test'].append(mean_squared_error(
                itexpr.predict(X_test), y_test, squared=False))

            results['Rep'].append(rep)
            results['Expr'].append(str(itexpr))
            results['Exectime'].append(reg.exectime_)
            
            df = pd.DataFrame(results)
            df.to_csv(results_fname, index=False)

    print('done')

    # Reporting the results
    print("====================================")
    print(f"Data set: {ds}")
    print(resultsDF.drop(['Rep'], axis=1).mean())
    print("====================================")
