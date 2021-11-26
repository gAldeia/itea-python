import os
import sys
import glob
import argparse

from multiprocessing import Pool
from itertools       import product

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

itea_configuration = {
    'gens'            : 250,
    'popsize'         : 400,
    'max_terms'       : 15,
    'expolim'         : (-3, 3),
    'verbose'         : False,
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


def experiment_worker(ds_name, rep):

    with FileLock(filelock_name):
        try:
            ds_data = pd.read_csv(f'{datasets_folder}/{ds_name}.csv', delimiter=',')
        except Exception as e:
            print(f'Could not load {ds_name} data set. Got exception {e}')
            sys.exit()

        columns   = [
            'Dataset', 'Rep', 'RMSE_train', 'RMSE_test', 'Exectime', 'Expr']

        results   = {c:[] for c in columns}
        resultsDF = pd.DataFrame(columns=columns)

        if os.path.isfile(results_fname):
            resultsDF = pd.read_csv(results_fname)
            results   = resultsDF.to_dict('list')

        # Checking if this ds_name-repetition was already executed    
        if len(resultsDF[
            (resultsDF['Dataset']==ds_name) & (resultsDF['Rep']==rep)])>0:

            print(f'already executed experiment {ds_name}-{rep}')

            return

    print(f'Executing experiment {ds_name}-{rep}...')

    # Random train and test split
    X_train, X_test, y_train, y_test = train_test_split(
        ds_data.iloc[:, :-1].astype('float64'),
        ds_data.iloc[:, -1].astype('float64'),
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

        results['Dataset'].append(ds_name)
        
        results['RMSE_train'].append(mean_squared_error(
            itexpr.predict(X_train), y_train, squared=False))
        
        results['RMSE_test'].append(mean_squared_error(
            itexpr.predict(X_test), y_test, squared=False))

        results['Rep'].append(rep)
        results['Expr'].append(str(itexpr))
        results['Exectime'].append(reg.exectime_)
        
        df = pd.DataFrame(results)
        df.to_csv(results_fname, index=False)


if __name__ == '__main__':

    # Finding available datasets in the specific folder. data sets must be
    # a csv file with a header. Removing the path and file extension
    # before creating the final data sets list
    datasets = [os.path.splitext(os.path.basename(ds))[0]
        for ds in glob.glob('data sets/*.csv')]

    parser = argparse.ArgumentParser()

    parser.add_argument('-d', '--datasets', type=list, default=datasets,
        help='name of the data sets to run')

    parser.add_argument('-n', '--n_repetitions', type=int, default=30,
        help='number of repetitions for each dataset')

    parser.add_argument('-p', '--processes', type=int, default=1,
        help='number of subprocesses to execute the benchmark')   

    args = parser.parse_args()

    if len(sys.argv)== 1:
        print('You can specify a list of data sets to execute the experiments '
              'with --datasets. If no list is given, then all csv files inside '
              '"\data sets" folder will be executed.'
             f'executing for data sets: {args.datasets}.\n\n'
        
              'Since ITEA is stochastic, you can set a number of repetitions '
              'for each data set with --n_repetitions. If none is given, then '
              '30 repetitions will be made for each data set.'
             f'Executing with {args.n_repetitions} repetitions.\n\n' 
        
              'Finally, you can execute multiple experiments in parallel by '
              'specifying the number of subprocesses with --processes. If none '
              'is given, then 1 process will be used.'
             f'Using {args.processes} subprocesses.')

    print("PRESS ENTER TO START")
    input()

    if not set(args.datasets).issubset(set(datasets)):
        print(f'Data sets {args(sys.datasets[1])} not found.')
        sys.exit()

    # Creating all experiments configurations
    configurations = list(product(
        [str(ds) for ds in args.datasets],
        range(args.n_repetitions)
    ))

    np.random.shuffle(configurations)

    p_pool = Pool(args.processes)
    p_pool.starmap(experiment_worker, configurations)
    
    p_pool.close()
    p_pool.join()
        
    # Reporting the results ----------------------------------------------------
    import pandas as pd

    results = pd.read_csv('regression_benchmark_res.csv')

    # Will print the mean of all executions in the results file for each data set
    print(results.drop(columns=['Rep']).groupby('Dataset').mean())