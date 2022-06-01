import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

import os

if __name__ == '__main__':
    output_dir = 'time_plots'

    plt.style.use('seaborn-darkgrid')

    # Read data
    df_gpu = pd.read_csv('full_run_results_gpu.csv')
    df_cpu = pd.read_csv('full_run_test_cpu.csv').sort_values(by=['indexVal'])
    df_matlab = pd.read_csv('../matlab/full_run_test.csv').sort_values(by=['indexVal'])
    df_R = pd.read_csv('../R/full_run_test.csv', sep=' ').sort_values(by=['indexVal'])

    # Set up other index values
    df_gpu['indexVal'] = [i+1 for i in df_gpu.index]

    # Find intersection of indices
    list_of_idx = list(set(df_R['indexVal']).intersection(set(df_gpu['indexVal'])).intersection(set(df_cpu['indexVal'])).intersection(set(df_matlab['indexVal'])))

    # Subset dfs
    df_R = df_R[df_R['indexVal'].isin(list_of_idx)].sort_values(by=['indexVal'])
    df_cpu = df_cpu[df_cpu['indexVal'].isin(list_of_idx)].sort_values(by=['indexVal'])
    df_gpu = df_gpu[df_gpu['indexVal'].isin(list_of_idx)].sort_values(by=['indexVal'])
    df_matlab = df_matlab[df_matlab['indexVal'].isin(list_of_idx)].sort_values(by=['indexVal'])

    # Modify df_R
    df_R = df_R[[i for i in df_R.columns if i != 'indexVal']]
    df_gpu = df_gpu[[i for i in df_gpu.columns if i != 'indexVal']]
    df_cpu = df_cpu[[i for i in df_cpu.columns if i != 'indexVal']]
    df_matlab = df_matlab[[i for i in df_matlab.columns if i != 'indexVal']]

    # NOTE: Failed to calculate std properly for chi^2 dist in Python, so we need to correct
    df_gpu[df_gpu['dist1'] == 'chisquare']['std1'] = np.sqrt(np.power(df_gpu[df_gpu['dist1'] == 'chisquare']['std1'].values,2)*2)
    df_gpu[df_gpu['dist2'] == 'chisquare']['std2'] = np.sqrt(np.power(df_gpu[df_gpu['dist2'] == 'chisquare']['std2'].values,2)*2)

    df_cpu[df_cpu['dist1'] == 'chisquare']['std1'] = np.sqrt(np.power(df_cpu[df_cpu['dist1'] == 'chisquare']['std1'].values,2)*2)
    df_cpu[df_cpu['dist2'] == 'chisquare']['std2'] = np.sqrt(np.power(df_cpu[df_cpu['dist2'] == 'chisquare']['std2'].values,2)*2)

    df_gpu[df_gpu['dist1'] == 'exponential']['std1'] = df_gpu[df_gpu['dist1'] == 'exponential']['mean1']
    df_gpu[df_gpu['dist1'] == 'exponential']['std2'] = df_gpu[df_gpu['dist1'] == 'exponential']['mean2']

    df_cpu[df_cpu['dist1'] == 'exponential']['std1'] = df_cpu[df_cpu['dist1'] == 'exponential']['mean1']
    df_cpu[df_cpu['dist2'] == 'exponential']['std2'] = df_cpu[df_cpu['dist1'] == 'exponential']['mean2']
    
    df_R[df_R['dist1'] == 'exponential']['std1'] = df_R[df_R['dist1'] == 'exponential']['mean1']
    df_R[df_R['dist2'] == 'exponential']['std2'] = df_R[df_R['dist1'] == 'exponential']['mean2']

    total_length = np.min([len(df_gpu), len(df_cpu), len(df_matlab), len(df_R)])


    #df = df_R.iloc[:total_length]
    df = df_R.copy(deep=True)
    

    for name in list(set(df_cpu.columns).difference(set(df.columns))):
        df[name] = df_cpu[name].values
    

    for name in list(set(df_matlab.columns).difference(set(df.columns))):
        df[name] = df_matlab[name].values
    
    for name in list(set(df_gpu.columns).difference(set(df.columns))):
        df[name] = df_gpu[name].values

    # Compute total number of points
    df['n'] = df['n1'] + df['n2']

    rename_dict = {'time_pald_cupy_multistream' : 'CuPy Multistream',
                    'time_pald_cupy_loop' : 'CuPy',
                    'matlab_time' : 'Parallel MATLAB',
                    'time_pald_numpy' : 'NumPy',
                    'time_pald_R' : 'Sequential R'}

    #cmap = plt.cm.get_cmap('tab10', len(rename_dict.keys())).reversed().colors
    cmap = sns.color_palette()[:len(rename_dict.keys())]
    palette_dict = {k : c for k,c in zip(rename_dict.values(), cmap)}

    

    df.columns = [rename_dict[i] if i in rename_dict.keys() else i for i in df.columns]

    print('\n1820 Points Mean')
    print(df[df['n'] == 1820][['Sequential R', 'NumPy', 'CuPy Multistream', 'CuPy', 'Parallel MATLAB']].mean())

    # Pivot on sizes
    df_runtime = pd.melt(df, 
                    id_vars=['n','mean1', 'std1', 'n1', 'dim1', 'dist1', 'mean2', 'std2', 'n2', 'dim2','dist2'], 
                    value_vars=['NumPy', 'Sequential R', 'CuPy Multistream', 'CuPy', 'Parallel MATLAB'],
                    var_name='Run Type',
                    value_name='Runtime (s)')

    # Plot size vs time
    ax = sns.lineplot(x='n', y='Runtime (s)', hue='Run Type', ci='sd', marker='o', palette=palette_dict, data=df_runtime)
    ax.set_ylim(bottom=0)
    ax.set_xlim(left=0)
    plt.title('Runtime vs. Problem Size')

    # Save figure
    plt.savefig(os.path.join(output_dir,'runtime.png'))
    plt.clf()

    # Pivot on sizes
    df_runtime = pd.melt(df, 
                    id_vars=['n','mean1', 'std1', 'n1', 'dim1', 'dist1', 'mean2', 'std2', 'n2', 'dim2','dist2'], 
                    value_vars=['CuPy Multistream', 'CuPy'],
                    var_name='Run Type',
                    value_name='Runtime (s)')

    # Plot size vs time
    ax = sns.lineplot(x='n', y='Runtime (s)', hue='Run Type', ci='sd', marker='o', palette=palette_dict, data=df_runtime)
    ax.set_ylim(bottom=0)
    ax.set_xlim(left=0)
    plt.title('Runtime vs. Problem Size')

    # Save figure
    plt.savefig(os.path.join(output_dir,'runtime_fast.png'))
    plt.clf()

    # Pivot on sizes
    df_runtime = pd.melt(df, 
                    id_vars=['n','mean1', 'std1', 'n1', 'dim1', 'dist1', 'mean2', 'std2', 'n2', 'dim2','dist2'], 
                    value_vars=['NumPy', 'Sequential R', 'Parallel MATLAB'],
                    var_name='Run Type',
                    value_name='Runtime (s)')

    # Plot size vs time
    ax = sns.lineplot(x='n', y='Runtime (s)', hue='Run Type', ci='sd', marker='o', palette=palette_dict, data=df_runtime)
    ax.set_ylim(bottom=0)
    ax.set_xlim(left=0)
    plt.title('Runtime vs. Problem Size')

    # Save figure
    plt.savefig(os.path.join(output_dir,'runtime_slow.png'))
    plt.clf()

    # Speedup
    #df_speedup = df.copy(deep=True).groupby(['n','mean1', 'std1', 'n1', 'dim1', 'dist1', 'mean2', 'std2', 'n2', 'dim2','dist2']).mean()
    df_speedup = df.copy(deep=True)
    df_speedup = pd.DataFrame(df_speedup)
    

    for col_name in ['NumPy', 'CuPy Multistream', 'CuPy', 'Parallel MATLAB']:
        df_speedup[col_name] = df_speedup['Sequential R']/df_speedup[col_name]

    print('\nMean')
    print(df_speedup[['NumPy', 'CuPy Multistream', 'CuPy', 'Parallel MATLAB']].mean())

    print('\nMax')
    print(df_speedup[['NumPy', 'CuPy Multistream', 'CuPy', 'Parallel MATLAB']].max())

    print('\n1820 Points Mean')
    print(df_speedup[df_speedup['n'] == 1820][['NumPy', 'CuPy Multistream', 'CuPy', 'Parallel MATLAB']].mean())

    df_speedup = pd.melt(df_speedup, 
                    id_vars=['n','mean1', 'std1', 'n1', 'dim1', 'dist1', 'mean2', 'std2', 'n2', 'dim2','dist2', 'Sequential R'], 
                    value_vars=['NumPy', 'CuPy Multistream', 'CuPy', 'Parallel MATLAB'],
                    var_name='Run Type',
                    value_name='Speedup')

    # Plot size vs time
    ax = sns.lineplot(x='n', y='Speedup', hue='Run Type', ci='sd', marker='o', palette=palette_dict, data=df_speedup)
    ax.set_ylim(bottom=0)
    ax.set_xlim(left=0)
    plt.ylabel(r'Speedup $\left(\frac{T_{R}}{T_{New}}\right)$')
    plt.title('Speedup vs. Problem Size')

    # Save figure
    plt.savefig(os.path.join(output_dir,'speedup.png'))
    plt.clf()

    # Speedup
    #df_speedup = df.copy(deep=True).groupby(['n','mean1', 'std1', 'n1', 'dim1', 'dist1', 'mean2', 'std2', 'n2', 'dim2','dist2']).mean()
    df_speedup = df.copy(deep=True)
    df_speedup = pd.DataFrame(df_speedup.reset_index())

    for col_name in ['CuPy Multistream', 'CuPy']:
        df_speedup[col_name] = df_speedup['Sequential R']/df_speedup[col_name]
    
    df_speedup = pd.melt(df_speedup, 
                    id_vars=['n','mean1', 'std1', 'n1', 'dim1', 'dist1', 'mean2', 'std2', 'n2', 'dim2','dist2', 'Sequential R'], 
                    value_vars=['CuPy Multistream', 'CuPy'],
                    var_name='Run Type',
                    value_name='Speedup')

    # Plot size vs time
    ax = sns.lineplot(x='n', y='Speedup', hue='Run Type', ci='sd', marker='o', palette=palette_dict, data=df_speedup)
    ax.set_ylim(bottom=0)
    ax.set_xlim(left=0)
    plt.ylabel(r'Speedup $\left(\frac{T_{R}}{T_{New}}\right)$')
    plt.title('Speedup vs. Problem Size')

    # Save figure
    plt.savefig(os.path.join(output_dir,'speedup_fast.png'))
    plt.clf()

    # Speedup
    #df_speedup = df.copy(deep=True).groupby(['n','mean1', 'std1', 'n1', 'dim1', 'dist1', 'mean2', 'std2', 'n2', 'dim2','dist2']).mean()
    df_speedup = df.copy(deep=True)
    df_speedup = pd.DataFrame(df_speedup.reset_index())

    for col_name in ['NumPy', 'Parallel MATLAB']:
        df_speedup[col_name] = df_speedup['Sequential R']/df_speedup[col_name]
    
    df_speedup = pd.melt(df_speedup, 
                    id_vars=['n','mean1', 'std1', 'n1', 'dim1', 'dist1', 'mean2', 'std2', 'n2', 'dim2','dist2', 'Sequential R'], 
                    value_vars=['NumPy', 'Parallel MATLAB'],
                    var_name='Run Type',
                    value_name='Speedup')

    # Plot size vs time
    ax = sns.lineplot(x='n', y='Speedup', hue='Run Type', ci='sd', marker='o', palette=palette_dict, data=df_speedup)
    ax.set_ylim(bottom=0)
    ax.set_xlim(left=0)
    plt.ylabel(r'Speedup $\left(\frac{T_{R}}{T_{New}}\right)$')
    plt.title('Speedup vs. Problem Size')

    # Save figure
    plt.savefig(os.path.join(output_dir,'speedup_slow.png'))
    plt.clf()

    # Bound
    df_bound = df.copy(deep=True)

    for col_name in ['NumPy', 'CuPy Multistream', 'CuPy', 'Parallel MATLAB']:
        df_bound[col_name] = df_bound['Sequential R']/df_bound[col_name]
    
    df_bound = pd.melt(df_bound, 
                    id_vars=['n','mean1', 'std1', 'n1', 'dim1', 'dist1', 'mean2', 'std2', 'n2', 'dim2','dist2', 'bound_pald_R'], 
                    value_vars=['bound_pald_cupy_multistream', 'bound_pald_cupy_loop', 'matlab_bound', 'bound_pald_numpy'],
                    var_name='Run Type',
                    value_name='Bound')

    # Plot size vs time
    ax = sns.scatterplot(x='bound_pald_R', y='Bound', hue='Run Type', style='Run Type', data=df_bound)
    ax.set_ylim(bottom=0)
    ax.set_xlim(left=0)

    # Save figure
    plt.savefig(os.path.join(output_dir,'bound_scatter.png'))
    plt.clf()

    # Strong scaling
    df_strong_scaling = pd.read_csv('../matlab/full_results_scaling.csv')
    df_strong_scaling.columns = [rename_dict[i] if i in rename_dict.keys() else i for i in df_strong_scaling.columns]

    palette_dict['Theoretical Runtime (Amdahl\'s Law)'] = 'black'

    def speedup_fun(p, n):
        #f = 1 - ((4*n*n + 12*n + 4)/(10*n*n*n - 6*n*n + 2))
        f = 1 - ((20*n*n - 15*n + 3)/(16*n*n*n -12*n*n + 2*n + 2))
        return np.array((df_strong_scaling[df_strong_scaling['Processors'] == 1]['Parallel MATLAB'].mean()*((1-f) + f/p)), dtype=np.float32)

    vals = np.linspace(1, 44)
    #vals = np.unique(df_strong_scaling['Processors'].values)

    # Plot size vs time
    #ax = sns.lineplot(x='Processors', y='Parallel MATLAB', hue=['Parallel MATLAB']*len(df_strong_scaling), ci='sd', marker='o', palette=palette_dict, data=df_strong_scaling)
    ax = sns.lineplot(x=list(df_strong_scaling['Processors'].values)+list(vals), 
                    y=list(df_strong_scaling['Parallel MATLAB'].values) + list(speedup_fun(vals, 2000)), 
                    hue=['Parallel MATLAB']*len(df_strong_scaling) + ['Theoretical Runtime (Amdahl\'s Law)']*len(vals), 
                    ci='sd', markers=['o', None], palette=palette_dict)
    ax.set_ylim(bottom=0)
    ax.set_xlim(left=0)
    plt.ylabel('Runtime (s)')
    plt.title(r'Runtime vs. Number of Processors for $N=2000$')

    # Save figure
    plt.savefig(os.path.join(output_dir,'strong_scaling.png'))
    plt.clf()

    