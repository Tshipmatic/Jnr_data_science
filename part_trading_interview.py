import yfinance as yf
import yahoo_fin.stock_info as ys
import pandas as pd
pd.core.common.is_list_like = pd.api.types.is_list_like
from pandas.api.types import is_list_like
from pandas_datareader import data as pdr
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import OPTICS
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import coint, OLS
yf.pdr_override() 



def remove_missing_data(csv):
    '''
    inputs: 
            csv : raw csv file

    outputs: 
            dataframe of prices with no missing values        
    '''
    # Removing missing data
    universe_data = pd.read_csv('random_sp300.csv', index_col=0, parse_dates=[0])
    universe_data.index = pd.to_datetime(universe_data.index)
    num_missing = universe_data.isna().sum()
    min_missing = num_missing.value_counts().index.min()
    prices_df = universe_data[(
    num_missing[num_missing == min_missing]).index.tolist()].dropna()
    return prices_df

#Reduce price data to its componets
def dimension_reduction(prices_df, num_componets):
    '''
    input:
        df : cleaned dataframe from remove missing data
        num_componets : no. of components used in PCA

    output:
            transposed principal components, returns    
    '''
    # Calculate returns
    returns600 = prices_df.pct_change().dropna()

    # Normalizing returns using (Return - mean(return))/Return std  : Removing outliers
    # so that each one of them contributes equally to the analysis.
    normalized_returns600 = (returns600 - returns600.mean())/(returns600.std())

    # Application of PCA
    pca = PCA(n_components=num_componets)
    pca.fit(normalized_returns600)
    return pca.components_.T,returns600
    

def cluster_using_optics(minsamples, pca_componets,returns600):
    '''
    Group similar stocks

    input:
            minsamples : minimum points to form a core point
     output:
            1. a 2D plot of clustered stocks  and 
            2. new dataframe that contains all the correctly grouped stocks     
    '''
    # Using OPTICS to group similar stocks
    print(f'min_samples',minsamples)
    op = OPTICS(min_samples = minsamples)
    # Transpose to make the rows = the number of stocks
    op.fit(pca_componets)
    labels = op.labels_
    # Calculating the number of cluster s
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    print ("\nClusters discovered: %d" % n_clusters_)

    clustered = op.labels_

    # Applying t-SNE to reduce the dimension from 5 to 2 in order to plot in 2D
    X_tsne = TSNE(learning_rate=1000 ,init='pca').fit_transform(pca_componets)

    # Express the 2D array from TSNE as a data frame
    groups = pd.DataFrame(X_tsne,  columns=['t-SNE Dim. 1', 't-SNE Dim. 2'] ,index=returns600.columns)
# Add the labels columns to the datafram
    groups['grouped']=labels

    # Noisy samples and points which are not included in a leaf cluster of cluster_hierarchy_ are labeled as -1.
# Remove stocks that have label = -1
    new_groups = groups.loc[groups['grouped'] != -1]

    # Visualizing clusters
    names = new_groups.index

    plt.figure(figsize=(30,20))
    sns.scatterplot(x=new_groups.columns[0], y=new_groups.columns[1], hue=new_groups.index, data=new_groups,  legend = False)
    # plt.xlim(-200,200)
    # plt.ylim(-250,750)

    for i, language in enumerate (names):
        plt.annotate(language, ( new_groups['t-SNE Dim. 1'][i]+0.7,  new_groups['t-SNE Dim. 2'][i]+0.5) )

    return new_groups  


# Selects pairs of stocks from the universe that satiesfies all three tests namely, Engle-Granger, Hurst exponent and half life tests
def unsupervised_pair_selector(new_groups,prices_df):
    '''
    input:
        new_groups : dataframe containing to correctly classified stocks (obtained from the OPTICS)
    output:
          pairs of stocks satiesfying all three tests
    '''
    # tickers of all clusters
    clustered_tickers = new_groups.index
    # the corresponding price dataframe
    clusted_df = prices_df[clustered_tickers]
    b = lambda ts: ts.iloc[:,0].pct_change().dropna().std()/ts.iloc[:,1].pct_change().dropna().std() 

    def pair_selection(clusted_df):
        '''
        clusted_df is the dataframe that contain all the prices of the clusted tickers

        It returns cointegrated pairs calculated using the Engle-Granger test
        '''
        # Number of stocks in the clusters
        no_stock = clusted_df.shape[1]
        print(f'clusted_df')
        pairs = []
        cvs = []
        for i in range(no_stock-1):
            s1 = clusted_df[clusted_df.columns[i]]
            s2 = clusted_df[clusted_df.columns[i+1]] 
            t_statistc1, pvalue1,ccv = coint(s1,s2,'c')    # 1st possibility
            t_statistc2, pvalue2,ccv = coint(s2,s1,'c')    # 2nd possibility

            # Combination with the lowest t_statistcis is selected
            if t_statistc1< t_statistc2:
                pairs.append((clusted_df.columns[i],clusted_df.columns[i+1]))
            if  t_statistc1 > t_statistc2:  
                pairs.append((clusted_df.columns[i+1],clusted_df.columns[i]))
        # return pairs
        print(f'No. of pairs from EG test: {len(pairs)}\n')
        return Apply_hust_exponent_test(pairs)




    def Apply_hust_exponent_test(pairs, max_lag=20):
        '''
        inputs:
                pairs : list of pairs obtained from from Engle-Granger (EG)
                max_lag : max number of lags (If its too high it might produce incorrect results)

        outputs:
                produces a list of pairs that certfies the EG test      
        '''
        # b = std(log return(A))/std(log return(B)) : eqn obtain from the INTRODUCING HURST EXPONENT IN PAIR TRADING aka hedge ratio
        b = lambda ts: ts.iloc[:,0].pct_change().dropna().std()/ts.iloc[:,1].pct_change().dropna().std() 

        
        def get_hurst_exponent(time_series, max_lag):
            """
            Inputs : 
                    time_series : time series for a pair 
                    max_lag :  max number of lags (If its too high it might produce incorrect results)
            
            Outputs:
                    Returns the Hurst Exponent of the time series""" 
            
            lags = range(2, max_lag)

            # variances of the lagged differences
            tau = [np.std(np.subtract(time_series[lag:], time_series[:-lag])) for lag in lags]

            # calculate the slope of the log plot -> the Hurst Exponent
            reg = np.polyfit(np.log(lags), np.log(tau), 1)

            return reg[0]

        updated_pairs = []
        hurst_values = []
        hedge_ratios = []
        counter = 0
        count_new_pairs  = 0

        for pair in pairs:
            par_1 = clusted_df.loc[:, pair]
            pair_AB = par_1.loc[:,pair[0]]-b(par_1)*par_1.loc[:,pair[1]] #pair series
            # print(f'pair_AB{pair_AB}')
            counter += 1
            H = get_hurst_exponent(pair_AB.values,max_lag) # Hurst exponent
            # print(f'Hust for {pair} is {H}')
            if H<0.5:
                updated_pairs.append(pair)
                hurst_values.append(H)
                hedge_ratios.append(b(par_1))
                # print(f'Pair {pair} survived')
                count_new_pairs += 1

        print(f'from {counter} pairs from EG test {count_new_pairs} survived the test from Hursttest. \n')
        return final_pairs(updated_pairs) #,hurst_values,hedge_ratios


    def final_pairs(pairs_from_hurst):
        '''
        This function calculates the half life of a pair, then discards pairs with a half life less than a day and more than a year

        input:
            pairs_from_hurst : list pairs from the hurst function
        outputs :
                Pairs with a half life between a day and a year    
        '''
        half_life_pairs = []
        half_life = []
        counter = 0
        for pair in pairs_from_hurst:
            pair_df = clusted_df.loc[:, pair]  #dataframe for a single a pair of stock
            pair_series = pair_df.loc[:,pair[0]]-b(pair_df)*pair_df.loc[:,pair[1]] 

            previous_series = pair_series.shift()
            previous_series = previous_series.dropna()   #removing missing values
            delta_prices = pair_series - previous_series
            delta_prices = delta_prices.dropna() #removing missing values
            # pair_series, pair_series.shift(), pair_series.sub(pair_series.shift()[1:]),pair_series.sub(pair_series.shift()[1:]).dropna()
            model = OLS(delta_prices, previous_series-np.mean(previous_series))  #model(delta y, lambda*y(t-1)+mu)
            # delta_prices.plot()
            output = model.fit()
            lambda_ = output.params[0]
            lambda_
            half_life_in_days = -np.log(2)/lambda_
            # print(f'half life for {pair} is {half_life_in_days} days')
            if half_life_in_days >1 and half_life_in_days < 365:
                half_life_pairs.append(pair)
                half_life.append(half_life_in_days)

        print(f'No. of pairs after considering half life is {len(half_life_pairs)}')
        return half_life_pairs, half_life    
   
    # pairs_passed_EG_tests =  pair_selection(clusted_df)  
    # pairs_passed_Hurst_tests = Apply_hust_exponent_test(pairs_passed_EG_tests)
    # pairs_pass_all_tests = final_pairs(pairs_passed_Hurst_tests)

    return pair_selection(clusted_df)

#Generate information of selected stock     
def describe_data(final_pair,half_lives):
    '''
    inputs:
            final_pair, half_lives calculated from the half life function
    output:
            table that describe the final pairs        
            
    '''
    finall_df = pd.DataFrame(final_pair, columns = ['leg_1', 'leg_2'])
    finall_df['Half lifes (days)'] = half_lives
    return finall_df
    

    


