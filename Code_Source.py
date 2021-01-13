


# -----------------------------------------------------------------------------
# import the necessary library
import numpy as np
import pandas as pd
import random 
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

from xgboost import XGBRegressor, plot_importance
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor, VotingRegressor, AdaBoostRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import BayesianRidge
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

from sklearn.metrics import mean_squared_error
from scipy.sparse import csr_matrix
# -----------------------------------------------------------------------------






# -----------------------------------------------------------------------------
# Data Pre-processing for training data

# loading raw training data
df_raw = pd.read_csv(r'C:\Users\123\Dropbox\Data Science\IMBD2020\0714train.csv')
df_raw.info() 

# --------------------------------------------------------
# set lists of simple columns' name for raw training data
columns_A1 = ['IA1_%a' % a for a in range(1,25)]
columns_A2 = ['IA2_%a' % a for a in range(1,25)]
columns_A3 = ['IA3_%a' % a for a in range(1,25)]
columns_A4 = ['IA4_%a' % a for a in range(1,25)]
columns_A5 = ['IA5_%a' % a for a in range(1,25)]
columns_A6 = ['IA6_%a' % a for a in range(1,25)]

columns_A = columns_A1 + columns_A2 + columns_A3 + columns_A4 + columns_A5 + columns_A6
columns_C = ['IC_%s' % c for c in range(1,138)]
columns_O = ['OA_%o' % o for o in range(1,7)]

columns_all = columns_A + columns_C + columns_O

# remove 'Number' column and reset columns' name of raw training data 
df_raw = df_raw.drop(['Number'], axis = 1)
df_raw.columns = columns_all
df_raw.info() 



# features_CT denote parts of "columns_C" in Text format
# from   C_015 to C_038   &   C_063 to C_082, total 44
features_CT = columns_C[14:38] + columns_C[62:82]

# features_CN denote parts of "columns_C" in Numerical format
features_CN = columns_C[0:14] + columns_C[38:62] + columns_C[82:137]


# set the key features given by official
features_key_A = ['IA1_20',
                  'IA2_16','IA2_17','IA2_24',
                  'IA3_13','IA3_15','IA3_16','IA3_17','IA3_18',                  
                  'IA6_1' ,'IA6_11','IA6_19','IA6_24']
features_key_C = ['IC_13','IC_46','IC_49','IC_50',
                  'IC_57','IC_58','IC_96']

features_key = features_key_A + features_key_C



# define the columns of group which exclude key features
columns_A1_r = [a for a in columns_A1 if a not in (['IA1_20'])]
columns_A2_r = [a for a in columns_A2 if a not in tuple(['IA2_16','IA2_17','IA2_24'])]
columns_A3_r = [a for a in columns_A3 if a not in tuple(['IA3_13','IA3_15','IA3_16','IA3_17','IA3_18'])]
columns_A6_r = [a for a in columns_A6 if a not in tuple(['IA6_1' ,'IA6_11','IA6_19','IA6_24'])]
features_CN_r = [c for c in features_CN if c not in tuple(features_key_C)]
columns_all_test = [a for a in columns_all if a not in tuple(features_key)]

# End of loading raw training data and setting columns' name
# -----------------------------------------------------------------------------






# -----------------------------------------------------------------------------
# convert text columns to numerical data 
df_text_features = pd.DataFrame(index = df_raw.index)

for c in features_CT:
    locals()['s_{}'.format(c)] = df_raw[c].str.split(";",expand = True).\
        rename(columns = {0: 'y_1', 1: 'y_2', 2: 'x_1', 3: 'x_2'})
    
    
    locals()['s_{}'.format(c)]['y'] = locals()['s_{}'.format(c)]['y_1']
    locals()['s_{}'.format(c)]['x'] = locals()['s_{}'.format(c)]['x_1']
    
    locals()['s_{}'.format(c)]['y'] = np.where(locals()['s_{}'.format(c)]['y_1'] == 'U',
                       locals()['s_{}'.format(c)]['y_2'].astype('float64'),
                       np.where(locals()['s_{}'.format(c)]['y_1'] == 'D',
                                - locals()['s_{}'.format(c)]['y_2'].astype('float64'),
                                 np.where(locals()['s_{}'.format(c)]['y_1'] == 'N', 0,
                                          locals()['s_{}'.format(c)]['y_1'])
                                 )
                       )
    
    locals()['s_{}'.format(c)]['x'] = np.where(locals()['s_{}'.format(c)]['x_1'] == 'R',
                           locals()['s_{}'.format(c)]['x_2'].astype('float64'),
                           np.where(locals()['s_{}'.format(c)]['x_1'] == 'L',
                                    - locals()['s_{}'.format(c)]['x_2'].astype('float64'),
                                     np.where(locals()['s_{}'.format(c)]['x_1'] == 'N', 0,
                                              locals()['s_{}'.format(c)]['x_1'])
                                     )
                           )
    
    
    locals()['s_{}'.format(c)]['y'] = locals()['s_{}'.format(c)]['y'].replace(-0,0)
    locals()['s_{}'.format(c)]['x'] = locals()['s_{}'.format(c)]['x'].replace(-0,0)
    locals()['s_{}'.format(c)] = locals()['s_{}'.format(c)][list('xy')]
    
    df_text_features = pd.merge(df_text_features,
                                locals()['s_{}'.format(c)],
                                right_index = True, left_index = True, how = 'left')
    
    
# reset columns' name
text_features_xy = ['IC_%s_x' % s for s in range(15,39,1)] +\
                   ['IC_%s_y' % s for s in range(15,39,1)] +\
                   ['IC_%s_x' % s for s in range(63,83,1)] +\
                   ['IC_%s_y' % s for s in range(63,83,1)] 
text_features_xy = sorted(text_features_xy)

df_text_features.columns = text_features_xy

df_features_C = pd.merge(df_raw[features_CN], df_text_features,
                         right_index = True, left_index = True, how = 'left')

# End of converting text columns to numerical data 
# -----------------------------------------------------------------------------






# -----------------------------------------------------------------------------
# reform raw training data


# Create a customized function to impute missing data for each input group
def impute_data(Dataframe, Columns, Impute_strategy = 'most_frequent'):
    """
    Impute missing values in dataframe by sklearn.impute.IterativeImputer
    
    Args:
        Dataframe: A pd.DataFrame which we want to impute its missing values
        Columns  : A list of spesific columns name which we want to create a subset of imputated dataframe    
        Impute_strategy: The imputaion method to impute missing value, default 'most_frequent'
    
    Return:
        df_imp: An imputed pd.DataFrame from input of this function.
        
    """
    
    df_mice = Dataframe[Columns]
    
    # set a large n_iter for base estimator and large max_iter for imputer
    imputer = IterativeImputer(estimator = BayesianRidge(n_iter = 1024),
                               max_iter = 666, initial_strategy = Impute_strategy)
    imputer.fit(df_mice.to_numpy())
    
    df_imp = pd.DataFrame(imputer.transform(df_mice.to_numpy()))
    df_imp.columns = df_mice.columns
    
    return df_imp
    # End of function


# impute missing data by "impute_data" for each input group
df_imp_A1 = impute_data(df_raw, columns_A1)
df_imp_A2 = impute_data(df_raw, columns_A2)
df_imp_A3 = impute_data(df_raw, columns_A3)
df_imp_A4 = impute_data(df_raw, columns_A4)
df_imp_A5 = impute_data(df_raw, columns_A5)
df_imp_A6 = impute_data(df_raw, columns_A6)
df_imp_C = impute_data(df_features_C, df_features_C.columns)


# merge all dataframes which has been imputed
#   the suffix of name of object 'df_imp_mode' denote that the imputed pd.DataFrame
#   was imputed by strategy of 'most_frequent' (Mode)
df_imp_mode = pd.merge(df_imp_A1, df_imp_A2, right_index = True, left_index = True, how = 'left')
df_imp_mode = pd.merge(df_imp_mode, df_imp_A3, right_index = True, left_index = True, how = 'left')
df_imp_mode = pd.merge(df_imp_mode, df_imp_A4, right_index = True, left_index = True, how = 'left')
df_imp_mode = pd.merge(df_imp_mode, df_imp_A5, right_index = True, left_index = True, how = 'left')
df_imp_mode = pd.merge(df_imp_mode, df_imp_A6, right_index = True, left_index = True, how = 'left')
df_imp_mode = pd.merge(df_imp_mode, df_imp_C, right_index = True, left_index = True, how = 'left')
df_imp_mode = pd.merge(df_imp_mode, df_raw[columns_O], right_index = True, left_index = True, how = 'left')


# create a customized funtion to generate rolling feature in pd.DataFrame
def create_rolling_feature(Dataframe, Columns, a = 4, b = 12, c = 8):
    """
    Impute missing values in dataframe by sklearn.impute.IterativeImputer
    
    Args:
        Dataframe  : A base pd.DataFrame which used to create rolling features
        Columns    : A list of spesific columns name which we want to create a subset of rolling features
        {a, b, c}  : Numbers of window size to create rolling features,
                    where {a,b} are used to compute both rolling mean and rolling standard deviation;                    
                    and {c} is used to measure range between max value and min vaue.
                    {a,b,c} defaut to {4,12,8}
    Return:
        df_rolling_feature: An pd.DataFrame from 1st input of this function.
        
    """        
    
    df_stage = Dataframe[Columns]  
        
    # get rolling mean with window size is a
    locals()['df_stage_avg_{}'.format(a)] = pd.DataFrame(index = df_stage.index)
    for d in Columns:
        locals()['df_stage_avg_{}'.format(a)][d] = round(df_stage[d].rolling(a).mean(), 3)        
    locals()['df_stage_avg_{}'.format(a)].columns = ['%s_avg_a' % s for s in Columns]
    
    
    # get rolling mean with window size is b
    locals()['df_stage_avg_{}'.format(b)] = pd.DataFrame(index = df_stage.index)
    for d in Columns:
        locals()['df_stage_avg_{}'.format(b)][d] = round(df_stage[d].rolling(b).mean(), 3)        
    locals()['df_stage_avg_{}'.format(b)].columns = ['%s_avg_b' % s for s in Columns]
    
    
    # get rolling std with window size is a
    locals()['df_stage_std_{}'.format(a)] = pd.DataFrame(index = df_stage.index)
    for d in Columns:
        locals()['df_stage_std_{}'.format(a)][d] = round(df_stage[d].rolling(a).std(), 3)        
    locals()['df_stage_std_{}'.format(a)].columns = ['%s_std_a' % s for s in Columns]
    
    
    # get rolling std with window size is b   
    locals()['df_stage_std_{}'.format(b)] = pd.DataFrame(index = df_stage.index)
    for d in Columns:
        locals()['df_stage_std_{}'.format(b)][d] = round(df_stage[d].rolling(b).std(), 3)     
    locals()['df_stage_std_{}'.format(b)].columns = ['%s_std_b' % s for s in Columns]
    
    
    # get rolling range with window size is c
    locals()['df_stage_range_{}'.format(c)] = pd.DataFrame(index = df_stage.index)
    for d in Columns:
        locals()['df_stage_range_{}'.format(c)][d] = round(df_stage[d].rolling(c).max() - df_stage[d].rolling(c).min(), 3)
    locals()['df_stage_range_{}'.format(c)].columns = ['%s_range_c' % s for s in Columns]
    
    
    
    # merge the above rolling features in one pd.DataFrame 
    df_mix = pd.merge(pd.DataFrame(index = df_stage.index), locals()['df_stage_avg_{}'.format(a)], 
                      right_index = True, left_index = True, how = 'left')
    
    df_mix = pd.merge(df_mix, locals()['df_stage_avg_{}'.format(b)], 
                      right_index = True, left_index = True, how = 'left')
    
    df_mix = pd.merge(df_mix, locals()['df_stage_std_{}'.format(a)], 
                      right_index = True, left_index = True, how = 'left')
    
    df_mix = pd.merge(df_mix, locals()['df_stage_std_{}'.format(b)], 
                      right_index = True, left_index = True, how = 'left')
    
    df_mix = pd.merge(df_mix, locals()['df_stage_range_{}'.format(c)], 
                      right_index = True, left_index = True, how = 'left')    
        
   
    # Use MICE algorithm to impute missing value in df_mix 
    # where imputation strategy is 'mean' 
    df_mice = df_mix.copy()
    imp_mean = IterativeImputer(estimator = BayesianRidge(n_iter = 1024),
                                max_iter = 666, initial_strategy = 'mean')
    imp_mean.fit(df_mice.to_numpy())
    df_rolling_feature = pd.DataFrame(imp_mean.transform(df_mice.to_numpy()))
    df_rolling_feature.columns = df_mice.columns
    
    return df_rolling_feature
    # End of function
# -----------------------------------------------------------------------------






# -----------------------------------------------------------------------------
# Feature Engineering and Training Models


# ----------------------------------------
# Start training process of features_key_C

# 1-1-1
# use [features_CN, columns_O] to create rolling features which exclude key features
df_rolling_nk_CN = create_rolling_feature(df_imp_mode, features_CN_r)
df_rolling_OA = create_rolling_feature(df_imp_mode, columns_O)


# 1-1-2
# create "df_base_c", which contain text_features_xy, and [features_CN, columns_O]
#    and thier rolling features (but without key features) to generate synthetic data of features_key_C
df_base_c = pd.merge(df_imp_mode[features_CN_r],
                     df_rolling_nk_CN, right_index = True, left_index = True, how = 'left')

df_base_c = pd.merge(df_base_c, df_imp_mode[text_features_xy],
                     right_index = True, left_index = True, how = 'left')

df_base_c = pd.merge(df_base_c, df_imp_mode[columns_O],
                     right_index = True, left_index = True, how = 'left')

df_base_c = pd.merge(df_base_c, df_rolling_OA,
                     right_index = True, left_index = True, how = 'left')


# 1-1-3
# set 4 base Regressors, VotingRegressor, and MultiOutputRegressor to generate synthetic data
# All parameters of the following regressors have benn tuned manually
reg_c_xgb = XGBRegressor(max_depth = 3, n_estimators = 169,
                         reg_lambda = 3, n_jobs = -1, verbosity = 0, random_state = 2020)

reg_c_lgb = LGBMRegressor(max_depth = 13, n_estimators = 64,
                          reg_lambda = 3, n_jobs = -1, random_state = 2020)

reg_c_ada = AdaBoostRegressor(n_estimators = 196, learning_rate = 0.01,
                              loss = 'exponential', random_state = 2020) 

reg_c_rf = RandomForestRegressor(n_estimators = 169, max_depth = 4, 
                                 n_jobs = -1, random_state = 2020)

# create an integrated regressor which combine from [reg_c_xgb, reg_c_lgb, reg_c_ada, reg_c_rf]
#   to generate synthetic data and predic final results
reg_c_vot = VotingRegressor(estimators = [('xgb', reg_c_xgb),
                                          ('lgb', reg_c_lgb),
                                          ('ada', reg_c_ada),
                                          ('lr' , reg_c_rf )], n_jobs = -1)

# create a multi-output regressor based on 'reg_vot' for df_base_c
#   to generate synthetic data of features_key_C at the same time
reg_c_mop = MultiOutputRegressor(reg_c_vot, n_jobs = -1)
reg_c_mop.fit(csr_matrix(df_base_c), 
              df_imp_mode[features_key_C].to_numpy() )
df_syn_key_C = pd.DataFrame(reg_c_mop.predict(csr_matrix(df_base_c)),
                            columns = features_key_C)


# 1-1-4
# fit series of reg_c_vot_{} of features_key_C 
for c in features_key_C:
    
    # drop one of feature from features_key_C, which is the objective 
    df_syn_key_C_stage = df_syn_key_C.drop([c], axis = 1)
    
    # create rolling features from df_syn_key_C_stage
    df_rolling_syn_key_C_stage = create_rolling_feature(df_syn_key_C_stage,
                                                        df_syn_key_C_stage.columns)
    
    # merge synthetic data and its rolling features with df_base_c
    df_base_c_plus = pd.merge(df_syn_key_C_stage, df_rolling_syn_key_C_stage,
                              right_index = True, left_index = True, how = 'left')
    
    df_base_c_plus = pd.merge(df_base_c_plus, df_base_c,
                              right_index = True, left_index = True, how = 'left')
    
    # set series of reg_c_vot_{} in every iteration
    locals()['reg_c_vot_{}'.format(c)] = VotingRegressor(estimators = [('xgb', reg_c_xgb),
                                                                       ('lgb', reg_c_lgb),
                                                                       ('ada', reg_c_ada),
                                                                       ('lr' , reg_c_rf)], n_jobs = -1)
    locals()['reg_c_vot_{}'.format(c)].fit(csr_matrix(df_base_c_plus), df_imp_mode[c])
    
    # End of loop

# End training process of features_key_C
# ----------------------------------------






# ----------------------------------------
# Start training process of ['IA1_20']

# 1-2-1
# use [['OA_1'],  columns_A1_r] to create rolling features which exclude key feaures
df_rolling_IA1 = create_rolling_feature(df_imp_mode, columns_A1_r)
df_rolling_OA1 = create_rolling_feature(df_imp_mode, ['OA_1'])


# 1-2-2
# create df_base_A1, which contain ['OA_1'], df_rolling_OA1, columns_A1_r, 
# df_rolling_IA1, and text_features_xy to generate synthetic data of ['IA1_20']

df_base_A1 = pd.merge(df_imp_mode[['OA_1']],
                      df_rolling_OA1, 
                      right_index = True, left_index = True, how = 'left')

df_base_A1 = pd.merge(df_base_A1 ,
                      df_rolling_IA1,
                      right_index = True, left_index = True, how = 'left')

df_base_A1 = pd.merge(df_base_A1 ,
                      df_imp_mode[columns_A1_r],
                      right_index = True, left_index = True, how = 'left')

df_base_A1 = pd.merge(df_base_A1 ,
                      df_imp_mode[text_features_xy],
                      right_index = True, left_index = True, how = 'left')


# 1-2-3
# set 4 base Regressors, VotingRegressor, and MultiOutputRegressor to generate synthetic data
#   All parameters of the following regressors have benn tuned manually
reg_a1_xgb_1 = XGBRegressor(max_depth = 1, n_estimators = 196,
                            reg_lambda = 3, n_jobs = -1, verbosity = 0, random_state = 2020)

reg_a1_lgb_1 = LGBMRegressor(max_depth = 1, n_estimators = 196, 
                             reg_lambda = 5, n_jobs = -1, random_state = 2020)

reg_a1_ada_1 = AdaBoostRegressor(n_estimators = 225, learning_rate = 0.01,
                                 loss = 'square', random_state = 2020) 

reg_a1_rf_1 = RandomForestRegressor(n_estimators = 196, max_depth = 1, 
                                    n_jobs = -1, random_state = 2020)

reg_a1_vot_1 = VotingRegressor(estimators = [('xgb', reg_a1_xgb_1),
                                             ('lgb', reg_a1_lgb_1),
                                             ('ada', reg_a1_ada_1),
                                             ('rf' , reg_a1_rf_1)],
                               weights = [2,1,1,1], n_jobs = -1)

# 1-2-4
# fit and generate synthetic data of ['IA1_20'] by reg_a1_vot_1
reg_a1_xgb_1.fit(csr_matrix(df_base_A1), df_imp_mode['IA1_20'].to_numpy() )
reg_a1_lgb_1.fit(csr_matrix(df_base_A1), df_imp_mode['IA1_20'].to_numpy() )
reg_a1_ada_1.fit(csr_matrix(df_base_A1), df_imp_mode['IA1_20'].to_numpy() )
reg_a1_rf_1.fit(csr_matrix(df_base_A1), df_imp_mode['IA1_20'].to_numpy() )

stk_a1_xgb = reg_a1_xgb_1.predict(csr_matrix(df_base_A1))
stk_a1_lgb = reg_a1_lgb_1.predict(csr_matrix(df_base_A1))
stk_a1_ada = reg_a1_ada_1.predict(csr_matrix(df_base_A1))
stk_a1_rf = reg_a1_rf_1.predict(csr_matrix(df_base_A1))

df_a1_stk = pd.DataFrame({'xgb': stk_a1_xgb,
                          'lgb': stk_a1_lgb,
                          'ada': stk_a1_ada,
                          'rf' : stk_a1_rf})

# 1-2-5
# use stacking learning method (but without cross-validation) to predict final 'IA1_20'
#   All parameters of the following regressors have benn tuned manually
reg_a1_xgb_2 = XGBRegressor(max_depth = 3, n_estimators = 144,
                            reg_lambda = 3, n_jobs = -1, verbosity = 0, random_state = 2020)

reg_a1_lgb_2 = LGBMRegressor(max_depth = 1, n_estimators = 256, 
                             reg_lambda = 5, n_jobs = -1, random_state = 2020)

reg_a1_ada_2 = AdaBoostRegressor(n_estimators = 256, learning_rate = 0.01,
                                  loss = 'exponential', random_state = 2020) 

reg_a1_rf_2 = RandomForestRegressor(n_estimators = 289, max_depth = 3, 
                                    n_jobs = -1, random_state = 2020)

reg_a1_vot_2 = VotingRegressor(estimators = [('xgb', reg_a1_xgb_2),
                                             ('lgb', reg_a1_lgb_2),
                                             ('ada', reg_a1_ada_2),
                                             ('rf' , reg_a1_rf_2)],
                               weights = [2,1,1,1], n_jobs = -1)

reg_a1_vot_2.fit(csr_matrix(df_a1_stk), df_imp_mode['IA1_20'].to_numpy() )

# End training process of ['IA1_20']
# ----------------------------------------






# ----------------------------------------
# Start training process of ['IA2_16','IA2_17','IA2_24']

# 1-3-1
# use [['OA_2'], columns_A2_r] to create rolling features which exclude key features
df_rolling_IA2 = create_rolling_feature(df_imp_mode, columns_A2_r )
df_rolling_OA2 = create_rolling_feature(df_imp_mode, ['OA_2'])


# 1-3-2
# create df_base_A2: which contain ['OA_2'], df_rolling_OA2, columns_A2_r, 
#   df_rolling_IA2, features_CN (features_CN_r + [features_key_C), and 
#   text_features_xy to generate synthetic data of ['IA2_16','IA2_17','IA2_24']

df_base_A2 = pd.merge(df_imp_mode[['OA_2']],
                      df_rolling_OA2, right_index = True, left_index = True, how = 'left')

df_base_A2 = pd.merge(df_base_A2 ,
                      df_rolling_IA2, right_index = True, left_index = True, how = 'left')

df_base_A2 = pd.merge(df_base_A2 ,
                      df_imp_mode[columns_A2_r], right_index = True, left_index = True, how = 'left')

df_base_A2 = pd.merge(df_base_A2 ,
                      df_imp_mode[text_features_xy], right_index = True, left_index = True, how = 'left')

df_base_A2 = pd.merge(df_base_A2 ,
                      df_imp_mode[features_CN_r], right_index = True, left_index = True, how = 'left')

df_base_A2 = pd.merge(df_base_A2 ,
                      df_imp_mode[features_key_C], right_index = True, left_index = True, how = 'left')


# 1-3-3
# set 4 base Regressors, VotingRegressor, and MultiOutputRegressor to generate synthetic data
#   All parameters of the following regressors have benn tuned manually
reg_a2_xgb_1 = XGBRegressor(max_depth = 1, n_estimators = 225,
                            reg_lambda = 3, n_jobs = -1, verbosity = 0, random_state = 2020)

reg_a2_lgb_1 = LGBMRegressor(max_depth = 13, n_estimators = 256, 
                             reg_lambda = 5, n_jobs = -1, random_state = 2020)

reg_a2_ada_1 = AdaBoostRegressor(n_estimators = 256, learning_rate = 0.01,
                                 loss = 'square', random_state = 2020) 

reg_a2_rf_1 = RandomForestRegressor(max_depth = 13, n_estimators = 256, 
                                    n_jobs = -1, random_state = 2020)

reg_a2_vot_1 = VotingRegressor(estimators = [('xgb', reg_a2_xgb_1),
                                             ('lgb', reg_a2_lgb_1),
                                             ('ada', reg_a2_ada_1),
                                             ('rf' , reg_a2_rf_1)], 
                               weights = [2,1,1,1], n_jobs = -1)

# createa multi-output regressor based on 'reg_a2_vot_1' for df_base_A2
#   to generate synthetic data of features_key_C at the same time
reg_a2_mop = MultiOutputRegressor(reg_a2_vot_1, n_jobs = -1)
reg_a2_mop.fit(csr_matrix(df_base_A2), 
               df_imp_mode[['IA2_16','IA2_17','IA2_24']].to_numpy() )
df_syn_key_A2 = pd.DataFrame(reg_a2_mop.predict(csr_matrix(df_base_A2)),
                             columns = ['IA2_16','IA2_17','IA2_24'])


# 1-3-4
# fit series of reg_a2_vot_{} of ['IA2_16','IA2_17','IA2_24']
for a in ['IA2_16','IA2_17','IA2_24']:
    
    # drop one of feature from ['IA2_16','IA2_17','IA2_24'], which is the objective 
    df_syn_key_A2_stage = df_syn_key_A2.drop([a], axis = 1)
    
    # create rolling features from df_syn_key_C_stage
    df_rolling_syn_key_A2_stage = create_rolling_feature(df_syn_key_A2_stage,
                                                         df_syn_key_A2_stage.columns)
    
    # merge synthetic data and its rolling features with df_base_A2
    df_base_A2_plus = pd.merge(df_syn_key_A2_stage, df_rolling_syn_key_A2_stage,
                              right_index = True, left_index = True, how = 'left')
    
    df_base_A2_plus = pd.merge(df_base_A2_plus, df_base_A2,
                              right_index = True, left_index = True, how = 'left')
        
    # set series of reg_a2_vot_{} in every iteration
    locals()['reg_a2_vot_{}'.format(a)] = VotingRegressor(estimators = [('xgb', reg_a2_xgb_1),
                                                                        ('lgb', reg_a2_lgb_1),
                                                                        ('ada', reg_a2_ada_1),
                                                                        ('rf' , reg_a2_rf_1)], 
                                                          weights = [2,1,1,1], n_jobs = -1)
    locals()['reg_a2_vot_{}'.format(a)].fit(csr_matrix(df_base_A2_plus), df_imp_mode[a])
    
    # End of loop


# End training process of ['IA2_16','IA2_17','IA2_24']
# ----------------------------------------






# ----------------------------------------
# Start training process of ['IA3_13','IA3_15','IA3_16','IA3_17','IA3_18']

# 1-4-1
# use [['OA_3'], columns_A3_r]
#   to create rolling features which exclude key features
df_rolling_IA3 = create_rolling_feature(df_imp_mode, columns_A3_r)
df_rolling_OA3 = create_rolling_feature(df_imp_mode, ['OA_3'])


# 1-4-2
# create df_base_A3: which contain ['OA_3'], df_rolling_OA3, columns_A3_r, 
#   df_rolling_IA3, and text_features_xy to generate synthetic data of ['IA3_13','IA3_15','IA3_16','IA3_17','IA3_18']

df_base_A3 = pd.merge(df_imp_mode[['OA_3']],
                      df_rolling_OA3, right_index = True, left_index = True, how = 'left')

df_base_A3 = pd.merge(df_base_A3 ,
                      df_rolling_IA3, right_index = True, left_index = True, how = 'left')

df_base_A3 = pd.merge(df_base_A3 ,
                      df_imp_mode[columns_A3_r], right_index = True, left_index = True, how = 'left')

df_base_A3 = pd.merge(df_base_A3 ,
                      df_imp_mode[text_features_xy], right_index = True, left_index = True, how = 'left')


# 1-4-3
# set 4 base Regressors, VotingRegressor, and MultiOutputRegressor to generate synthetic data
#   All parameters of the following regressors have benn tuned manually
reg_a3_xgb_1 = XGBRegressor(max_depth = 1, n_estimators = 144,
                            reg_lambda = 3, n_jobs = -1, verbosity = 0, random_state = 2020)

reg_a3_lgb_1 = LGBMRegressor(max_depth = 1, n_estimators = 99, 
                              reg_lambda = 5, n_jobs = -1, random_state = 2020)

reg_a3_ada_1 = AdaBoostRegressor(n_estimators = 144, learning_rate = 0.01,
                                  loss = 'square', random_state = 2020) 

reg_a3_rf_1 = RandomForestRegressor(max_depth = 7, n_estimators = 256, 
                                    n_jobs = -1, random_state = 2020)

reg_a3_vot_1 = VotingRegressor(estimators = [('xgb', reg_a3_xgb_1),
                                             ('lgb', reg_a3_lgb_1),
                                             ('ada', reg_a3_ada_1),
                                             ('rf' , reg_a3_rf_1)], 
                               weights = [2,1,2,1], n_jobs = -1)

# createa multi-output regressor based on 'reg_a3_mop' for df_base_A3
#   to generate synthetic data of ['IA3_13','IA3_15','IA3_16','IA3_17','IA3_18'] at the same time
reg_a3_mop = MultiOutputRegressor(reg_a3_vot_1, n_jobs = -1)
reg_a3_mop.fit(csr_matrix(df_base_A3),                
               df_imp_mode[['IA3_13','IA3_15','IA3_16','IA3_17','IA3_18']].to_numpy() )
df_syn_key_A3 = pd.DataFrame(reg_a3_mop.predict(csr_matrix(df_base_A3)),
                             columns =['IA3_13','IA3_15','IA3_16','IA3_17','IA3_18'])


# 1-4-4
# fit series of reg_a3_vot_{} of ['IA3_13','IA3_15','IA3_16','IA3_17','IA3_18']
for a in ['IA3_13','IA3_15','IA3_16','IA3_17','IA3_18']:
    
    # drop one of feature from ['IA3_13','IA3_15','IA3_16','IA3_17','IA3_18'], which is the objective
    df_syn_key_A3_stage = df_syn_key_A3.drop([a], axis = 1)
    
    # create rolling features from ['IA3_13','IA3_15','IA3_16','IA3_17','IA3_18']
    df_rolling_syn_key_A3_stage = create_rolling_feature(df_syn_key_A3_stage,
                                                         df_syn_key_A3_stage.columns)
    
    # merge synthetic data and its rolling features with df_base_A3
    df_base_A3_plus = pd.merge(df_syn_key_A3_stage, df_rolling_syn_key_A3_stage,
                              right_index = True, left_index = True, how = 'left')
    
    df_base_A3_plus = pd.merge(df_base_A3_plus, df_base_A3,
                              right_index = True, left_index = True, how = 'left')
    
    # set series of reg_a3_vot_{} in every iteration
    locals()['reg_a3_vot_{}'.format(a)] = VotingRegressor(estimators = [('xgb', reg_a3_xgb_1),
                                                                        ('lgb', reg_a3_lgb_1),
                                                                        ('ada', reg_a3_ada_1),
                                                                        ('rf' , reg_a3_rf_1)], 
                                                          weights = [2,1,2,1], n_jobs = -1)
    locals()['reg_a3_vot_{}'.format(a)].fit(csr_matrix(df_base_A3_plus), df_imp_mode[a])
    
    # End of loop

# End training process of ['IA3_13','IA3_15','IA3_16','IA3_17','IA3_18']
# ----------------------------------------






# ----------------------------------------
# Start training process of ['IA6_1' ,'IA6_11','IA6_19','IA6_24']

# 1-5-1
# use [['OA_6'], columns_A6_r]
# to create rolling features which exclude key features
df_rolling_IA6 = create_rolling_feature(df_imp_mode, columns_A6_r )
df_rolling_OA6 = create_rolling_feature(df_imp_mode, ['OA_6'])


# 1-5-2
# create df_base_A6: which contain ['OA_6'], df_rolling_OA6, columns_A6_r, 
#   df_rolling_IA6, features_CN (features_CN_r + [features_key_C), 
#   and text_features_xy to generate synthetic data of ['IA6_1' ,'IA6_11','IA6_19','IA6_24']

df_base_A6 = pd.merge(df_imp_mode[['OA_6']],
                      df_rolling_OA6, right_index = True, left_index = True, how = 'left')

df_base_A6 = pd.merge(df_base_A6 ,
                      df_rolling_IA6, right_index = True, left_index = True, how = 'left')

df_base_A6 = pd.merge(df_base_A6 ,
                      df_imp_mode[columns_A6_r], right_index = True, left_index = True, how = 'left')

df_base_A6 = pd.merge(df_base_A6 ,
                      df_imp_mode[text_features_xy], right_index = True, left_index = True, how = 'left')

df_base_A6 = pd.merge(df_base_A6 ,
                      df_imp_mode[features_CN_r], right_index = True, left_index = True, how = 'left')

df_base_A6 = pd.merge(df_base_A6 ,
                      df_imp_mode[features_key_C],right_index = True, left_index = True, how = 'left')


# 1-5-3
# set 4 base Regressors, VotingRegressor, and MultiOutputRegressor to generate synthetic data
#   All parameters of the following regressors have benn tuned manually
reg_a6_xgb_1 = XGBRegressor(max_depth = 2, n_estimators = 72,
                            reg_lambda = 3, n_jobs = -1, verbosity = 0, random_state = 2020)

reg_a6_lgb_1 = LGBMRegressor(max_depth = 13, n_estimators = 64, 
                              reg_lambda = 3, n_jobs = -1, random_state = 2020)

reg_a6_ada_1 = AdaBoostRegressor(n_estimators = 361, learning_rate = 0.01,
                                  loss = 'linear', random_state = 2020) 

reg_a6_rf_1 = RandomForestRegressor(max_depth = 3, n_estimators = 256, 
                                    n_jobs = -1, random_state = 2020)

reg_a6_vot_1 = VotingRegressor(estimators = [('xgb', reg_a6_xgb_1),
                                             ('lgb', reg_a6_lgb_1),
                                             ('ada', reg_a6_ada_1),
                                             ('rf' , reg_a6_rf_1)], 
                               weights = [2,1,2,2], n_jobs = -1)

# createa multi-output regressor based on 'reg_a6_mop' for df_base_A6
#   to generate synthetic data of ['IA6_1' ,'IA6_11','IA6_19','IA6_24'] at the same time
reg_a6_mop = MultiOutputRegressor(reg_a6_vot_1, n_jobs = -1)
reg_a6_mop.fit(csr_matrix(df_base_A6), 
               df_imp_mode[['IA6_1' ,'IA6_11','IA6_19','IA6_24']].to_numpy() )
df_syn_key_A6 = pd.DataFrame(reg_a6_mop.predict(csr_matrix(df_base_A6)),
                             columns = ['IA6_1' ,'IA6_11','IA6_19','IA6_24'])


# 1-5-4
# fit series of reg_a6_vot_{} of ['IA6_1' ,'IA6_11','IA6_19','IA6_24']
for a in ['IA6_1' ,'IA6_11','IA6_19','IA6_24']:
    
    # drop one of feature from ['IA6_1' ,'IA6_11','IA6_19','IA6_24'] , which is the objective 
    df_syn_key_A6_stage = df_syn_key_A6.drop([a], axis = 1)
    
    # create rolling features from ['IA6_1' ,'IA6_11','IA6_19','IA6_24'] 
    df_rolling_syn_key_A6_stage = create_rolling_feature(df_syn_key_A6_stage,
                                                         df_syn_key_A6_stage.columns)
    
    # merge synthetic data and its rolling features with df_base_A6
    df_base_A6_plus = pd.merge(df_syn_key_A6_stage, df_rolling_syn_key_A6_stage,
                              right_index = True, left_index = True, how = 'left')
    
    df_base_A6_plus = pd.merge(df_base_A6_plus, df_base_A6,
                              right_index = True, left_index = True, how = 'left')    
        
    # set series of reg_a6_vot_{} in every iteration
    locals()['reg_a6_vot_{}'.format(a)] = VotingRegressor(estimators = [('xgb', reg_a6_xgb_1),
                                                                        ('lgb', reg_a6_lgb_1),
                                                                        ('ada', reg_a6_ada_1),
                                                                        ('rf' , reg_a6_rf_1)], 
                                                          weights = [2,1,2,2], n_jobs = -1)
    locals()['reg_a6_vot_{}'.format(a)].fit(csr_matrix(df_base_A6_plus), df_imp_mode[a])
    
    # End of loop

# End training process of ['IA6_1' ,'IA6_11','IA6_19','IA6_24']
# ----------------------------------------

# End of phrase of Feature Engineering and Training Models
# -----------------------------------------------------------------------------










# -----------------------------------------------------------------------------
# Data Pre-processing for testing data

# loading testing data
df_raw_test = pd.read_csv(r'C:\Users\123\Dropbox\Data Science\IMBD2020\0728test.csv').drop(['Number'], axis = 1)

# rename the columns' name of df_raw_test
df_raw_test.columns = columns_all_test



# -------------------------------------------------------
# convert text columns to numerical data for testing data
df_text_features_test = pd.DataFrame(index = df_raw_test.index)

for c in features_CT:    
    locals()['s_{}'.format(c)] = df_raw_test[c].str.split(";",expand = True).\
        rename(columns = {0: 'y_1', 1: 'y_2', 2: 'x_1', 3: 'x_2'})
    
    
    locals()['s_{}'.format(c)]['y'] = locals()['s_{}'.format(c)]['y_1']
    locals()['s_{}'.format(c)]['x'] = locals()['s_{}'.format(c)]['x_1']
    
    locals()['s_{}'.format(c)]['y'] = np.where(locals()['s_{}'.format(c)]['y_1'] == 'U',
                       locals()['s_{}'.format(c)]['y_2'].astype('float64'),
                       np.where(locals()['s_{}'.format(c)]['y_1'] == 'D',
                                - locals()['s_{}'.format(c)]['y_2'].astype('float64'),
                                 np.where(locals()['s_{}'.format(c)]['y_1'] == 'N', 0,
                                          locals()['s_{}'.format(c)]['y_1'])
                                 )
                       )
    
    locals()['s_{}'.format(c)]['x'] = np.where(locals()['s_{}'.format(c)]['x_1'] == 'R',
                           locals()['s_{}'.format(c)]['x_2'].astype('float64'),
                           np.where(locals()['s_{}'.format(c)]['x_1'] == 'L',
                                    - locals()['s_{}'.format(c)]['x_2'].astype('float64'),
                                     np.where(locals()['s_{}'.format(c)]['x_1'] == 'N', 0,
                                              locals()['s_{}'.format(c)]['x_1'])
                                     )
                           )
    
    
    locals()['s_{}'.format(c)]['y'] = locals()['s_{}'.format(c)]['y'].replace(-0,0)
    locals()['s_{}'.format(c)]['x'] = locals()['s_{}'.format(c)]['x'].replace(-0,0)
    locals()['s_{}'.format(c)] = locals()['s_{}'.format(c)][list('xy')]
    
    df_text_features_test = pd.merge(df_text_features_test, locals()['s_{}'.format(c)],
                                right_index = True, left_index = True,
                                how = 'left')


df_text_features_test.columns = text_features_xy
df_features_C_test = pd.merge(df_raw_test[features_CN_r], df_text_features_test,
                              right_index = True, left_index = True, how = 'left')

# End of converting text columns to numerical data  for testing data
# -----------------------------------------------------------------------------




# -----------------------------------------------------------------------------
# reform raw data of test
# impute missing data by "impute_data" for each input group
df_imp_A1_test = impute_data(df_raw_test, columns_A1_r)
df_imp_A2_test = impute_data(df_raw_test, columns_A2_r)
df_imp_A3_test = impute_data(df_raw_test, columns_A3_r)
df_imp_A4_test = impute_data(df_raw_test, columns_A4)
df_imp_A5_test = impute_data(df_raw_test, columns_A5)
df_imp_A6_test = impute_data(df_raw_test, columns_A6_r)
df_imp_C_test = impute_data(df_features_C_test, df_features_C_test.columns)


# merge all dataframes which has been imputed
#   the suffix of name of object 'df_imp_mode' denote that the imputed pd.DataFrame
#   was imputed by strategy of 'most_frequent' (Mode)
df_imp_mode_test = pd.merge(df_imp_A1_test, df_imp_A2_test, right_index = True, left_index = True, how = 'left')
df_imp_mode_test = pd.merge(df_imp_mode_test, df_imp_A3_test, right_index = True, left_index = True, how = 'left')
df_imp_mode_test = pd.merge(df_imp_mode_test, df_imp_A4_test, right_index = True, left_index = True, how = 'left')
df_imp_mode_test = pd.merge(df_imp_mode_test, df_imp_A5_test, right_index = True, left_index = True, how = 'left')
df_imp_mode_test = pd.merge(df_imp_mode_test, df_imp_A6_test, right_index = True, left_index = True, how = 'left')
df_imp_mode_test = pd.merge(df_imp_mode_test, df_imp_C_test, right_index = True, left_index = True, how = 'left')
df_imp_mode_test = pd.merge(df_imp_mode_test, df_raw_test[columns_O], right_index = True, left_index = True, how = 'left')





# -----------------------------------------------------------------------------
# Feature Engineering and Predict Final Results by prepared Models

# ----------------------------------------
# Start predicting process of features_key_C

# 2-1-1
# use [features_CN_r, columns_O] to create rolling features which exclude key features
df_rolling_nk_CN_test = create_rolling_feature(df_imp_mode_test, features_CN_r)
df_rolling_OA_test = create_rolling_feature(df_imp_mode_test, columns_O)


# 2-1-2
# create df_base_c_test to generate synthetic data of features_key_C
df_base_c_test = pd.merge(df_imp_mode_test[features_CN_r],
                          df_rolling_nk_CN_test, 
                          right_index = True, left_index = True, how = 'left')

df_base_c_test = pd.merge(df_base_c_test, 
                          df_imp_mode_test[text_features_xy],
                          right_index = True, left_index = True, how = 'left')

df_base_c_test = pd.merge(df_base_c_test, 
                          df_imp_mode_test[columns_O],
                          right_index = True, left_index = True, how = 'left')

df_base_c_test = pd.merge(df_base_c_test, 
                          df_rolling_OA_test,
                          right_index = True, left_index = True, how = 'left')


# 2-1-3
# use fitted reg_c_mop to generate synthetic data: df_syn_key_C_test
df_syn_key_C_test = pd.DataFrame(reg_c_mop.predict(csr_matrix(df_base_c_test)),
                                 columns = features_key_C)

df_key_C_final = pd.DataFrame(index = df_imp_mode_test.index)
# use a for loop to predict final results in a pd.DataFrame
for c in features_key_C:
    
    # drop one of feature from features_key_C, which is the objective in this iteration
    df_syn_key_C_stage = df_syn_key_C_test.drop([c], axis = 1)
    
    
    # create rolling features from df_syn_key_C_stage
    df_rolling_syn_key_C_stage = create_rolling_feature(df_syn_key_C_stage, df_syn_key_C_stage.columns)
    
    # merge synthetic data and its rolling features with df_base_c_test
    df_base_c_plus = pd.merge(df_syn_key_C_stage, df_rolling_syn_key_C_stage,
                              right_index = True, left_index = True, how = 'left')
    
    df_base_c_plus = pd.merge(df_base_c_plus, df_base_c_test,
                              right_index = True, left_index = True, how = 'left')
    
    
    # use series of reg_c_vot_{} and synthetic data to predict individual final resault of features_key_C
    locals()['{}_final'.format(c)] = pd.DataFrame(locals()['reg_c_vot_{}'.format(c)].predict(csr_matrix(df_base_c_plus)),
                                                  columns = [c])
    
    
    df_key_C_final = pd.merge(df_key_C_final, locals()['{}_final'.format(c)],
                              right_index = True, left_index = True, how = 'left')
    
    # End of loop
    
# End predicting process of features_key_C
# Get final prediction of features_key_C: df_key_C_final 
# -----------------------------------------------------------------------------






# -----------------------------------------------------------------------------
# Start predicting process of features_key_C 'IA1_20'

# 2-2-1
# use [['OA_2'],  columns_A1_r ] to create rolling features which exclude key features
df_rolling_IA1_test = create_rolling_feature(df_imp_mode_test, columns_A1_r)
df_rolling_OA1_test = create_rolling_feature(df_imp_mode_test, ['OA_1'])


# 2-2-2
# create df_base_A1_test to generate synthetic data of ['IA1_20']

df_base_A1_test = pd.merge(df_imp_mode_test[['OA_1']],
                           df_rolling_OA1_test, right_index = True, left_index = True, how = 'left')

df_base_A1_test = pd.merge(df_base_A1_test, 
                           df_rolling_IA1_test, right_index = True, left_index = True, how = 'left')

df_base_A1_test = pd.merge(df_base_A1_test,
                           df_imp_mode_test[columns_A1_r], right_index = True, left_index = True, how = 'left')

df_base_A1_test = pd.merge(df_base_A1_test, 
                           df_imp_mode_test[text_features_xy], right_index = True, left_index = True, how = 'left')


# 2-2-3
# fit and generate synthetic data of ['IA1_20'] by reg_a1_vot_1

stk_a1_xgb_test = reg_a1_xgb_1.predict(csr_matrix(df_base_A1_test))
                                   
stk_a1_lgb_test = reg_a1_lgb_1.predict(csr_matrix(df_base_A1_test))

stk_a1_ada_test = reg_a1_ada_1.predict(csr_matrix(df_base_A1_test))

stk_a1_rf_test = reg_a1_rf_1.predict(csr_matrix(df_base_A1_test))

df_a1_stk_test = pd.DataFrame({'xgb': stk_a1_xgb_test,
                               'lgb': stk_a1_lgb_test,
                               'ada': stk_a1_ada_test,
                               'rf' : stk_a1_rf_test})

# predict final results in a pd.DataFrame
df_key_A1_final = pd.DataFrame(reg_a1_vot_2.predict(csr_matrix(df_a1_stk_test)),
                               columns = ['IA1_20'])

# End predicting process of ['IA1_20']
# Get final prediction of features_key_A1: df_key_A1_final
# -----------------------------------------------------------------------------






# -----------------------------------------------------------------------------
# Start predicting process of ['IA2_16','IA2_17','IA2_24']

# 2-3-1
# use [['OA_1'],  columns_A2_r] to create rolling features which exclude key features
df_rolling_IA2_test = create_rolling_feature(df_imp_mode_test, columns_A2_r)
df_rolling_OA2_test = create_rolling_feature(df_imp_mode_test, ['OA_2'])


# 2-3-2
# create df_base_A2_test to generate synthetic data of ['IA2_16','IA2_17','IA2_24']
#   Cause we use whole 'features_CN' in df_base_A2 in training phrase,
#   so, in testing phrase, we have to combine 'df_key_C_final' and 'features_CN_r'
#   when merging df_base_A2_test and make sure that the order of columns of 
#   df_base_A2 and df_base_A2_test are same
df_base_A2_test = pd.merge(df_imp_mode_test[['OA_2']],
                           df_rolling_OA2_test, 
                           right_index = True, left_index = True, how = 'left')

df_base_A2_test = pd.merge(df_base_A2_test ,
                           df_rolling_IA2_test,
                           right_index = True, left_index = True, how = 'left')

df_base_A2_test = pd.merge(df_base_A2_test ,
                           df_imp_mode_test[columns_A2_r],
                           right_index = True, left_index = True, how = 'left')

df_base_A2_test = pd.merge(df_base_A2_test ,
                           df_imp_mode_test[text_features_xy],
                           right_index = True, left_index = True, how = 'left')

df_base_A2_test = pd.merge(df_base_A2_test ,
                           df_imp_mode_test[features_CN_r],
                           right_index = True, left_index = True, how = 'left')

df_base_A2_test = pd.merge(df_base_A2_test ,
                           df_key_C_final,
                           right_index = True, left_index = True, how = 'left')

# The following row of code should be return 'True', or there may be something wrong
list(df_base_A2_test.columns) == list(df_base_A2.columns)



# 2-3-3
# fit and generate synthetic data of ['IA2_16','IA2_17','IA2_24'] by reg_a2_mop_test
df_syn_key_A2_test = pd.DataFrame(reg_a2_mop.predict(csr_matrix(df_base_A2_test)),
                                  columns = ['IA2_16','IA2_17','IA2_24'])


df_key_A2_final = pd.DataFrame(index = df_imp_mode_test.index)
# use fitted reg_c_mop to generate synthetic data: df_key_A2_final
for a in ['IA2_16','IA2_17','IA2_24']:
    
    # drop one of feature from ['IA2_16','IA2_17','IA2_24'], which is the objective in this iteration
    df_syn_key_A2_stage = df_syn_key_A2_test.drop([a], axis = 1)
    
    # create rolling features from df_syn_key_A2_stage
    df_rolling_syn_key_A2_stage = create_rolling_feature(df_syn_key_A2_stage,
                                                         df_syn_key_A2_stage.columns)
    
    # merge synthetic data and its rolling features with df_base_A2_test
    df_base_A2_plus = pd.merge(df_syn_key_A2_stage, 
                               df_rolling_syn_key_A2_stage,
                               right_index = True, left_index = True, how = 'left')
    
    df_base_A2_plus = pd.merge(df_base_A2_plus,
                               df_base_A2_test,
                               right_index = True, left_index = True, how = 'left')
    
    
    # use series of reg_a2_vot_{} to predict individual final resault of ['IA2_16','IA2_17','IA2_24']
    locals()['{}_final'.format(a)] = pd.DataFrame(locals()['reg_a2_vot_{}'.format(a)].predict(csr_matrix(df_base_A2_plus) ),
                                                  columns = [a])
    
    df_key_A2_final = pd.merge(df_key_A2_final, locals()['{}_final'.format(a)],
                               right_index = True, left_index = True, how = 'left')
    
    # End of loop

# End predicting process of ['IA2_16','IA2_17','IA2_24']
# Get final prediction of features_key_A2: df_key_A2_final
# -----------------------------------------------------------------------------






# -----------------------------------------------------------------------------
# Start predicting process of ['IA3_13','IA3_15','IA3_16','IA3_17','IA3_18']

# 2-4-1
# use [['OA_3'], columns_A3_r]
# to create rolling features which exclude key features
df_rolling_IA3_test = create_rolling_feature(df_imp_mode_test, columns_A3_r)
df_rolling_OA3_test = create_rolling_feature(df_imp_mode_test, ['OA_3'])


# 2-4-2
# create df_base_A3_test to generate synthetic data of ['IA3_13','IA3_15','IA3_16','IA3_17','IA3_18']

df_base_A3_test = pd.merge(df_imp_mode_test[['OA_3']],
                           df_rolling_OA3_test, 
                           right_index = True, left_index = True, how = 'left')

df_base_A3_test = pd.merge(df_base_A3_test ,
                           df_rolling_IA3_test,
                           right_index = True, left_index = True, how = 'left')

df_base_A3_test = pd.merge(df_base_A3_test ,
                           df_imp_mode_test[columns_A3_r],
                           right_index = True, left_index = True, how = 'left')

df_base_A3_test = pd.merge(df_base_A3_test ,
                           df_imp_mode_test[text_features_xy],
                           right_index = True, left_index = True, how = 'left')


# 2-4-3
# fit and generate synthetic data of ['IA3_13','IA3_15','IA3_16','IA3_17','IA3_18'] by reg_a3_mop
df_syn_key_A3_test = pd.DataFrame(reg_a3_mop.predict(csr_matrix(df_base_A3_test)),
                                  columns =['IA3_13','IA3_15','IA3_16','IA3_17','IA3_18'])


df_key_A3_final = pd.DataFrame(index = df_imp_mode_test.index)
# use fitted reg_c_mop to generate synthetic data: df_key_A3_final
for a in ['IA3_13','IA3_15','IA3_16','IA3_17','IA3_18']:
    
    # drop one of feature from ['IA3_13','IA3_15','IA3_16','IA3_17','IA3_18'], which is the objective in this iteration
    df_syn_key_A3_stage = df_syn_key_A3_test.drop([a], axis = 1)
    
    # create rolling features from ['IA3_13','IA3_15','IA3_16','IA3_17','IA3_18']
    df_rolling_syn_key_A3_stage = create_rolling_feature(df_syn_key_A3_stage,
                                                         df_syn_key_A3_stage.columns)
    
    # merge synthetic data and its rolling features with df_base_A3_test
    df_base_A3_plus = pd.merge(df_syn_key_A3_stage, df_rolling_syn_key_A3_stage,
                              right_index = True, left_index = True, how = 'left')
    
    df_base_A3_plus = pd.merge(df_base_A3_plus, df_base_A3_test,
                              right_index = True, left_index = True, how = 'left')
    
        
    # use series of reg_a3_vot_{} to predict individual final result  of ['IA3_13','IA3_15','IA3_16','IA3_17','IA3_18']
    locals()['{}_final'.format(a)] = pd.DataFrame(locals()['reg_a3_vot_{}'.format(a)].predict(csr_matrix(df_base_A3_plus)),
                                                  columns = [a])
    
    df_key_A3_final = pd.merge(df_key_A3_final, locals()['{}_final'.format(a)],
                              right_index = True, left_index = True, how = 'left')
    
    # End of loop

# End predicting process of ['IA3_13','IA3_15','IA3_16','IA3_17','IA3_18']
# Get final prediction of features_key_A3: df_key_A3_final
# -----------------------------------------------------------------------------






# -----------------------------------------------------------------------------
# Start predicting process of ['IA6_1' ,'IA6_11','IA6_19','IA6_24']

# 2-5-1
# use [['OA_6'],  columns_A6_r]
# to create rolling features which exclude key features
df_rolling_IA6_test = create_rolling_feature(df_imp_mode_test, columns_A6_r )
df_rolling_OA6_test = create_rolling_feature(df_imp_mode_test, ['OA_6'])


# 2-5-2
# create df_base_A6_test:
# which contain ['OA_6'] to generate synthetic data of ['IA6_1' ,'IA6_11','IA6_19','IA6_24']
#   Cause we use whole 'features_CN' in df_base_A6 in training phrase,
#   so, in testing phrase, we have to combine 'df_key_C_final' and 'features_CN_r'
#   when merging df_base_A6_test and make sure that the order of columns of 
#   df_base_A6 and df_base_A6_test are same
df_base_A6_test = pd.merge(df_imp_mode_test[['OA_6']], 
                           df_rolling_OA6_test,
                           right_index = True, left_index = True, how = 'left')

df_base_A6_test = pd.merge(df_base_A6_test, 
                           df_rolling_IA6_test,
                           right_index = True, left_index = True, how = 'left')

df_base_A6_test = pd.merge(df_base_A6_test, 
                           df_imp_mode_test[columns_A6_r],
                           right_index = True, left_index = True, how = 'left')

df_base_A6_test = pd.merge(df_base_A6_test, 
                           df_imp_mode_test[text_features_xy],
                           right_index = True, left_index = True, how = 'left')

df_base_A6_test = pd.merge(df_base_A6_test, 
                           df_imp_mode_test[features_CN_r],
                           right_index = True, left_index = True, how = 'left')

df_base_A6_test = pd.merge(df_base_A6_test, 
                           df_key_C_final,
                           right_index = True, left_index = True, how = 'left')

# The following row of code should be return 'True', or there may be something wrong
list(df_base_A6_test.columns) == list(df_base_A6.columns)


# 2-5-3
# fit and generate synthetic data of  ['IA6_1' ,'IA6_11','IA6_19','IA6_24'] by reg_a6_mop
df_syn_key_A6_test = pd.DataFrame(reg_a6_mop.predict(csr_matrix(df_base_A6_test)),
                                  columns = ['IA6_1' ,'IA6_11','IA6_19','IA6_24'])


df_key_A6_final = pd.DataFrame(index = df_imp_mode_test.index)
# use a for loop to predict final results
for a in ['IA6_1' ,'IA6_11','IA6_19','IA6_24']:
    
    # drop one of feature from ['IA6_1' ,'IA6_11','IA6_19','IA6_24'], which is the objective in this iteration
    df_syn_key_A6_stage = df_syn_key_A6_test.drop([a], axis = 1)
    
    # create rolling features from ['IA6_1' ,'IA6_11','IA6_19','IA6_24'] 
    df_rolling_syn_key_A6_stage = create_rolling_feature(df_syn_key_A6_stage,
                                                         df_syn_key_A6_stage.columns)
    
    # merge synthetic data and its rolling features with df_base_A6_test
    df_base_A6_plus = pd.merge(df_syn_key_A6_stage, df_rolling_syn_key_A6_stage,
                              right_index = True, left_index = True, how = 'left')
    
    df_base_A6_plus = pd.merge(df_base_A6_plus, df_base_A6_test,
                               right_index = True, left_index = True, how = 'left')
    
    
    # use series of reg_a6_vot_{} to predict individual final resault of ['IA6_1' ,'IA6_11','IA6_19','IA6_24']
    locals()['{}_final'.format(a)] = pd.DataFrame(locals()['reg_a6_vot_{}'.format(a)].predict(csr_matrix(df_base_A6_plus)),
                                                  columns = [a])
        
    df_key_A6_final = pd.merge(df_key_A6_final, locals()['{}_final'.format(a)],
                                right_index = True, left_index = True, how = 'left')
    
    # End of loop

# End predicting process of ['IA6_1' ,'IA6_11','IA6_19','IA6_24'] 
# Get final prediction of features_key_A6: df_key_A6_final
# -----------------------------------------------------------------------------


# import time
# start_time = time.time()
# end_time = time.time()
# print( round(end_time - start_time ,2) )