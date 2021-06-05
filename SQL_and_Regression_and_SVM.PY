#!/usr/bin/python

from __future__ import print_function
import datetime
import numpy
import pandas
import sqlalchemy
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.utils import resample
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import f1_score

# Forming column titles out of HMI tagNames by data type and tagType (alarm/status/control/setup)
# Total columns = Rec.Num + Oper.ID + 73 variables (tags)
# 17 bool alarm tags (3-20) excluding 20;  - 0=DefaultRowNo 1=Rec.Num. 2=Oper.ID
arrTagNamesBOOL_ALM = ["ALM_COOL_PUMP_RUN_COL_RET_VALVE_CLS","ALM_COOL_PUMP_WORK_HRS_MAX",
                        "ALM_INLET_FLOW_HIGH","ALM_INLET_FLOW_LOW","ALM_INLET_PRESSURE_MIN",
                        "ALM_P1_FILTER_CLOGGED","ALM_P1_WORK_HRS_MAX","ALM_P2_FILTER_CLOGGED",
                        "ALM_P2_WORK_HRS_MAX","ALM_P3_FILTER_CLOGGED","ALM_P3_WORK_HRS_MAX",
                        "ALM_RP_FILTER_CLOGGED","ALM_RP_WORK_HRS_MAX","ALM_TANK_LEVEL_HIGH",
                        "ALM_TANK_LEVEL_LOW","ALM_TANK_TEMP_MAX","ALM_TANK_TEMP_MIN"]
# 9 bool status tags (20-29) excluding 29;
arrTagNamesBOOL_STAT = ["STAT_COOL_PUMP_RUN","STAT_COOL_RET_VALVE_OPEN","STAT_DISCHG_VALVE_OPEN",
                        "STAT_HEATER_RUN","STAT_INLET_VALVE_OPEN","STAT_P1_RUN","STAT_P2_RUN",
                        "STAT_P3_RUN","STAT_RP_RUN"]                      
# 6 int status tags (29-35)  !! operatorID tag excluded -> ID will be taken from any table in sql query
arrTagNamesINT_STAT = ["STAT_COOL_PUMP_WORK_HRS","STAT_FLOW_IN","STAT_P1_WORK_HRS",
					    "STAT_P2_WORK_HRS","STAT_P3_WORK_HRS","STAT_RP_WORK_HRS"]
# 8 real status tags (35-43) excluding 43;
arrTagNamesREAL_STAT = ["STAT_OUTSIDE_TEMP","STAT_P1_FILTER_CLOG","STAT_P2_FILTER_CLOG",
					    "STAT_P3_FILTER_CLOG","STAT_PRESSURE_IN","STAT_RP_FILTER_CLOG",
						"STAT_TANK_LEVEL","STAT_TANK_TEMP"]
# 27 bool cmd tags (43-70) excluding 70;  ###########  Targets below this line ####################
arrTagNamesBOOL_CMD = ["CMD_COOL_PUMP_START","CMD_COOL_PUMP_STOP","CMD_COOL_PUMP_WORK_HRS_RESET",
                        "CMD_COOL_RET_VALVE_CLOSE","CMD_COOL_RET_VALVE_OPEN","CMD_DISCHG_VALVE_CLOSE",
                        "CMD_DISCHG_VALVE_OPEN","CMD_HEATER_START","CMD_HEATER_STOP","CMD_INLET_VALVE_CLOSE",
                        "CMD_INLET_VALVE_OPEN","CMD_P1_FILTER_CLEAN","CMD_P1_START","CMD_P1_STOP",
                        "CMD_P1_WORK_HRS_RESET","CMD_P2_FILTER_CLEAN","CMD_P2_START","CMD_P2_STOP",
						"CMD_P2_WORK_HRS_RESET","CMD_P3_FILTER_CLEAN","CMD_P3_START","CMD_P3_STOP",
                        "CMD_P3_WORK_HRS_RESET","CMD_RP_FILTER_CLEAN","CMD_RP_START","CMD_RP_STOP",
                        "CMD_RP_WORK_HRS_RESET"]
# 6 int setup tags (70-76) - excluding 76
arrTagNamesINT_SET = ["SET_COOL_PUMP_POWER","SET_HEAT_EXCHG_VALVE_OPEN","SET_P1_POWER",
					    "SET_P2_POWER","SET_P3_POWER","SET_RP_POWER"]
def get_df_from_pg():
    # engine to connect to the PostgreSQL database on another VM (ML plaform connecting to SCADA server).
    engine = sqlalchemy.create_engine('postgresql://postgres:Pero1!@192.168.135.218:5432/dbHMI')
        
    # dataset with bool alarms
    sql_data = pandas.read_sql_table('rt_bool_alm',engine) # get table rt_bool_alm
    recColumnLoaded = 0
    for tag in arrTagNamesBOOL_ALM:
        df = sql_data.loc[sql_data['bool_alm_tag_name'] == tag] # df with only one tag (matching name in array)
        df1 = pandas.DataFrame({"Record_num": df['bool_alm_rec_num'], 
                                "Operator_id": df['bool_alm_operator_id'], tag: df['bool_alm_tag_value']})
        if recColumnLoaded == 0: # first loop only: getting record-number column
            df2 = df1
            recColumnLoaded = 1 # prevent next steps to load record-number column
        else:
            tagVal = df1[tag].to_numpy() # column with tag values in numpy array
            df2[tag] = tagVal # adding tagneme column with values from array

    # dataset with bool alarms + bool stat
    sql_data = pandas.read_sql_table('rt_bool_stat',engine) # get table rt_bool_stat
    recColumnLoaded = 0
    for tag in arrTagNamesBOOL_STAT:
        df = sql_data.loc[sql_data['bool_stat_tag_name'] == tag] # df with only one tag (matching name in array)
        df1 = pandas.DataFrame({"Record_No": df['bool_stat_rec_num'], tag: df['bool_stat_tag_value']})
        tagVal = df1[tag].to_numpy() # column with tag values in numpy array
        df2[tag] = tagVal # adding tagneme column with values from array

    # dataset with bool alarms + bool stat + int stat
    sql_data = pandas.read_sql_table('rt_int_stat',engine) # get table rt_int_stat
    recColumnLoaded = 0
    for tag in arrTagNamesINT_STAT:
        df = sql_data.loc[sql_data['int_stat_tag_name'] == tag] # df with only one tag (matching name in array)
        df1 = pandas.DataFrame({"Record_No": df['int_stat_rec_num'], tag: df['int_stat_tag_value']})
        tagVal = df1[tag].to_numpy() # column with tag values in numpy array
        df2[tag] = tagVal # adding tagneme column with values from array

    # dataset with bool alarms + bool stat + int stat + real stat
    sql_data = pandas.read_sql_table('rt_real_stat',engine) # get table rt_real_stat
    recColumnLoaded = 0
    for tag in arrTagNamesREAL_STAT:
        df = sql_data.loc[sql_data['real_stat_tag_name'] == tag] # df with only one tag (matching name in array)
        df1 = pandas.DataFrame({"Record_No": df['real_stat_rec_num'], tag: df['real_stat_tag_value']})
        tagVal = df1[tag].to_numpy() # column with tag values in numpy array
        df2[tag] = tagVal # adding tagneme column with values from array

    # dataset with bool alarms + bool stat + int stat + real stat + bool cmd
    sql_data = pandas.read_sql_table('rt_bool_cmd',engine) # get table rt_bool_cmd
    recColumnLoaded = 0
    for tag in arrTagNamesBOOL_CMD:
        df = sql_data.loc[sql_data['bool_cmd_tag_name'] == tag] # df with only one tag (matching name in array)
        df1 = pandas.DataFrame({"Record_No": df['bool_cmd_rec_num'], tag: df['bool_cmd_tag_value']})
        tagVal = df1[tag].to_numpy() # column with tag values in numpy array
        df2[tag] = tagVal # adding tagneme column with values from array

    # Final dataset with bool alarms + bool stat + int stat + real stat + bool cmd = 73 tag-value columns
    # + Record_num. + Oper_id column
    sql_data = pandas.read_sql_table('rt_int_set',engine) # get table rt_int_set
    recColumnLoaded = 0
    for tag in arrTagNamesINT_SET:
        df = sql_data.loc[sql_data['int_set_tag_name'] == tag] # df with only one tag (matching name in array)
        df1 = pandas.DataFrame({"Record_No": df['int_set_rec_num'], tag: df['int_set_tag_value']})
        tagVal = df1[tag].to_numpy() # column with tag values in numpy array
        df2[tag] = tagVal # adding tagneme column with values from array

    df2.to_csv('FullDataSet.csv', sep='\t', encoding='utf-8')

#-------------------------------------------------------------------------------
# ############ !!! - comment below line if reading from db !!! ###############
df2 = pandas.read_csv("FullDataSet.csv", sep='\t', encoding='utf-8')
#--------------------------------------------------------------------------------

df_full = df2.iloc[:15000,] # Will be working with 15k rows due to PC (CPU/RAM) limitation

print("\n ########### Full dataframe info ##########")
df_full.info(verbose=True, show_counts=True)
print("\n ########### Full dataframe describe ##########")
print(df_full.describe)

# There are two operators defined in HMI simulation
df_oper1 = df_full.loc[df_full['Operator_id'] == 1]
df_oper2 = df_full.loc[df_full['Operator_id'] == 2]

print("\n ########### Oper1 dataframe info ##########")
df_oper1.info(verbose=True, show_counts=True)

print("\n ########### Oper2 dataframe info ##########")
df_oper2.info(verbose=True, show_counts=True)

# Average of pumps power by operator (for selecting operator with better resources management)
def avg_power_by_oper(df, oper_name): 
    avg_power = (df['SET_P1_POWER'].mean() + df['SET_P2_POWER'].mean() + df['SET_P3_POWER'].mean() + df['SET_COOL_PUMP_POWER'].mean() + df['SET_RP_POWER'].mean())/5
    print('{0} avg power per pump:  {1} % \n'.format(oper_name, avg_power))
    return avg_power

# Dataset from operator with lower consumption rate achieved
if (avg_power_by_oper(df_oper1, "Operator 1") > avg_power_by_oper(df_oper2, "Operator 2")):
    df_full = df_oper2
else:
    df_full = df_oper1

allFeatrueNames = df_full.iloc[:, 3:43].columns
analogFeatrueNames = df_full.iloc[:, 29:43].columns
allTargetNames = df_full.iloc[:, 43:76].columns
setupTargetNames = df_full.iloc[:, 70:76].columns

# Preparing df for SVM - removing clumns with only one class (0)
def reduce(df):
    df_full = df.loc[:, (df != 0).any(axis=0)] # Remove zero columns
    print("\n ########### NO ZERO COLUMNS dataframe info ##########")
    df_full.info(verbose=True, show_counts=True)
    # Eliminate all columns with sum < 200 (which represents activation count for digital signals)
    for col in df_full.columns:
        if df_full[col].sum() < 200:
            #print('Col to delete: {0}'.format(col))
            df_full.drop([col], axis=1, inplace = True)
    # Reducing df on factors only (features and labels)
    df_reduced = df_full.iloc[:, numpy.r_[3:12, 26:36]]
    print("\n ########### df_reduced dataframe info (factors only)) ##########")
    print(df_reduced.info(verbose=True, show_counts=True))
    return df_reduced

def oversampling(df):
    # Oversampling (Upsampling) minority class
    for col in df.columns:
        df_majority = df[df[col]==0]
        df_minority = df[df[col]==1]
    df_minority_upsampled = resample(df_minority, 
                                    replace=True,  
                                    n_samples=len(df_majority), # upscale to majority class
                                    random_state=1234)
    df_upsampled = pandas.concat([df_majority, df_minority_upsampled])

    print("\n ########### df_upsampled dataframe info ##########")
    print(df_upsampled.info(verbose=True, show_counts=True))
    return df_upsampled

# Calculating Variance Inflation Factor for each feature
vif_data = pandas.DataFrame()
vif_data["feature"] = allFeatrueNames
vif_data["VIF"] = [variance_inflation_factor(df_full.iloc[:, 3:43].values, i) for i in range(len(df_full.iloc[:, 3:43].columns))]
print('Variance Inflation Factor (VIF) for all features: \n {0}'.format(vif_data))

# barplot - Variance Inflation Factor
sns.set_context('paper')
vif_sort = vif_data.sort_values('VIF', ascending = False)
sns.set_color_codes('pastel')
bar_pl = sns.barplot(x = 'feature', y = 'VIF', data = vif_sort, color = 'b', edgecolor = 'w')
bar_pl.set(xlabel='Nezavisne varijable', ylabel='VIF')
bar_pl.set_xticklabels(vif_data['feature'], rotation=90,)
bar_pl.axes.set_title('VIF (Variance Inflation Factor)',fontsize=20)
plt.tight_layout()
plt.ylim(0, 20) #Limited y-axis
plt.show()

# Heat map of predictors
kor = sns.heatmap(df_full.iloc[:, 3:43].corr(), annot = False, vmin=-1, vmax=1, center= 0, cmap= 'coolwarm', cbar_kws= {'orientation': 'vertical'})
kor.set(xlabel ="", ylabel = "")
kor.axes.set_title('Korelogram nezavisnih varijabli',fontsize=20)
plt.gcf().subplots_adjust(left=0.25, bottom=0.35)
kor.set_yticks(range(len(df_full.iloc[:, 3:43].columns)))
kor.set_yticklabels(df_full.iloc[:, 3:43].columns)
kor.set_xticks(range(len(df_full.iloc[:, 3:43].columns)))
kor.set_xticklabels(df_full.iloc[:, 3:43].columns) 
plt.show()

# Heat map of labels
kor = sns.heatmap(df_full.iloc[:, 43:76].corr(), annot = False, vmin=-1, vmax=1, center= 0, cmap= 'coolwarm', cbar_kws= {'orientation': 'vertical'})
kor.set(xlabel ="", ylabel = "")
kor.axes.set_title('Korelogram zavisnih varijabli',fontsize=20)
plt.gcf().subplots_adjust(left=0.25, bottom=0.35)
kor.set_yticks(range(len(df_full.iloc[:, 43:76].columns)))
kor.set_yticklabels(df_full.iloc[:, 43:76].columns)
kor.set_xticks(range(len(df_full.iloc[:, 43:76].columns)))
kor.set_xticklabels(df_full.iloc[:, 43:76].columns)
plt.show()

# boxPlot for analog features(int+float)
dfBox = df_full.iloc[:, 29:43]
count = 1
n_col = 14
for i in range(n_col):
    plt.subplot(2, 7, count)
    ax =sns.boxplot(data=dfBox.iloc[:,i, ], orient="h", color='#6890F0')
    ax.set(yticks=[]) #removed scale on y-axis
    ax.set_ylabel(dfBox.columns[i], fontsize=14) # labels on y-axis: df columns
    ax.yaxis.labelpad = 2 # padding
    count += 1 
plt.suptitle('Box plot: Nezavisne varijable (analogne vrijednosti)',  y=0.92, fontsize = 20)  
plt.show()


# below section is for model validation metrics over various numbers of observations
#normalize all exept Observations (for below plot only)
def normalize(dataset):
    dataNorm=((dataset-dataset.min())/(dataset.max()-dataset.min()))*100
    dataNorm["Observations"]=dataset["Observations"]
    return dataNorm

# Plot regression model (previously generated result arrays) with various number of observations(rows)
# - to undestand trend of R-square, MSE, MAE in relation to df size
observations = [2022, 3847, 4573, 7573, 10573]
r_square = [0.71, 0.79, 0.82, 0.85, 0.87]
mse = [152.55, 80.81, 74.03, 47.09, 32.34]
mae = [3.79, 2.09, 1.92, 1.30, 0.93]
dict = {'Observations': observations, 'R-square': r_square, 'MSE': mse, 'MAE': mae}
results_df = pandas.DataFrame(dict, dtype = float)
norm_df =normalize(results_df)
plt.plot( 'Observations', 'R-square', data=norm_df, marker='o', color='blue', linewidth=2, label='R-square')
plt.plot( 'Observations', 'MSE', data=norm_df, marker='D', color='green', linewidth=4, label='MSE')
plt.plot( 'Observations', 'MAE', data=norm_df, marker='<', color='red', linewidth=2, label="MAE")
plt.title("Trend metrika validacije modela regresije porastom broja opservacija", y=1.0, fontsize = 15)
plt.ylabel("[%]", fontsize = 12)
plt.xlabel("Br. opservacija", fontsize = 12)
plt.legend()
plt.show()

# ---------------------------------------------------------------
# Multi-output regression model with Gradient boost regressor
# ---------------------------------------------------------------
def multi_regression(X, Y, no_feat):

    # Test size - 80:20 ratio
    xtrain, xtest, ytrain, ytest=train_test_split(X, Y, test_size=0.20,  random_state = 0)
    print("xtrain:", xtrain.shape, "ytrian:", ytrain.shape)
    print("xtest:", xtest.shape, "ytest:", ytest.shape)

    xtrain.info(verbose=True)
    ytrain.info(verbose=True)

    time_model_start = datetime.datetime.now()
    gbr = GradientBoostingRegressor()
    model = MultiOutputRegressor(estimator=gbr)
    
    model.fit(xtrain, ytrain)
    time_model_fitted = datetime.datetime.now()
    delta_time = time_model_fitted - time_model_start
    print('\n Model training time: {0} sec. \n'.format(delta_time.total_seconds()))
    
    # Featrue importance plot
    sum_est = []
    for i in range(0,no_feat):
        sum_est.append(0) # array with null values size of No.features
    for i in range(len(model.estimators_)):
        print('Estimator {0} importance: {1}'.format(i, model.estimators_[i].feature_importances_))
        sum_est = numpy.add(sum_est, model.estimators_[i].feature_importances_) #add to summ array

    sum_est = sum_est / len(model.estimators_) #normalize summ array
    d = {'Names':X.columns,'Importance':sum_est}    
    df = pandas.DataFrame(d)
    sns.barplot( x='Names', y='Importance', data=df,  order=df.sort_values('Importance').Names)
    plt.xticks(rotation=90)
    plt.suptitle("Feature importance",  y=0.95, fontsize = 15)
    plt.tight_layout()   
    plt.show()  

    score = model.score(xtrain, ytrain)
    print("Training score:", score) # Return the coefficient of determination of the prediction - R-squared (max = 1.0, can be negative)
                                    # (goodnes of fit)   

    time_model_predict_start = datetime.datetime.now()
    # Prediction of Y based on Xtest (resutlt can be compared with Ytest part of dataset)
    ypred = model.predict(xtest)
    time_model_predict_end = datetime.datetime.now()
    delta_time = time_model_predict_end - time_model_predict_start
    print('\n Model prediction time: {0} sec. \n'.format(delta_time.total_seconds()))
    
    print("ypred MSE:%.4f" % mean_squared_error(ytest, ypred))
    print("ypred MAE:%.4f" % mean_absolute_error(ytest, ypred))
# --------- end of Multi-output regression model----------------

# ---------------------------------------------------------------
# Multi-output clasification model - SVM
# ---------------------------------------------------------------
def multi_clasification(df):
    X = df.iloc[:, 0:9]
    Y = df.iloc[:, 9:19]
    
    print('X shape: {0}'.format(X.shape[0]))
    print('Y shape: {0}'.format(Y.shape[0]))
    
    # barplot - classes ratio
    col_name = []
    col_sum = []
    col_false = []
    for col in df.columns:
        labels = numpy.unique(df[col])
        print('df_reduced: {0} = {1}, sum = {2}'.format(col, labels, df[col].sum()))
        col_name.append(col)
        col_sum.append(df[col].sum())
        col_false.append(df[df[col] == 0].shape[0])
    df = pandas.DataFrame()
    df['Variable'] = col_name
    df['Count'] = col_sum
    df['CountFalse'] = col_false
    x = numpy.arange(len(df['Variable']))
    width = 0.35  # width of the bars
    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width/2, df['Count'], width, label='True')
    rects2 = ax.bar(x + width/2, df['CountFalse'], width, label='False')
    ax.set_ylabel('Br. opservacija', fontsize=12)
    ax.set_xlabel('Kategorijske varijable', fontsize=12)
    ax.set_title('Odnos grupa unutar klase', fontsize=16)
    ax.set_xticks(x)
    ax.set_xticklabels(df['Variable'])
    ax.legend()
    plt.xticks(rotation=90)
    fig.tight_layout()
    plt.show()

    # barplot - class ratio between labels
    col_name = []
    col_sum = []
    for col in Y.columns:
        labels = numpy.unique(Y[col]); 
        print('Y: {0} = {1}, sum = {2}'.format(col, labels, Y[col].sum()))
        col_name.append(col)
        col_sum.append(Y[col].sum())
    df1 = pandas.DataFrame()
    df1['Variable'] = col_name
    df1['Count'] = col_sum
    bar_pl = sns.barplot(x = 'Variable', y = 'Count', data = df1, color = 'b', edgecolor = 'w')
    bar_pl.set_xlabel('Kategorijske zavsine varijable', fontsize=12)
    bar_pl.set_ylabel('Br aktivacija signala', fontsize=12)
    bar_pl.axes.set_title('Aktivacije pravljačkh signala',fontsize=18)
    for p in bar_pl.patches:
        bar_pl.annotate(format(p.get_height(), '.0f'), 
                    (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha = 'center', va = 'center', 
                    xytext = (0, 9), 
                    textcoords = 'offset points')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()

    xtrain, xtest, ytrain, ytest=train_test_split(X, Y, train_size=0.85, random_state=0)
    print('xtest no. records = ', len(xtest))
    print('xtrain no. records = ', len(xtrain))
    print('ytest no. records = ', len(ytest))
    print('ytrain no. records = ', len(ytrain))

    svc = SVC(gamma="scale")
    model = MultiOutputClassifier(estimator=svc)
    print(model)

    model.fit(xtrain, ytrain)
    print(model.score(xtrain, ytrain))

    ypred = model.predict(xtest)

    # Area Under the Curve (AUC) of Receiver Characteristic Operator (ROC)
    i = 0
    for col in Y.columns:
        auc_y1 = roc_auc_score(ytest.iloc[:,i],ypred[:,i])
        print('ROC-AUC {0}: {1}'.format(col, auc_y1))
        cm_y1 = confusion_matrix(ytest.iloc[:,i],ypred[:,i])
        print(cm_y1)
        cr_y1 = classification_report(ytest.iloc[:,i],ypred[:,i])
        print(cr_y1)
        i += 1
# --------- end of Multi-output clasification model----------------

# Get dataframe form pg database on SCADA server VM  (Comment below line if reading form FullDataSet.csv          
#get_df_from_pg()

#-----------------------------------------------------------------------------
#           RUN MODELS BELOW THIS LINE
#-----------------------------------------------------------------------------

# Run regression model with full df (73 variables)
X = df_full.iloc[:, 3:43]  # 0:39 = 40 columns (sve nezavisne)
Y = df_full.iloc[:, 43:76] # 0:32 = 33 columns (sve zavisne varijable)
no_feat = 40
multi_regression(X, Y, no_feat)

# Run regression model only with analog values (20 variables)
X = df_full.iloc[:, 29:43]  # 0:13 = 14 columns (analogne nezavisne)
Y = df_full.iloc[:, 70:76] # 0:6 = 6 columns (analogne zavisne)
no_feat = 14
multi_regression(X, Y, no_feat)

# Run regression model with reduced df
# - Without all alarms except ALM_INLET_FLOW_HIGH, ALM_P2_FILTER_CLOGGED - high importance
# - Without all WORK_HRS - outliers,and low importance. 
df_red = df_full
df_red.drop(df_red.iloc[:, numpy.r_[:, 3:5, 6:10, 11:20]], axis=1, inplace=True) # alarms - all but two
df_red.drop(df_red.columns[df_red.columns.str.contains('WORK_HRS')], axis=1, inplace=True) # work hrs variables
X = df_red.iloc[:, 3:23]  # 0:19 = 20 columns (sve nezavisne)
Y = df_red.iloc[:, 23:51] # 0:19 = 18 columns (sve zavisne varijable)
no_feat = 20
multi_regression(X, Y, no_feat)

# Run clasification model with reduced df
multi_clasification(reduce(df_full))

# Run clasification model with reduced and oversampled df
multi_clasification(oversampling(reduce(df_full)))


