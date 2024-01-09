import numpy as np
import pandas as pd

import os

## locations where the files with confidence scores need to be saved
save_location = 'confidence_results/'

## csv file with the SQL code repo
sql_repo_csv_location =  'sql_code_repo_v5.csv'

sql_folder_location =  'SQL_dump/'
gamma = 1
max_t_w_value = 0.975
maximum_number_of_iterations = 100 
minimum_absolute_difference =  0.001

if not os.path.exists(f"./{save_location}"):
    os.makedirs(save_location)

#%% Pre-defined variables

## Function to carry out the iterations of confidence algo 
def carry_out_iterations( data,list_of_cols,t_w,id_colname,gamma): 
    
    """
    Arguments: 
        data : the table serving as input for the confidence algo 
        list_of_cols:  list of column names that serve as unique sources for the data points 
        t_w:  array of initial confidence values (source trustworthiness) for each source. Usually set to 0.5 for each source
        id_colname: Column name with unique id value for each row
        
    Returned values: 
    t_w_df : table with the changing source trustworthiness with each iteration
    train_data_confidence : Final table with the final confidence values for each data point for each source
    
    """
        
    
    
    train_data =  data[list_of_cols].copy()
    
    train_data = train_data.loc[np.sum(~train_data.isna(),axis = 1) > 1,:]
    ## creating empty data frame with same structure as traindata to copy confidence scores 
    
    train_data_confidence =  train_data.copy()
    train_data_confidence.loc[:,:]= 0

    ## calculating (1-t(w)). Carrying out calculation required for the equation
    t_w_inv =  1- t_w
    tau_w =  -np.log(t_w_inv)

    ## creating dataframe that maintains list of confidence values through each iteration
    confidence_iterations = pd.DataFrame(columns =train_data.columns.tolist() + ['iteration'])
    t_w_df = pd.DataFrame(columns = train_data.columns)

    for iteration in range(0,maximum_number_of_iterations):

        for col_name in list_of_cols:
            column_matching_df=  train_data_confidence.copy()
            column_matching_df.loc[:,:]= 0
            current_source =  train_data[col_name]

            other_sources_cols = [x for x in list_of_cols if x != current_source.name]

            column_matching_df[col_name] = 1
            for col_name_others in other_sources_cols:
                column_matching_df[col_name_others] = np.where(train_data[col_name_others]==current_source,1,-1)
            column_matching_df[pd.isnull(train_data)]=0

            for col_ii in range(0,column_matching_df.shape[1]):
                column_matching_df.iloc[:,col_ii] = column_matching_df.iloc[:,col_ii] * tau_w[col_ii]

            train_data_confidence[col_name]= np.where(pd.isnull(current_source),np.nan,1/(1 + np.exp( -1 * gamma * ( column_matching_df.sum(axis=1)  ) )))


        ## maintaining record of the trusworthiness scores of websites
        t_w_prev =  t_w.copy()
        t_w_df.loc[iteration]= t_w
        t_w = train_data_confidence.mean()
        t_w [t_w >= max_t_w_value] = max_t_w_value
        t_w_inv =  1- t_w
        tau_w =  -np.log(t_w_inv)

        ## printing itertion number and the trustworthiness score
        print(iteration, np.array(t_w_prev))
        if iteration > 5:
            if np.nansum(np.abs(t_w.values - t_w_prev.values)) < minimum_absolute_difference:
                break
    
    train_data_confidence[id_colname] =  data[id_colname]
    
    return(t_w_df,train_data_confidence )


## Function to get the final confidence values from the source trustworthiness values 
def get_final_confidence(data,list_of_cols, column_to_check_confidence,t_w ,id_colname):
    
    """
    Arguments: 
        data : the table serving as input for the confidence algo 
        list_of_cols:  list of column names that serve as unique sources for the data points 
        column_to_check_confidence:  column for which final confidence needs to be calcuated 
        t_w: trustworthiness score for each source (final values from the iterations)
        id_colname: colum with unqiue id values for each row
        
    Returned values: 
    data  : Table which returns the final confidence scores for the required columns    
    """
        
    train_data =  data[list_of_cols].copy()
    
    column_matching_df =  train_data.copy()
    column_matching_df.loc[:,:]= 0
    
    if (np.isnan(t_w[0])):
        t_w[0] = t_w[1]
    if (np.isnan(t_w[1])):
        t_w[1]= t_w[0]
    
    
    ## calculating (1-t(w)). Carrying out calculation required for the equation
    t_w_inv =  1- t_w
    tau_w =  -np.log(t_w_inv)

    current_source =  data[column_to_check_confidence]

    other_sources_cols = [x for x in list_of_cols ]

    for col_name_others in other_sources_cols:
        column_matching_df[col_name_others] = np.where(train_data[col_name_others]==current_source,1,-1)
    column_matching_df[pd.isnull(train_data)]=0

    for col_ii in range(0,column_matching_df.shape[1]):
        column_matching_df.iloc[:,col_ii] = column_matching_df.iloc[:,col_ii] * tau_w[col_ii]

    final_conf_scores= np.where(pd.isnull(current_source),np.nan,1/(1 + np.exp( -1 * gamma * ( column_matching_df.sum(axis=1)  ) )))
       

    data['final_confidence'] = final_conf_scores

    return(data)


## reading the table with the SQL queries for downloading necessary tables 
codes_df = pd.read_csv(sql_repo_csv_location)
codes_df_run = codes_df.loc[codes_df.Multiple_confidence_columns == 1,: ]


for table_no in codes_df_run.index:
    print('Table current :',  table_no , '\n')
    string  = codes_df.loc[table_no,'SQL Code']
    string = string.replace('\n'," ")
    string = string.replace('\t'," ")
    columns = codes_df.loc[table_no,'Columns_list']
    list_of_cols = np.array(columns.split (","))
    table_name_str  = codes_df.loc[table_no,'Parent Label']+'_'+ codes_df.loc[table_no,'Field Name']
    print(table_name_str)
    
    try:
        df = pd.read_csv(sql_folder_location + table_name_str + '.csv')
        if (df.shape[0] > 1): 
            df['id'] = df.index
            df['Krushak_Odisha'] = df.field.combine_first(df.self)
            no_cols =  len(list_of_cols)
            t_w = np.repeat(0.5,no_cols)
            id_colname = 'id'
            
            t_w_df,train_data_confidence = carry_out_iterations( df,list_of_cols,t_w,id_colname, gamma)

            column_to_check_confidence = 'Krushak_Odisha'

            data_copy = get_final_confidence(df, list_of_cols, column_to_check_confidence,t_w_df.loc[t_w_df.shape[0]-1,:] ,id_colname)

            conf_table = data_copy[['Krushak_Odisha','int_krushk_id','final_confidence']]

            conf_table.to_csv( save_location + table_name_str+str(table_no)+'.csv', encoding = "utf-8")
        else :
            print('Table has <= 1 row of data')
    except :
        print('Table ',table_name_str, ' does not exist')