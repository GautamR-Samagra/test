import numpy as np
import pandas as pd
import configparser

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

import pymysql #Changed for DB Credential || Mrutunjay Pani

# CONFIG
config = configparser.ConfigParser()
config.read('./config.ini')

database_username = config.get('DATABASE','ko_database_username') ## Changed for DB Credential || Mrutunjay Pani
database_password = config.get('DATABASE','ko_database_password')  ## Changed for DB Credential || Mrutunjay Pani
database_name = config.get('DATABASE','ko_database_name') ## Changed for DB Credential || Mrutunjay Pani
localhost = config.get('DATABASE','ko_localhost') ## Changed for DB Credential || Mrutunjay Pani

#Changed for DB Credential || Mrutunjay Pani
def mysql_connect():
    """Connect to a MySQL server using the SSH tunnel connection
    
    :return connection: Global MySQL database connection
    """
    
    connection = pymysql.connect(
        host=localhost,
        user=database_username,
        passwd=database_password,
        db=database_name,
        port=3306
    )
    return connection

#Changed for DB Credential || Mrutunjay Pani
def run_query(sql, conn):
    """Runs a given SQL query via the global database connection.
    
    :param sql: MySQL query
    :return: Pandas dataframe containing results
    """
    
    return pd.read_sql_query(sql, conn)

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

## Gete Column name for Inserting Single confidence data || Mrutunjay Pani
def saveFieldConfidenceField(mrutuFieldName):
    mrutuFieldName = mrutuFieldName.strip()
    val = '0'
    if(mrutuFieldName=='social category'):
        val = 'vch_social_category'
        return val
    elif(mrutuFieldName=='primary mobile number'):
        val = 'vch_primary_mobile'
        return val
    elif(mrutuFieldName=='farmer type'):
        val = 'vch_farmer_type'
        return val
    elif(mrutuFieldName=='district'):
        val = 'vch_district'
        return val
    elif(mrutuFieldName=='block,nac,ulb'):
        val = 'vch_block'
        return val
    elif(mrutuFieldName=='gram panchayat,ward'):
        val = 'vch_gp'
        return val
    elif(mrutuFieldName=='village'):
        val = 'vch_village'
        return val
    else:
        return val

## Inserting or update Single Column confidence in table according to script || Mrutunjay Pani
def saveFieldConfidence(obj,mrutuFieldName):
    fieldName = saveFieldConfidenceField(mrutuFieldName)
    for index, mrutuItemRow in obj.iterrows():
        mrutuAadhaarNo = mrutuItemRow['vch_aadharno']
        mrutuKrushakId = int(mrutuItemRow['int_krushk_id'])
        mrutuFinalConfidence = mrutuItemRow['final_confidence']
        # mrutuFinalConfidence = mrutuFinalConfidence if pd.notna(mrutuFinalConfidence) else 0
        connection = pymysql.connect(host=localhost, user=database_username, password=database_password, database=database_name)
        cursor = connection.cursor()
        quesryCheck = f"SELECT * FROM t_single_confidence_details WHERE int_krushak_id={mrutuKrushakId} and vch_aadhaar=%s"
        cursor.execute(quesryCheck,(mrutuAadhaarNo))
        rowFetch = cursor.fetchone()
        if rowFetch:
            formatIntId = int(rowFetch[0])
            upQuery = f"UPDATE t_single_confidence_details SET {fieldName} = %s WHERE int_id = {formatIntId}"
            # Execute the update query with the column values
            cursor.execute(upQuery, (mrutuFinalConfidence))
            executed_query = cursor._executed
            # Commit the changes to the database
            connection.commit()
        else:
            query = f"INSERT INTO t_single_confidence_details (int_krushak_id,vch_aadhaar,{fieldName}) VALUES ({mrutuKrushakId}, %s, %s)"
            values = (mrutuAadhaarNo, mrutuFinalConfidence)
            cursor.execute(query, values)
            executed_query = cursor._executed
            # Commit the changes to the database
            connection.commit()

## Inserting or update confidence in table according to script || Mrutunjay Pani
def saveMultipleFieldConfidence(obj,mrutuFieldName):
    for index, mrutuItemRow in obj.iterrows():
        mrutuAadhaarNo = mrutuItemRow['vch_aadharno']
        mrutuKrushakId = int(mrutuItemRow['int_krushk_id'])
        if(mrutuFieldName=='crop district'):
            mrutuKrushakData = mrutuItemRow['Krushak_Odisha']
        else:
            mrutuKrushakData = int(mrutuItemRow['Krushak_Odisha'])
        mrutuKrushakData = None if pd.isna(mrutuKrushakData) else mrutuKrushakData
        mrutuFinalConfidence = mrutuItemRow['final_confidence']
        # mrutuFinalConfidence = mrutuFinalConfidence if pd.notna(mrutuFinalConfidence) else 0
        connection = pymysql.connect(host=localhost, user=database_username, password=database_password, database=database_name)
        cursor = connection.cursor()
        if(mrutuFieldName=='farmer occupation'):
            quesryCheck = f"SELECT * FROM t_confidence_details_occupation WHERE int_krushak_id={mrutuKrushakId} and vch_aadhaar=%s and int_occupation={mrutuKrushakData}"
            cursor.execute(quesryCheck,(mrutuAadhaarNo))
            rowFetch = cursor.fetchone()
            if rowFetch:
                formatIntId = int(rowFetch[0])
                upQuery = f"UPDATE t_confidence_details_occupation SET vch_confidence = %s WHERE int_id = {formatIntId}"
                # Execute the update query with the column values
                cursor.execute(upQuery, (mrutuFinalConfidence))
                connection.commit()
            else:
                query = f"INSERT INTO t_confidence_details_occupation (int_krushak_id,vch_aadhaar,int_occupation,vch_confidence) VALUES ({mrutuKrushakId}, %s, {mrutuKrushakData}, %s)"
                values = (mrutuAadhaarNo, mrutuFinalConfidence)
                cursor.execute(query, values)
                connection.commit()
        elif(mrutuFieldName=='type of crop cultivator'):
            quesryCheck = f"SELECT * FROM t_confidence_details_crop_cultivator_type WHERE int_krushak_id={mrutuKrushakId} and vch_aadhaar=%s and int_cultivator={mrutuKrushakData}"
            cursor.execute(quesryCheck,(mrutuAadhaarNo))
            rowFetch = cursor.fetchone()
            if rowFetch:
                formatIntId = int(rowFetch[0])
                upQuery = f"UPDATE t_confidence_details_crop_cultivator_type SET vch_confidence = %s WHERE int_id = {formatIntId}"
                # Execute the update query with the column values
                cursor.execute(upQuery, (mrutuFinalConfidence))
                connection.commit()
            else:
                query = f"INSERT INTO t_confidence_details_crop_cultivator_type (int_krushak_id,vch_aadhaar,int_cultivator,vch_confidence) VALUES ({mrutuKrushakId}, %s, {mrutuKrushakData}, %s)"
                values = (mrutuAadhaarNo, mrutuFinalConfidence)
                cursor.execute(query, values)
                connection.commit()
        elif(mrutuFieldName=='activities'):
            quesryCheck = f"SELECT * FROM t_confidence_details_activities WHERE int_krushak_id={mrutuKrushakId} and vch_aadhaar=%s and int_activities={mrutuKrushakData}"
            cursor.execute(quesryCheck,(mrutuAadhaarNo))
            rowFetch = cursor.fetchone()
            if rowFetch:
                formatIntId = int(rowFetch[0])
                upQuery = f"UPDATE t_confidence_details_activities SET vch_confidence = %s WHERE int_id = {formatIntId}"
                # Execute the update query with the column values
                cursor.execute(upQuery, (mrutuFinalConfidence))
                connection.commit()
            else:
                query = f"INSERT INTO t_confidence_details_activities (int_krushak_id,vch_aadhaar,int_activities,vch_confidence) VALUES ({mrutuKrushakId}, %s, {mrutuKrushakData}, %s)"
                values = (mrutuAadhaarNo, mrutuFinalConfidence)
                cursor.execute(query, values)
                connection.commit()
        elif(mrutuFieldName=='crop district'):
            mrutuAadhaarNo = mrutuItemRow['vch_aadharno']
            mrutuKrushakId = int(mrutuItemRow['int_krushk_id'])
            mrutuFinalConfidence = mrutuItemRow['final_confidence']
            mrutuDistrict = mrutuItemRow['vch_district']
            mrutuTahasil = mrutuItemRow['vch_tahsil']
            mrutuRiCircle = mrutuItemRow['vch_revenue_circle']
            mrutuVillage = mrutuItemRow['vch_village']
            mrutuKhataNo = mrutuItemRow['vch_khata_no']
            mrutuPlotNo = mrutuItemRow['vch_plot_no']
            mrutuAadhaarNo = None if pd.isna(mrutuAadhaarNo) else mrutuAadhaarNo
            mrutuDistrict = None if pd.isna(mrutuDistrict) else mrutuDistrict
            mrutuTahasil = None if pd.isna(mrutuTahasil) else mrutuTahasil
            mrutuRiCircle = None if pd.isna(mrutuRiCircle) else mrutuRiCircle
            mrutuVillage = None if pd.isna(mrutuVillage) else mrutuVillage
            mrutuKhataNo = None if pd.isna(mrutuKhataNo) else mrutuKhataNo
            mrutuPlotNo = None if pd.isna(mrutuPlotNo) else mrutuPlotNo
            # mrutuFinalConfidence = mrutuFinalConfidence if pd.notna(mrutuFinalConfidence) else 0
            quesryCheck = f"SELECT * FROM t_confidence_cropping_land_details WHERE int_krushak_id={mrutuKrushakId} and vch_aadhaar=%s and vch_district=%s and vch_tahsil=%s and vch_revenue_circle=%s and vch_village=%s and vch_khata_no=%s and vch_plot_no=%s"
            cursor.execute(quesryCheck,(mrutuAadhaarNo,mrutuDistrict, mrutuTahasil, mrutuRiCircle, mrutuVillage, mrutuKhataNo, mrutuPlotNo))
            rowFetch = cursor.fetchone()
            if rowFetch:
                formatIntId = int(rowFetch[0])
                upQuery = f"UPDATE t_confidence_cropping_land_details SET vch_confidence = %s WHERE int_id = {formatIntId}"
                # Execute the update query with the column values
                cursor.execute(upQuery, (mrutuFinalConfidence))
                connection.commit()
            else:
                query = f"INSERT INTO t_confidence_cropping_land_details (int_krushak_id,vch_aadhaar,vch_district,vch_tahsil,vch_revenue_circle,vch_village,vch_khata_no,vch_plot_no,vch_confidence) VALUES ({mrutuKrushakId}, %s, %s, %s, %s, %s, %s, %s, %s)"
                values = (mrutuAadhaarNo, mrutuDistrict, mrutuTahasil, mrutuRiCircle, mrutuVillage, mrutuKhataNo, mrutuPlotNo, mrutuFinalConfidence)
                cursor.execute(query, values)
                connection.commit()

## reading the table with the SQL queries for downloading necessary tables 
codes_df = pd.read_csv(sql_repo_csv_location, low_memory=False)
codes_df_run = codes_df.loc[codes_df.Multiple_confidence_columns == 1,: ]
#codes_df_run = codes_df.loc[range(8,13),: ]

for table_no in codes_df_run.index:
    print('Table current :',  table_no , '\n')
    
    columns = codes_df.loc[table_no,'Columns_list']
    list_of_cols = np.array(columns.split (","))
    table_name_str  = codes_df.loc[table_no,'Parent Label']+'_'+ codes_df.loc[table_no,'Field Name']
    mrutuField = codes_df.loc[table_no,'Field Name'].lower() # Changed for DB Credential || Mrutunjay Pani
    try:
        df = pd.read_csv(sql_folder_location + table_name_str + '.csv', low_memory=False)
        if (df.shape[0] > 1): 
            df['id'] = df.index
            df['Krushak_Odisha'] = df.ko.combine_first(df.ko)
            no_cols =  len(list_of_cols)
            t_w = np.repeat(0.5,no_cols)
            id_colname = 'id'
            
            t_w_df,train_data_confidence = carry_out_iterations( df,list_of_cols,t_w,id_colname, gamma)

            column_to_check_confidence = 'Krushak_Odisha'

            data_copy = get_final_confidence(df, list_of_cols, column_to_check_confidence,t_w_df.loc[t_w_df.shape[0]-1,:] ,id_colname)

            # Inserting Confidence Data to Below DB Table
            # ################ START ################
            # t_single_confidence_details
            # t_confidence_details_activities
            # t_confidence_details_occupation
            # t_confidence_cropping_land_details

            # t_confidence_details_crop_cultivator_type 
 
            # Created By :: Mrutunjay Pani

            if(table_name_str=='Crop Production Details_District'):
                conf_table = data_copy[['Krushak_Odisha','int_krushk_id','vch_aadharno','vch_district','vch_tahsil','vch_revenue_circle','vch_village','vch_khata_no','vch_plot_no','final_confidence']]
                mrutuField='crop district'
            else:
                conf_table = data_copy[['Krushak_Odisha','int_krushk_id','vch_aadharno','final_confidence']]
            listOfMultipleFields = ['farmer occupation','type of crop cultivator','activities','crop district']
            if mrutuField in listOfMultipleFields:
                saveMultipleFieldConfidence(conf_table,mrutuField)
            else :
                saveFieldConfidence(conf_table,mrutuField)
                
            # Inserting Confidence Data to Below DB Table
            # ################ END ################
            # Created By :: Mrutunjay Pani

            conf_table.to_csv( save_location + table_name_str+str(table_no)+'.csv', encoding = "utf-8")
        else :
            print('Table has <= 1 row of data')
    except :
        print('Table ',table_name_str, ' has an issue in columns provided')