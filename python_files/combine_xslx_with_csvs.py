import pandas as pd 
import os
import re
import csv
import numpy as np  

path = os.getcwd()

#find the output{#}.csv files in the directory
pattern = re.compile(r'output\d+\.csv')

# Iterate over the files in the directory and find matches
matching_files = [file for file in os.listdir(path) if pattern.match(file)]

#read the annika.xlsx with pandas 
annika_parameters = pd.read_excel("ML_Prusa_FiberAjustment_Setup.xlsx")
column_names = annika_parameters.columns
print(column_names)
#set columns names of the annika.xslx that you want to be combined with the output{#}.csv
## only if there is a project extension! 
## TODO : Instead of being a static variable this should/could be a user input and then with the re find the corresponding user-input columns 
## TODO: Create a function in returnpoints.py that does the same as the code below
filtered_columns = ['High \nVoltage\n[kV]', 'Speed\n[mm/\nmin]', 'Extrusion\nTotal\n[mm]', 'Divisor\n[mm/\nmm]', 'Humid.\n[%]', 'Room\nT_R\n[°C]']
nummerical_values_csv = []

#open output{#}.csv
for file in matching_files:
    #print('file:',file)
    df = pd.read_csv(file)
    img_names = df["Image_Point"]
    returnp_columns = pd.DataFrame(columns=df.columns)
    #iterate over the names of the csv 
    df_columnns = df.columns.tolist()
    df_columnns.extend(filtered_columns)
    
    for i, name in enumerate(img_names):
        # Find columnd and row of the image in the output.csv in  the annika.xlsx 
        csv_elements = df.iloc[i]
        csv_elements = csv_elements.values.tolist()
        grid_num = name.split("_")[1]
        img_name = name.split("_")[0]
        # Find the column in the annika.xlsx, which we know are in the of the Grid columns 
        pattern = re.compile(f'Grid\n({int(grid_num)})')
        matching_column_names = [col for col in column_names if pattern.match(col)]
        
        # After finding the column search in that specific column to find the corresponding row of the image name,
        # for that purpose we use the img_name  
        name_pattern = re.compile(f'PCL_{img_name}')
        flattened_list = [item for sublist in annika_parameters[matching_column_names].values.tolist() for item in sublist]
        
        #find the corresponding row of the name of in  the output{#}.csv file in the annika.xlsx
        row_index = [i for i, img in enumerate(flattened_list) if name_pattern.match(str(img))]#if name_pattern.match(img)
        extension_values = annika_parameters.loc[row_index[0], filtered_columns].values.tolist()
        
        #check for nan values and replace them with 0
        for i, el in enumerate(extension_values): 
            temp = np.array(el)
            if np.isnan(el): extension_values[i] = float(0.0)
        
        #add the values from the rows of the annika.xlsx 
        csv_elements.extend(extension_values)
        
        string_elements = [str(element) for element in csv_elements]
        string_representation = ', '.join(string_elements)
        nummerical_values_csv.append(string_representation)

csv_filename = 'combined_output1.csv'
with open(csv_filename, 'w', newline='') as csvfile:
                csv_writer = csv.writer(csvfile)
                
                #first row should be the categories
                csv_writer.writerow(df_columnns)
                
                #write the elementns of each category in the combined csv file
                for data_point in nummerical_values_csv:
                    split_data = data_point.split(',')
                    csv_writer.writerow(split_data)