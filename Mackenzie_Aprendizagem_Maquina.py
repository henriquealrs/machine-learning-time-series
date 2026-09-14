
from sys import displayhook
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as scist

# Creating the Data Frame with the data from the first sheet of the xlsx file
# Creating the Data Frame with the parameters from the second sheet of the xlsx file

mileage = '1000GD'
excel_file = "data/1000GD.xlsx"

parameters_sheet = 1
data_dict = pd.read_excel(excel_file, sheet_name = [0, parameters_sheet])

data = data_dict[0]
param = data_dict[parameters_sheet]

print('\nAll the data:')
print(data)
print('\n\nParameters Table:')
print(param)

# Creating a dictionary containing all the information for each parameter
# Creating a list with the headers and their indexes
# Creating a list with the fundamental dimensions

headers = []
count_hd = 0
for title in param.head(0).columns:
    headers.append({title:count_hd})
    count_hd += 1

FundDim = []
for title in headers:
    if len(list(title.keys())[0]) == 1:
        FundDim.append(title)

print("\nFundamental Dimensions:\n",FundDim,'\n')

all_parameters = {}
index = 0

for name in param[param.columns[0]]:
    
    info = []
    for fund in FundDim:
        info.append({param.columns[list(fund.values())[0]]:param.at[index, param.columns[list(fund.values())[0]]]})

    #info.append({param.columns[4]:param.at[index, param.columns[4]]})
    #info.append({param.columns[5]:param.at[index, param.columns[5]]})
    #info.append({param.columns[6]:param.at[index, param.columns[6]]})

    info.append({param.columns[3]:param.at[index, param.columns[3]]})
    info.append({param.columns[1]:param.at[index, param.columns[1]]})
    info.append({param.columns[0]:param.at[index, param.columns[0]]})

    all_parameters['{}'.format(name)] = info
    
    print(param.at[index, param.columns[1]],'- variable name:',name,'\n', info, '\n')
    index += 1

# Setting the repeatable variables and pulling them together

for item in headers:
    if list(item.keys())[0] == 'Repeatable':
        column_rep = list(item.values())[0]

rep_parameters = []
variables = param.loc[(param[param.columns[column_rep]] == "ok"),[param.columns[0], param.columns[7]]]

for parameter in variables[variables.columns[0]]:
    rep_parameters.append(parameter)

dimension_rep = len(rep_parameters)

print('The repeatable variables are:')
for var in rep_parameters:
    ind = len(all_parameters[var])
    print(all_parameters[var][ind-1][param.columns[0]],'-',all_parameters[var][ind-2][param.columns[1]])

# Creating an array 'A' with rows - MLT for instance 0 and one column for each repeatable parameter
# Also creating an array for the non-repeatable parameters

A = np.array([[]])

n_rep_parameters = {}
n_rep_parameters.update(all_parameters)

count = 0
for rep_var in rep_parameters:

    dimensions = []
    counter = 0
    for fund in FundDim:
        dimensions.append(all_parameters[rep_var][counter][list(fund.keys())[0]])
        counter += 1

    del(n_rep_parameters[rep_var])
    
    if count == 0:
        A = np.array([dimensions])
        count = 1
    else:
        A = np.append(A, [dimensions], axis = 0)

A_checkB = A

# Deleting a null column

count_check = 0
leftDim = []
for item in FundDim:

    checknull = np.all((A[:,count_check] == 0))
    if checknull == True:
        A = np.delete(A, count_check, 1)
        print('The column relative to',list(item.keys())[0],'dimension was deleted from A because it is null.')
        count_check -= 1
    elif checknull == False:
        print('The column relative to',list(item.keys())[0],'has values')
        leftDim.append(list(item.keys())[0])

    count_check += 1

print('\nA matrix legend\nRow order:',leftDim,'\nColumn order:',rep_parameters)
A = np.transpose(A)
print('\n',A)

# Dimensions of A
' CRIAR CONDICIONAL DE ERRO QUANDO ROW != COLUMN PARA REESCOLHER A QUANTIDADE DE VAR REPT OU MLT DAS VAR'
rows_A = A.shape[0]
columns_A = A.shape[1]

# Creating an array for each non-repeatable parameter and solving the equation Ax=B
# To solve this problem, it is important to follow some conditions
# Creating the Pi list with its paramenters inside

PiList = []

counter = 1
for n_rep_var in n_rep_parameters:

    print(n_rep_var)

    loopcheck = True
    VarList = []
    B = np.array([])
    count_ = 0

    for item in FundDim:

        checknull = np.all((A_checkB[:,count_] == 0))
        if checknull == True:
            if n_rep_parameters[n_rep_var][count_][list(FundDim[count_].keys())[0]] != 0:
                print('It is not possible to create a Pi with parameter', n_rep_var,'because the repeatable parameters do not have',list(FundDim[count_].keys())[0],'dimension, unlike this parameter.')
                loopcheck = False
        else:
            Dim_rep_var = n_rep_parameters[n_rep_var][count_][list(FundDim[count_].keys())[0]]
            B = np.append(B, Dim_rep_var, axis = None)
        
        count_ += 1
    
    if loopcheck == True:
        B = -1 * B
        X = np.linalg.inv(A).dot(B)

        check_zero = np.all(X==0)

        if check_zero == False:
            print('\nPi', counter,': ')
            counter_2 = 0

            for rep_var in rep_parameters:
                print(rep_var, '^', X[counter_2])
                VarList.append({rep_var:X[counter_2]})
                counter_2 += 1

            print(n_rep_var, '^ 1', '\n')
            VarList.append({n_rep_var:1})
            PiList.append({str("Pi" + str(counter)):VarList})

        else:
            print('It is not possible to create a Pi with parameter', n_rep_var,'because the only solution to the matrix results in null powers.')

    counter += 1

print(PiList)

# New cell code option to plot graphs without PDF and conditions

# Changing from alias to varible name of Pi parameters

NewPiList = []
for Pi in PiList:

    Pi_number = list(Pi.keys())[0]

    Pi_name_list = []

    for PiVar in list(Pi.values()):

        idx = 0

        while idx <= (len(PiVar) - 1):

            Var_alias = list(PiVar[idx].keys())
            Var_alias = Var_alias[0]

            rowindex = param[param[param.columns[0]] == Var_alias].index.values
            rowindex = rowindex[0]

            Var_name = param.loc[rowindex, [param.columns[1]]]
            Var_name = Var_name[0]

            Pi_name_list.append({Var_name:list(PiVar[idx].values())[0]})
            
            if idx == len(PiVar) - 1:
                NewPiList.append({Pi_number:Pi_name_list})

            idx += 1

"Enter the index number of the column you want to plot in y axis (integer)."
"Fuel_rate = 6"
y_axis_index = 6
y_axis = str(list(data.columns)[y_axis_index])

# Deleting the row choice and null choice so all rows will be ploted
df = data

# Creating a calculated column in the dataframe, this column is a Pi parameter calculated based on other columns
for Pi in NewPiList:

    Pi_number = list(Pi.keys())[0]

    for PiVar in list(Pi.values()):

        idx = 0

        while idx <= (len(PiVar) - 1):
            if idx == 0:
                ColumnName = list(PiVar[idx].keys())[0]
                Power = list(PiVar[idx].values())[0]
                data[Pi_number] = data[ColumnName] ** Power
            else:
                ColumnName = list(PiVar[idx].keys())[0]
                Power = list(PiVar[idx].values())[0]
                data[Pi_number] = data[Pi_number] * (data[ColumnName] ** Power)

            idx += 1

    # Setting the initial and final row of the data

    initial = 0
    final = 3852
    df2 = df.loc[initial:final]

    # Creating a scatter plot for each Pi parameter in x and the choosen column in y
    #df2.plot(x = x_axis, y = Pi_number, kind = 'scatter')
    #plt.show()

    x = df2[Pi_number]
    y = df2[y_axis]
    slope, intercept, r_value, p_value, std_err = scist.linregress(x, y)
    plt.plot(x, y, 'o', label='Original data - ' + mileage)
    plt.plot(x, intercept + slope*x, 'r', label='Fitted line')
    plt.xlabel(Pi_number + '  rows(' + str(initial) + '-' + str(final) + ')', loc = 'center')
    plt.ylabel(y_axis)
    plt.legend()
    plt.grid()
    print("With trend line.\nR²: ", r_value**2)
    plt.show()
