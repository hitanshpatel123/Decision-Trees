from math import log

# Decision trees by first principles

parent = pd.read_csv("toy_df.csv", usecols=["adult", "money_in_pocket"], nrows=5)
print(parent)

thresh_adult = sum(parent['adult'].unique())/2

money_pocket_unique = sorted(parent['money_in_pocket'].unique())
thresh_money_pocket = [(money_pocket_unique[i] + money_pocket_unique[i+1])/2 for i in range(len(money_pocket_unique)-1)]

# Learning how to get the optimum threshold for regression trees

### MSE
thresh28_true = toy_salaries[(toy_salaries['worked_hours'] <=28)]
thresh28_false = toy_salaries[~(toy_salaries['worked_hours'] <=28)]

thresh28_true_mse = ((thresh28_true['salary'] - thresh28_true['salary'].mean())**2).mean()
thresh28_false_mse = ((thresh28_false['salary'] - thresh28_false['salary'].mean())**2).mean()

thresh28_mse = thresh28_false_mse+thresh28_true_mse


thresh37_true = toy_salaries[(toy_salaries['worked_hours'] <=37.5)]
thresh37_false = toy_salaries[~(toy_salaries['worked_hours'] <=37.5)]

thresh37_true_mse = ((thresh37_true['salary'] - thresh37_true['salary'].mean())**2).mean()
thresh37_false_mse = ((thresh37_false['salary'] - thresh37_false['salary'].mean())**2).mean()

thresh37_mse = thresh37_false_mse+thresh37_true_mse

answer = 37.5

### MAE

thresh28_true_mae = abs(thresh28_true['salary'] - thresh28_true['salary'].median()).mean()
thresh28_false_mae = abs(thresh28_false['salary'] - thresh28_false['salary'].median()).mean()
thresh28_mae = thresh28_false_mae+thresh28_true_mae


thresh37_true = toy_salaries[(toy_salaries['worked_hours'] <=37.5)]
thresh37_false = toy_salaries[~(toy_salaries['worked_hours'] <=37.5)]

thresh37_true_mae = abs(thresh37_true['salary'] - thresh37_true['salary'].median()).mean()
thresh37_false_mae = abs(thresh37_false['salary'] - thresh37_false['salary'].median()).mean()

thresh37_mae = thresh37_false_mae+thresh37_true_mae

answer = 37.5

### Gini impurity

parent_gini = 1- ((sum(toy_entrance['entrance'] == 'Allowed')/len(toy_entrance))**2 + (sum(toy_entrance['entrance'] == 'Denied')/len(toy_entrance))**2)

thresh37_true = toy_entrance[(toy_entrance['money_in_pocket'] <=37.5)]
thresh37_false = toy_entrance[~(toy_entrance['money_in_pocket'] <=37.5)]

thresh37_true_gini = 1- ((sum(thresh37_true['entrance'] == 'Allowed')/len(thresh37_true))**2 + (sum(thresh37_true['entrance'] == 'Denied')/len(thresh37_true))**2)

thresh37_false_gini = 1- ((sum(thresh37_false['entrance'] == 'Allowed')/len(thresh37_false))**2 + (sum(thresh37_false['entrance'] == 'Denied')/len(thresh37_false))**2)

thresh37_gini = (len(thresh37_true)/len(toy_entrance))*thresh37_true_gini +(len(thresh37_false)/len(toy_entrance))*thresh37_false_gini



thresh50_true = toy_entrance[(toy_entrance['money_in_pocket'] <=50)]
thresh50_false = toy_entrance[~(toy_entrance['money_in_pocket'] <=50)]

thresh50_true_gini = 1- ((sum(thresh50_true['entrance'] == 'Allowed')/len(thresh50_true))**2 + (sum(thresh50_true['entrance'] == 'Denied')/len(thresh50_true))**2)

thresh50_false_gini = 1- ((sum(thresh50_false['entrance'] == 'Allowed')/len(thresh50_false))**2 + (sum(thresh50_false['entrance'] == 'Denied')/len(thresh50_false))**2)

thresh50_gini = (len(thresh50_true)/len(toy_entrance))*thresh50_true_gini +(len(thresh50_false)/len(toy_entrance))*thresh50_false_gini


answer = 50

### Entropy

parent_entropy = -((sum(toy_entrance['entrance'] == 'Allowed')/len(toy_entrance))*log(sum(toy_entrance['entrance'] == 'Allowed')/len(toy_entrance),2) + (sum(toy_entrance['entrance'] == 'Denied')/len(toy_entrance))*log(sum(toy_entrance['entrance'] == 'Denied')/len(toy_entrance),2))


thresh50_true = toy_entrance[(toy_entrance['money_in_pocket'] <=50)]
thresh50_false = toy_entrance[~(toy_entrance['money_in_pocket'] <=50)]

thresh50_true_entropy = -((sum(thresh50_true['entrance'] == 'Denied')/len(thresh50_true))*log(sum(thresh50_true['entrance'] == 'Denied')/len(thresh50_true),2))

thresh50_false_entropy = -((sum(thresh50_false['entrance'] == 'Allowed')/len(thresh50_false))*log(sum(thresh50_false['entrance'] == 'Allowed')/len(thresh50_false),2) + (sum(thresh50_false['entrance'] == 'Denied')/len(thresh50_false))*log(sum(thresh50_false['entrance'] == 'Denied')/len(thresh50_false),2))

thresh50_info_gain = parent_entropy- (len(thresh50_false)/len(toy_entrance))*thresh50_false_entropy



thresh62_true = toy_entrance[(toy_entrance['money_in_pocket'] <=62.5)]
thresh62_false = toy_entrance[~(toy_entrance['money_in_pocket'] <=62.5)]

thresh62_true_entropy = -((sum(thresh62_true['entrance'] == 'Denied')/len(thresh62_true))*log(sum(thresh62_true['entrance'] == 'Denied')/len(thresh62_true),2)+ (sum(thresh62_true['entrance'] == 'Allowed')/len(thresh62_true))*log(sum(thresh62_true['entrance'] == 'Allowed')/len(thresh62_true),2))

thresh62_false_entropy = -((sum(thresh62_false['entrance'] == 'Allowed')/len(thresh62_false))*log(sum(thresh62_false['entrance'] == 'Allowed')/len(thresh62_false),2))

thresh62_info_gain = parent_entropy- (len(thresh62_true)/len(toy_entrance))*thresh62_true_entropy

answer = 50

