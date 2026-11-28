from sklearn.preprocessing import OrdinalEncoder
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.tree import DecisionTreeRegressor, export_text
from sklearn.model_selection import train_test_split
from sklearn.metrics import multilabel_confusion_matrix
import matplotlib.pyplot as plt

df = pd.read_csv('student-por.csv', sep = ';')
df = df[['absences', 'health', 'sex', 'internet', 'Mjob', 'Fjob', 'guardian', 'Pstatus', 'famrel', 'G3']]

############################################## For regression trees ###############################################
########## using loc to encode ordinal vars
df.loc[df['health'] == 'Very Bad', 'health'] = 0
df.loc[df['health'] == 'Bad','health'] = 1
df.loc[df['health'] == 'Regular','health'] = 2
df.loc[df['health'] == 'Good','health'] = 3
df.loc[df['health'] == 'Very Good','health'] = 4

########### using ordinal encoder


ordinal_famrel = [["Very Bad", "Bad", "Regular", "Good", "Excellent"]]
df['famrel'] = OrdinalEncoder(categories = ordinal_famrel).fit_transform(df[['famrel']])

########## encoding categorical vars

df["sex"].replace({"M": 0, "F": 1}, inplace=True)
df.rename(columns={"sex": "sex_fem"}, inplace=True)

guard_dummy = pd.get_dummies(df["guardian"], prefix="guardian")

df = pd.concat([df, guard_dummy], axis = 1)
df.drop("guardian", axis = 1, inplace = True)

df['internet'].replace({'no':0,'yes': 1}, inplace = True)

df['Pstatus'].replace({'T': 0, 'A': 1},inplace=True)
df.rename(columns={"Pstatus": "parents_apart"}, inplace = True)

mother_dummies = pd.get_dummies(df['Mjob'], prefix = 'mother')
df = pd.concat([df, mother_dummies], axis = 1)
df.drop('Mjob', axis = 1, inplace = True)

father_dummies = pd.get_dummies(df['Fjob'], prefix = 'father')
df = pd.concat([df, father_dummies], axis = 1)
df.drop('Fjob', axis = 1, inplace = True)

######### using onehotencoder instead
parent_col_trans = make_column_transformer((OneHotEncoder(),['guardian', 'Mjob', 'Fjob']), remainder='passthrough', verbose_feature_names_out= False)

onehot_df = parent_col_trans.fit_transform(df)
df = pd.DataFrame(onehot_df, columns = parent_col_trans.get_feature_names_out())


############################################## For classification trees ###############################################

parent_col_trans = make_column_transformer(
    (OneHotEncoder(), ["guardian", "Mjob", "Fjob"]), 
    remainder="passthrough", 
    verbose_feature_names_out=False)

onehot_df = parent_col_trans.fit_transform(df)

df = pd.DataFrame(onehot_df, columns = parent_col_trans.get_feature_names_out())

df_class = df.copy()

df_class['G3_qual'] = pd.Series(dtype = 'object')
df_class['G3_qual'].mask(df_class['G3'] <= 3.4, 'Poor', inplace = True)
df_class['G3_qual'].mask((df_class['G3'] <= 9.4) &(df_class['G3'] > 3.4), "Weak", inplace = True)
df_class['G3_qual'].mask((df_class['G3'] <= 13.4) &(df_class['G3'] > 9.4), 'Sufficient', inplace = True)
df_class['G3_qual'].mask((df_class['G3'] <= 15.4) &(df_class['G3'] > 13.4), 'Good', inplace = True)
df_class['G3_qual'].mask((df_class['G3'] <= 17.4) &(df_class['G3'] > 15.4), 'Very Good', inplace = True)
df_class['G3_qual'].mask(df_class['G3'] > 17.4, 'Excellent', inplace = True)

df_class.drop('G3', axis= 1, inplace=True)

################## building the decision tree

############################# regression
X = df.drop(['G3'], axis=1)
y = df['G3']

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3, shuffle= True, random_state=24)

reg_tree_mse = DecisionTreeRegressor(criterion="squared_error", max_depth=3, random_state=24)
reg_tree_mse.fit(X_train,y_train)

y_pred = reg_tree_mse.predict(X_test).round()

exported_reg_tree = export_text(reg_tree_mse, feature_names = list(X.columns))
print(exported_reg_tree)


############################# classification
class_tree_gini = DecisionTreeClassifier(criterion="gini", max_depth=3, random_state=24)
class_tree_gini.fit(X_train, y_train)

plt.figure(figsize= [24.0,8])

_ = plot_tree(class_tree_gini, filled = True, proportion=False, class_names = class_tree_gini.classes_, feature_names = X.columns, precision=3, fontsize= 11, rounded=True)  

############################################## optimising ##############################################



toy_df = pd.DataFrame(data={"y_pred": ["G", "G", "R", "R", "PG"],
                            "y_test": ["G", "PG", "R", "R", "R"]}
                     )

print(toy_df)

df_accuracy = sum(toy_df['y_pred']==toy_df['y_test'])/len(toy_df)

multimatrix = multilabel_confusion_matrix(toy_df['y_test'],toy_df['y_pred'], labels=['G', 'PG', 'R'])

unpruned_tree = DecisionTreeClassifier(random_state = 24)
unpruned_tree.fit(X_train, y_train)

score_unpruned_train = unpruned_tree.score(X_train,y_train)
score_unpruned_test = unpruned_tree.score(X_test, y_test)

pruned_tree = DecisionTreeClassifier(random_state = 24, max_depth = 3)
pruned_tree.fit(X_train, y_train)

score_pruned_train = pruned_tree.score(X_train, y_train)
score_pruned_test = pruned_tree.score(X_test, y_test)

plt.figure(figsize=(11.0,10))
plt.barh(reg_tree.feature_names_in_, reg_tree.feature_importances_)
plt.title('Feature Importance for Students Regression Tree')
plt.ylabel('Column')
plt.xlabel('Importance (0 - 1)')


class_tree = DecisionTreeClassifier(max_depth= 4, min_samples_split= 50, max_features = 30, min_samples_leaf = 4, random_state = 24)

class_tree.fit(X_train, y_train)
score = class_tree.score(X_test, y_test)


unpruned_tree = DecisionTreeRegressor(random_state=14)

cost_path = unpruned_tree.cost_complexity_pruning_path(X_train, y_train)

subtrees = []
for alpha in cost_path.ccp_alphas:
    subtree = DecisionTreeRegressor(random_state = 14, ccp_alpha = alpha)
    subtree.fit(X_train, y_train)
    subtrees.append(subtree)
    
ccp_alphas_slice = cost_path.ccp_alphas[-10:-3]
subtrees_slice = subtrees[-10:-3]

subtree_scores = [x.score(X_test,y_test) for x in subtrees_slice]

plt.subplots(figsize=(8,6))
plt.scatter(ccp_alphas_slice, subtree_scores, marker='o')
plt.xlabel('Alpha')
plt.ylabel('Score')
plt.title('Score per Alpha Value')

alpha_scores = tuple(zip(ccp_alphas_slice, subtree_scores))
