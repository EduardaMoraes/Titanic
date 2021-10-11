
from IPython import get_ipython
#============================================================
# # Solving Kaggle's Titanic challenge with Machine Learning
#============================================================

import pandas as pd
import numpy as np

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')


train.head()

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


modelo = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=0)

variaveis = ['Sex_binario', 'Age']

train['Sex'].value_counts()


def transformar_sexo(valor):
    if valor == 'female':
        return 1
    else:
        return 0

train['Sex_binario'] = train['Sex'].map(transformar_sexo)

train.head()


variaveis = ['Sex_binario', 'Age']


X = train[variaveis].fillna(-1)
y = train['Survived']

X.head()
y.head()

X = X.fillna(-1)


modelo.fit(X, y)

test['Sex_binario'] = test['Sex'].map(transformar_sexo)

X_prev = test[variaveis]
X_prev = X_prev.fillna(-1)
X_prev.head()


p = modelo.predict(X_prev)
p

test.head()


np.random.seed(0)
X_treino, X_valid, y_treino, y_valid = train_test_split(X, y, test_size=0.5)

X_treino.head()

X_treino.shape, X_valid.shape, y_treino.shape, y_valid.shape

modelo = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=0)
modelo.fit(X_treino, y_treino)

p = modelo.predict(X_valid)
np.mean(y_valid == p)

p = (X_valid['Sex_binario'] == 1).astype(np.int64)
np.mean(y_valid == p)


#===========================================================================
# # Create submission
#===========================================================================

sub = pd.Series(p, index=test['PassengerId'], name='Survived')
sub.shape


sub.to_csv('primeiro_modelo.csv', header=True)
get_ipython().system('head -n10 primeiro_modelo.csv')


#==========================================================================
# # Cross Validation
#==========================================================================

from sklearn.model_selection import KFold
from sklearn.model_selection import RepeatedKFold


resultados = []
kf = RepeatedKFold(n_splits=2, n_repeats=10, random_state=10)
    
for linhas_treino, linhas_valid in kf.split(X):
    print('Treino:', linhas_treino.shape[0])
    print('Valid:', linhas_valid.shape[0])

    X_treino, X_valid = X.iloc[linhas_treino], X.iloc[linhas_valid]
    y_treino, y_valid = y.iloc[linhas_treino], y.iloc[linhas_valid]

    modelo = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=0)
    modelo.fit(X_treino, y_treino)

    p = modelo.predict(X_valid)

    acc = np.mean(y_valid == p)
    resultados.append(acc)
    print('Acc:', acc)
    print()



get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('pylab', 'inline')


pylab.hist(resultados)

resultados
np.mean(resultados)


#========================================================
# # New variables
#========================================================

# Previous model = 0.759601451100922


train.head()

variaveis = ['Sex_binario', 'Age', 'Pclass', 'SibSp', 'Parch', 'Fare']


X = train[variaveis].fillna(-1)
y = train['Survived']


resultados = []
kf = RepeatedKFold(n_splits=2, n_repeats=10, random_state=10)
    
for linhas_treino, linhas_valid in kf.split(X):
    print('Treino:', linhas_treino.shape[0])
    print('Valid:', linhas_valid.shape[0])

    X_treino, X_valid = X.iloc[linhas_treino], X.iloc[linhas_valid]
    y_treino, y_valid = y.iloc[linhas_treino], y.iloc[linhas_valid]

    modelo = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=0)
    modelo.fit(X_treino, y_treino)

    p = modelo.predict(X_valid)

    acc = np.mean(y_valid == p)
    resultados.append(acc)
    print('Acc:', acc)
    print()



np.mean(resultados)

pylab.hist(resultados)


#==============================================
# # Model retraining
#==============================================

X.head()
y.head()

test[variaveis].head()


modelo = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=0)
modelo.fit(X, y)

p = modelo.predict(test[variaveis].fillna(-1))
p


#===============================================================================
# # Create submission
#===============================================================================

sub = pd.Series(p, index=test['PassengerId'], name='Survived')
sub.shape

sub.to_csv('segundo_modelo.csv', header=True)

get_ipython().system('head -n10 segundo_modelo.csv')


# Previous model = 0.8041457147175896

train.head()

#==============================================================
# # Error analysis
#==============================================================

resultados = []
kf = RepeatedKFold(n_splits=2, n_repeats=10, random_state=10)
    
for linhas_treino, linhas_valid in kf.split(X):
    print('Treino:', linhas_treino.shape[0])
    print('Valid:', linhas_valid.shape[0])

    X_treino, X_valid = X.iloc[linhas_treino], X.iloc[linhas_valid]
    y_treino, y_valid = y.iloc[linhas_treino], y.iloc[linhas_valid]

    modelo = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=0)
    modelo.fit(X_treino, y_treino)

    p = modelo.predict(X_valid)

    acc = np.mean(y_valid == p)
    resultados.append(acc)
    print('Acc:', acc)
    print()



X_valid_check = train.iloc[linhas_valid].copy()
X_valid_check['p'] = p
X_valid_check.head()


X_valid_check.shape


erros = X_valid_check[X_valid_check['Survived'] != X_valid_check['p']]
erros = erros[['PassengerId', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp', 
              'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked', 'Sex_binario', 'p', 'Survived']]
erros.head()


mulheres = erros[erros['Sex'] == 'female']
homens = erros[erros['Sex'] == 'male']


mulheres.sort_values('Survived')

homens.sort_values('Survived')


#==========================================================
# # New variables
#==========================================================

train['Embarked_S'] = (train['Embarked'] == 'S').astype(int)
train['Embarked_C'] = (train['Embarked'] == 'C').astype(int)
#train['Embarked_Q'] = (train['Embarked'] == 'Q').astype(int)

train['Cabine_nula'] = train['Cabin'].isnull().astype(int)

train['Nome_contem_Miss'] = train['Name'].str.contains('Miss').astype(int)
train['Nome_contem_Mrs'] = train['Name'].str.contains('Mrs').astype(int)

train['Nome_contem_Master'] = train['Name'].str.contains('Master').astype(int)
train['Nome_contem_Col'] = train['Name'].str.contains('Col').astype(int)
train['Nome_contem_Major'] = train['Name'].str.contains('Major').astype(int)
train['Nome_contem_Mr'] = train['Name'].str.contains('Mr').astype(int)



variaveis = ['Sex_binario', 'Age', 'Pclass', 'SibSp', 'Parch', 'Fare', 'Embarked_S', 'Embarked_C', 'Cabine_nula', 
             'Nome_contem_Miss', 'Nome_contem_Mrs', 
             'Nome_contem_Master', 'Nome_contem_Col', 'Nome_contem_Major', 'Nome_contem_Mr']

X = train[variaveis].fillna(-1)
y = train['Survived']


from sklearn.linear_model import LogisticRegression


resultados2 = []
kf = RepeatedKFold(n_splits=2, n_repeats=10, random_state=10)
    
for linhas_treino, linhas_valid in kf.split(X):
    print('Treino:', linhas_treino.shape[0])
    print('Valid:', linhas_valid.shape[0])

    X_treino, X_valid = X.iloc[linhas_treino], X.iloc[linhas_valid]
    y_treino, y_valid = y.iloc[linhas_treino], y.iloc[linhas_valid]

    #modelo = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=0)
    modelo = LogisticRegression()
    
    modelo.fit(X_treino, y_treino)

    p = modelo.predict(X_valid)

    acc = np.mean(y_valid == p)
    resultados2.append(acc)
    print('Acc:', acc)
    print()



pylab.hist(resultados2), pylab.hist(resultados,alpha=0.8)



np.mean(resultados2)



test['Embarked_S'] = (test['Embarked'] == 'S').astype(int)
test['Embarked_C'] = (test['Embarked'] == 'C').astype(int)
#test['Embarked_Q'] = (test['Embarked'] == 'Q').astype(int)

test['Cabine_nula'] = (test['Cabin'].isnull()).astype(int)

test['Nome_contem_Miss'] = test['Name'].str.contains('Miss').astype(int)
test['Nome_contem_Mrs'] = test['Name'].str.contains('Mrs').astype(int)

test['Nome_contem_Master'] = test['Name'].str.contains('Master').astype(int)
test['Nome_contem_Col'] = test['Name'].str.contains('Col').astype(int)
test['Nome_contem_Major'] = test['Name'].str.contains('Major').astype(int)
test['Nome_contem_Mr'] = test['Name'].str.contains('Mr').astype(int)

modelo = LogisticRegression()
#modelo = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=0)
modelo.fit(X, y)

p = modelo.predict(test[variaveis].fillna(-1))

#====================================================
# # Create submission
#====================================================

sub = pd.Series(p, index=test['PassengerId'], name='Survived')
sub.shape

sub.to_csv('terceiro_modelo_lr.csv', header=True)


# Conclusion: Better results were achieved when using LogisticRegression, rather than RandomForest

