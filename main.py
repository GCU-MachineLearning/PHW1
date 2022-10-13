import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt

def data_processor(data, k, scaler_type='standard', regress_model='decision_tree', regress_parameter=None, is_stratified=False):
    # First, split target and data
    if regress_parameter is None:
        # The regress_model should be decision_tree_g or decision_tree_e
        # so, check the regress_model if it is decision_tree_g or decision_tree_e
        # else, raise ValueError
        if regress_model != 'decision_tree_g' and regress_model != 'decision_tree_e':
            raise ValueError('regress_parameter must be a list of parameters')

    X, y = data[:, :-1], data[:, -1]
    y = y.astype('int')

    # Second, scale the data using data_scaler
    X = data_scaler(X, scaler_type=scaler_type)

    # Third, split the data using KFold, or StratifiedKFold
    X_train, X_test, y_train, y_test = split_k_fold(X, y, n_splits=k, is_stratified=is_stratified)

    # Lastly, regress the data using data_regression
    trained_model = data_regression(X_train, y_train, method=regress_model, regress_parameter=regress_parameter)

    y_pred = trained_model.predict(X_test)

    # Use Score mthos to get the accuracy of model
    score = trained_model.score(X_test, y_test)

    return y_pred, score, [scaler_type, regress_model, regress_parameter, is_stratified, k]

# I: split the data using sklearn KFold, stratifiedKFold
# return: the train and test data
def split_k_fold(data, target, n_splits=5, random_state=0, is_stratified=False):
    from sklearn.model_selection import KFold, StratifiedKFold

    x_train, x_test, y_train, y_test = None, None, None, None

    if is_stratified:
        skf = StratifiedKFold(n_splits=n_splits, random_state=random_state, shuffle=True)
        for train_index, test_index in skf.split(data, target):
            x_train, x_test = data[train_index], data[test_index]
            y_train, y_test = target[train_index], target[test_index]
    else:
        kf = KFold(n_splits=n_splits, random_state=random_state, shuffle=True)
        for train_index, test_index in kf.split(data):
            x_train, x_test = data[train_index], data[test_index]
            y_train, y_test = target[train_index], target[test_index]

    assert x_train is not None and x_test is not None \
           and y_train is not None and y_test is not None
    return x_train, x_test, y_train, y_test


# II: data scaler using sklearn StandardScaler, MinMaxScaler and RobustScaler
# return: the scaled data
def data_scaler(data, scaler_type='standard'):
    from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
    if scaler_type == 'standard':
        scaler = StandardScaler()

    elif scaler_type == 'minmax':
        scaler = MinMaxScaler()

    elif scaler_type == 'robust':
        scaler = RobustScaler()

    else:
        raise ValueError('scaler_type must be standard, minmax or robust')

    return scaler.fit_transform(data)


# III: data regression methods function using sklearn DecisionTreeRegressor with entropy,
# DecisionTreeRegressor with Gini, LogisticRegression and SVM
# return: trained_model
def data_regression(data, target, method='decision_tree_g', regress_parameter=None):
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVR

    if method == 'decision_tree_g':
        model = DecisionTreeClassifier(max_depth=5, criterion='gini', random_state=0)

    elif method == 'decision_tree_e':
        model = DecisionTreeClassifier(max_depth=5, criterion='entropy', random_state=0)

    elif method == 'logistic':
        solver = regress_parameter[0]
        model = LogisticRegression(solver=solver, random_state=0)

    elif method == 'svm':
        kernel = regress_parameter[0]
        C_value = regress_parameter[1]
        model = SVR(kernel=kernel, C=C_value)

    else:
        raise ValueError('method must be decision_tree, logistic or svm')

    model.fit(data, target)

    return model

# main
if __name__ == '__main__':
    # load dataset
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data'
    df = pd.read_csv(url)

    # since it has some random numerical values, rename the dataset's columns
    df.columns = ['Sample code number', 'Clump Thickness', 'Uniformity of Cell Size', 'Uniformity of Cell Shape',
                  'Marginal Adhesion', 'Single Epithelial Cell Size',
                  'Bare Nuclei', 'Bland Chromatin', 'Normal Nucleoli', 'Mitoses', 'Class']

    # check if there's a feature that has null values
    df.isnull().sum()

    # check if there are any dirty data which cannot be converted to float, drop it
    df = df[df['Bare Nuclei'] != '?']

    # since it has no correlation with class variable, drop 'Sample code number' column
    df.drop(df.columns[0], axis=1, inplace=True)

    # create a correlation matrix and show it
    corrMatrix = df.corr()
    plt.figure(figsize=(9, 7))
    sn.heatmap(corrMatrix, annot=True)
    plt.show()

    # Corr Matrix says 'Mistoses' is least correlated with class varibale, drop the 'Mistoses' column as well.
    df.drop(df.columns[8], axis=1, inplace=True)
    df.head(10)

    # Let's find the top 5 components that has the highest accuracy,
    # using all possible combinations by brute force.
    scalers = ['standard', 'minmax', 'robust']
    regressors = ['decision_tree_e', 'decision_tree_g', 'svm', 'logistic']

    # lots of parameters, so I'll use multiprocessing to speed up the process
    logistic_solver = ['lbfgs', 'liblinear', 'sag', 'saga']
    svm_kernels = ['linear', 'poly', 'rbf', 'sigmoid']
    svm_C = [0.1, 1, 10, 100]

    is_stratified = [True, False]
    k_lst = [2, 3, 4, 5, 6, 7]

    combinations = []  # total combinations: 3 * 4 * 2 * 6 * (5 + 25) = 1'800
    for scaler in scalers:  # 3
        for regressor in regressors:  # 4
            for strat in is_stratified:  # 2
                for k in k_lst:  # 6
                    if regressor == 'logistic':  # 5
                        for solver in logistic_solver:
                            _, accuracy, combination = data_processor(df.values, k, scaler_type=scaler,
                                                                      regress_model=regressor,
                                                                      regress_parameter=[solver], is_stratified=strat)
                            combinations.append((accuracy, combination))
                    elif regressor == 'svm':  # 20
                        for kernel in svm_kernels:
                            for c in svm_C:
                                _, accuracy, combination = data_processor(df.values, k, scaler_type=scaler,
                                                                          regress_model=regressor,
                                                                          regress_parameter=[kernel, c],
                                                                          is_stratified=strat)
                                combinations.append((accuracy, combination))
                    else:  # 4
                        _, accuracy, combination = data_processor(df.values, k, scaler_type=scaler,
                                                                  regress_model=regressor,
                                                                  is_stratified=strat)
                        combinations.append((accuracy, combination))

    # sort the combinations by accuracy
    combinations.sort(key=lambda x: x[0], reverse=True)

    combinations[:100]

    # plot result
    import matplotlib.pyplot as plt

    plt.rc('font', size=20)
    plt.rc('axes', labelsize=15)
    plt.rc('xtick', labelsize=10)
    plt.rc('ytick', labelsize=10)
    plt.rc('figure', titlesize=30)

    # print Top 5 Combinations list
    print("[Top 5 Combinations]")
    print()

    # plot top 5 combinations
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111)
    plt.title("Top 5 Combinations")

    for i in range(5):
        scaler_type = combinations[i][1][0]
        regress_model = combinations[i][1][1]

        if (combinations[i][1][3]):
            is_stratified = 'Stratified KFold'
        else:
            is_stratified = 'KFold'

        k = combinations[i][1][4]

        print("Top", i + 1)
        print("Scaling method :", scaler_type)
        # print used regression model with its parameters
        if regress_model == 'logistic':
            regress_parameter = combinations[i][1][2]
            print("Regression model : Logistic Regression")
            print("Solver :", regress_parameter[0])
        elif regress_model == 'svm':
            regress_parameter = combinations[i][1][2]
            print("Regression model : SVM")
            print("Kernel :", regress_parameter[0])
            print("C :", regress_parameter[1])
        else:
            print("Regression model :", regress_model)
        print("Using", is_stratified, ", k = ", k)
        print()

        ax.scatter(k, combinations[i][0], c='b', marker='o')
        ax.text(k, combinations[i][0], scaler_type + ' ' + regress_model + ' ' + is_stratified)

    ax.set_xlabel('k')
    ax.set_ylabel('accuracy')
    plt.show()

