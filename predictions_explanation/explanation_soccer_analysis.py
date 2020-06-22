from skater import Interpretation
from skater.model import InMemoryModel
import pandas as pd
from mord import LogisticAT
from sklearn import preprocessing
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.metrics import make_scorer
from sklearn.metrics import accuracy_score
from scipy.stats import ks_2samp
from sklearn.metrics import r2_score
from scipy.stats import pearsonr
from collections import Counter
from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_val_predict, RandomizedSearchCV
import math
from joblib import dump
import shap
import pickle


import numpy as np
from keras.wrappers.scikit_learn import KerasClassifier
from keras.models import Sequential
from keras.layers import Dense, Dropout
from xgboost import XGBClassifier
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# import the regressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split



listMarks = ['fantacalcio_score']
listRoles = ['C', 'A', 'D', 'P']
toRemoveRatings = ['ratings', 'ratings_3', 'ratings_5', 'ratings_7', 'ratings_9', 'ratings_total_alpha']
toRemoveWithoutContextual = ['big_match', 'winner', 'contextual_h/a', 'contextual_team_goal', 'contextual_goal_suffered',
                            'contextual_goal_difference', 'contextual_club_name', 'contextual_against_club_name', 'contextual_age',
                            'contextual_expecatition']
toRemove = ['match_id', 'match_day', 'team_id', 'player_name_fantacalcio', 'player_role_fantacalcio',
           'player_role_newspaper', 'player_name_newspaper', 'corriere_score', 'tuttosport_score', 'fantacalcio_score','gazzetta_score', 'team',
           'match_day_newspaper', 'match_day_fanta', 'player_id']
ratings = ['ratings', 'ratings_3', 'ratings_5', 'ratings_7', 'ratings_9', 'ratings_total_alpha']
select_features_for_radar = ['contextual_goal_difference', 'contextual_team_goal', 'contextual_goal_suffered', 'winner', 'contextual_h/a',
                            'contextual_age', 'country', 'contextual_expecatition', 'big_match', 'contextual_club_name', 'x_individual_mean_center_of_gravity',
                            'y_individual_mean_center_of_gravity', 'successful_action_partecipation', 'total_action_partecipation', 'failed_passes',
                            'completed_passes', 'assist_passes', 'failed_cross', 'foul_made', 'failed_cross', 'completed_cross', 'assist_cross', 'complete_headed_duel',
                            'completed_tackels', 'failed_tackels', 'tackle_foul', 'total_clearance', 'failed_clearance', 'catch_goal_keeping', 'failed_catch_goal_keeping',
                            'save_goal_keeping', 'punch_goal_keeping', 'yellow_card', 'red_card', 'wood_worker_shot', 'saved_shot', 'blocked_shot', 'goal_shot',
                            'off_target_shot', 'total_interceptions', 'suffered_foul', 'blame_on_penalty', 'merit_on_penalty', 'suffered_goal', 'failed_penalty',
                            'penalty_saved', 'own_goals', 'ejection']


def general_explanation_using_skater(all_roles_scores, labels_training_set, labels_test_set, df_train_set, df_test_set, alpha):
    '''
    Show the weight that more influenced a decision in eli 5 framework

    ----------------------------------------------------------------
    Params:
        all_roles_score = list of all the marks present in test and train set for each role
        labels_training_set
        labels_test_set
        df_train_set
        df_test_set

    '''
    le = preprocessing.LabelEncoder()
    le.fit(all_roles_scores)
    train_encoded_values = le.transform(labels_training_set)
    test_encoded_values = le.transform(labels_test_set)

    # boost_classifier = XGBClassifier(gamma = gamma, max_depth = maxde, min_child_weight = minchild)
    # boost_classifier.fit(df_train_set, train_encoded_values)

    # predictions = boost_classifier.predict(df_test_set)
    # predictions = predictions.astype('int')

    model_ordinal = LogisticAT(alpha=alpha)
    model_ordinal.fit(df_train_set.values, train_encoded_values)
    predictions = model_ordinal.predict(df_test_set)

    interpreter = Interpretation(df_train_set, feature_names=list(df_train_set.columns))

    model = InMemoryModel(model_ordinal.predict_proba, examples=df_train_set[:10])

    plots = interpreter.feature_importance.feature_importance(model, ascending=True)

    # fig, ax = plt.subplots(figsize=(5,35))
    # plots = interpreter.feature_importance.plot_feature_importance(model, ascending=True, ax= ax)

    return plots


def mae_fun(target_true, target_fit, le):
    target_true = le.inverse_transform(target_true)
    target_fit = le.inverse_transform(target_fit)
    return mean_absolute_error(target_true, target_fit)


def acc_fun(target_true, target_fit):
    target_fit = np.round(target_fit)
    target_fit.astype('int')
    return accuracy_score(target_true, target_fit)


def ks_fun(target_true, target_fit, le):
    target_true = le.inverse_transform(target_true)
    target_fit = np.round(target_fit, 0)
    target_fit = target_fit.astype('int')
    target_fit = le.inverse_transform(target_fit)
    ks = ks_2samp(target_true, target_fit)[0]
    return round(ks, 2)


def r2_fun(target_true, target_fit):
    target_fit = np.round(target_fit)
    target_fit.astype('int')
    return r2_score(target_true, target_fit)


def pearsonr_fun(target_true, target_fit):
    target_fit = np.round(target_fit)
    target_fit.astype('int')
    return pearsonr(target_true, target_fit)[0]


def trainordinalregressor(df, listMarks, listRoles, path):
    """
    Train a mord ordinal regression model

    Parameters
    ----------
    df : the dataset of player features.
    listMarks: the name of the field of the marks we want to train the different model
    listRoles: list of roles name ['A', 'C', 'D', 'P']
    path: where to store the models

    Returns
    -------
    results dictionaries:
        distributionPerNewspaper: a dictionary that for each newspaper has the true vales and predicted values
        resultPerRole: a dictionary that syntetize for each newspaper for each roles some predictions metrics
    """
    progress = 0
    resultPerRole = {}
    distributionPerNewspaper = {}
    # for each newspaper
    for newspaper in listMarks:
        distributionPerNewspaper[newspaper] = {}
        distributionPerNewspaper[newspaper]['true'] = []
        distributionPerNewspaper[newspaper]['pred'] = []
        # for each role
        for role in listRoles:
            progress += 1
            if (newspaper != 'fantacalcio_score'):
                subDF = df[df['player_role_newspaper'] == role]
            else:
                subDF = df[df['player_role_fantacalcio'] == role]
            # extract and transfrom categorical values
            le_teams = preprocessing.LabelEncoder()
            subDF['contextual_against_club_name'] = le_teams.fit_transform(subDF['contextual_against_club_name'])
            subDF['contextual_club_name'] = le_teams.transform(subDF['contextual_club_name'])
            le_country = preprocessing.LabelEncoder()
            subDF['country'] = le_country.fit_transform(subDF['country'])

            if (newspaper == 'corriere_score'):
                subDF = subDF[subDF['corriere_score'] != 10]
            if (newspaper == 'corriere_score' and role == 'D'):
                subDF = subDF[subDF['corriere_score'] != 8]
                subDF = subDF[subDF['corriere_score'] != 3.5]
            if (newspaper == 'corriere_score' and role == 'P'):
                subDF = subDF[subDF['corriere_score'] != 9]

            # check the size of the labels
            # vc = subDF[newspaper].value_counts()
            # indexes = vc[vc < n_min].index
            # subDF.drop(indexes, inplace=True)

            # ectract and encode labels
            le = preprocessing.LabelEncoder()
            labels = subDF[newspaper]
            le.fit(subDF[newspaper])
            labels = le.transform(labels)
            myset = set(labels)

            for el in toRemove:
                del subDF[el]

            # uncomment to train without contextual variables
            # for el in toRemoveWithoutContextual:
            # del subDF[el]

            # uncomment to train without ratings variables
            for el in toRemoveRatings:
                del subDF[el]

            # uncommentforonly contextual variables
            # subDF = subDF[toRemoveWithoutContextual]

            stringMatch = newspaper + '_' + role

            resultPerRole[stringMatch] = {}

            # remove player rank values for goalkeeper
            # if(role == 'P'):
            #   for rat in ratings:
            #        del subDF[rat]
            print(stringMatch)

            # rescale the robust scaler
            robust = preprocessing.RobustScaler()
            robust.fit(subDF)
            subDF = robust.transform(subDF)

            # splitting
            X_train, X_test, y_train, y_test = train_test_split(subDF, labels, random_state=17)

            # declare ordinal regressor
            model_ordinal = LogisticAT()  # alpha parameter set to zero to perform no regularisation
            seed = 17

            # kfold definition
            kfold = StratifiedKFold(n_splits=2, shuffle=True, random_state=seed)
            features = subDF
            target = labels
            # rscore
            rscore = make_scorer(pearsonr_fun, greater_is_better=True)
            # OUR OBJECTIVE IS TO INCREASE THE R SCORE
            # define the grid search
            svr = GridSearchCV(model_ordinal,
                               scoring=rscore,
                               cv=kfold,
                               param_grid={'alpha': [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]}, verbose=1)

            svr.fit(features, target)

            print("Best Score: {}".format(svr.best_score_))
            print("Best params: {}".format(svr.best_params_))

            resultPerRole[stringMatch]['r'] = svr.best_score_

            model_ordinal = LogisticAT(alpha=svr.best_params_['alpha'])

            y_pred = cross_val_predict(model_ordinal, features, target, cv=kfold)

            resultPerRole[stringMatch]['RSME'] = math.sqrt(
                mean_squared_error(le.inverse_transform(y_pred), le.inverse_transform(target)))
            resultPerRole[stringMatch]['Accuracy'] = acc_fun(target, y_pred)
            resultPerRole[stringMatch]['KS'] = ks_fun(target, y_pred, le)
            resultPerRole[stringMatch]['r2'] = r2_fun(target, y_pred)

            le_name_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
            print(le_name_mapping)

            print(Counter(target).keys())
            print(Counter(target).values())

            path_inserted = path+'mord_' + role + '.joblib'
            dump(model_ordinal, path_inserted)

            distributionPerNewspaper[newspaper]['true'].append(le.inverse_transform(target))
            distributionPerNewspaper[newspaper]['pred'].append(le.inverse_transform(y_pred))
    return distributionPerNewspaper, resultPerRole


def createNeuralNetworkAll(output_classes, optimizer, rate, activation, init_mode):
    model = Sequential()
    model.add(Dense(128, input_dim=154, kernel_initializer=init_mode, activation=activation))
    model.add(Dropout(rate=rate))
    model.add(Dense(64, kernel_initializer=init_mode, activation=activation))
    model.add(Dropout(rate=rate))
    model.add(Dense(64, kernel_initializer=init_mode, activation=activation))
    model.add(Dense(output_classes, activation='softmax'))
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model


def createNeuralNetworkGK(output_classes, optimizer, rate, activation, init_mode):
    model = Sequential()
    model.add(Dense(128, input_dim=148, kernel_initializer=init_mode, activation=activation))
    model.add(Dropout(rate=rate))
    model.add(Dense(64, kernel_initializer=init_mode, activation=activation))
    model.add(Dropout(rate=rate))
    model.add(Dense(64, kernel_initializer=init_mode, activation=activation))
    model.add(Dense(output_classes, activation='softmax'))
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model


def neuralnetwork_model(df, listMarks, listRoles, path):
    """
    Train Neural Network Model to predict player marks

    Parameters
    ----------
    df : the dataset of player features.
    listMarks: the name of the field of the marks we want to train the different model
    listRoles: list of roles name ['A', 'C', 'D', 'P']
    path: where to store the models

    Returns
    -------
    results dictionaries:
        distributionPerNewspaper: a dictionary that for each newspaper has the true vales and predicted values
        results: a dictionary that syntetize for each newspaper for each roles some predictions metrics
    """
    forNeural = df.copy(deep=True)
    results = {}
    distributionPerNewspaper = {}
    # for each newspaper
    for newspaper in listMarks:
        distributionPerNewspaper[newspaper] = {}
        distributionPerNewspaper[newspaper]['true'] = []
        distributionPerNewspaper[newspaper]['pred'] = []
        # for each role
        for role in listRoles:
            if (newspaper != 'fantacalcio_score'):
                X = forNeural[forNeural['player_role_newspaper'] == role]
            else:
                X = forNeural[forNeural['player_role_fantacalcio'] == role]
            # extract and transfrom categorical values
            le_teams = preprocessing.LabelEncoder()
            X['contextual_against_club_name'] = le_teams.fit_transform(X['contextual_against_club_name'])
            X['contextual_club_name'] = le_teams.transform(X['contextual_club_name'])
            le_country = preprocessing.LabelEncoder()
            X['country'] = le_country.fit_transform(X['country'])

            if (newspaper == 'corriere_score'):
                X = X[X['corriere_score'] != 10]
            if (newspaper == 'corriere_score' and role == 'D'):
                X = X[X['corriere_score'] != 8]
                X = X[X['corriere_score'] != 3.5]
            if (newspaper == 'corriere_score' and role == 'P'):
                X = X[X['corriere_score'] != 9]

            # ectract and encode labels
            le = preprocessing.LabelEncoder()
            y = X[newspaper]
            minimum = min(y)
            maximum = max(y)
            print(minimum)
            print(maximum)
            le.fit(X[newspaper])
            y = le.transform(y)

            for el in toRemove:
                del X[el]

            print(list(le.classes_))
            print(len(list(le.classes_)))

            # uncomment to train without contextual variables
            # for el in toRemoveWithoutContextual:
            #    del subDF[el]

            # uncomment to train without ratings variables
            # for el in toRemoveRatings:
            #    del subDF[el]

            stringMatch = newspaper + '_' + role

            results[stringMatch] = {}

            p = 0
            # remove player rank values for goalkeeper
            if (role == 'P'):
                for rat in ratings:
                    del X[rat]
                p = 1
            print(stringMatch)

            robust = preprocessing.RobustScaler()
            X = robust.fit_transform(X)

            seed = 17

            # HYPERPARAMETHERS TUNING
            if (role != 'P'):
                # create the model
                model = KerasClassifier(createNeuralNetworkAll, verbose=0)
            else:
                model = KerasClassifier(createNeuralNetworkGK, verbose=0)

            # define the grid search parameters
            batch_size = [16, 32]
            epochs = [2, 5, 10, 15, 20]
            optimizer = ['SGD', 'RMSprop', 'Adam']
            dropout_rate = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
            activation = ['softmax', 'softplus', 'softsign', 'relu', 'tanh', 'sigmoid', 'hard_sigmoid', 'linear']
            init_mode = ['uniform', 'lecun_uniform', 'normal', 'zero', 'glorot_normal', 'glorot_uniform', 'he_normal',
                         'he_uniform']
            classes_out = [len(list(le.classes_))]

            param_grid = dict(batch_size=batch_size, epochs=epochs, output_classes=classes_out, optimizer=optimizer,
                              rate=dropout_rate, activation=activation, init_mode=init_mode)
            kfold = StratifiedKFold(n_splits=2, shuffle=True, random_state=seed)

            # Search in action!
            n_iter_search = 16  # Number of parameter settings that are sampled.
            random_search = RandomizedSearchCV(estimator=model,
                                               param_distributions=param_grid,
                                               n_iter=n_iter_search,
                                               n_jobs=1,
                                               cv=kfold,
                                               verbose=1)
            random_search.fit(X, y)

            bestP = random_search.best_params_

            # Show the results
            print("Best: %f using %s" % (random_search.best_score_, random_search.best_params_))

            # CREATE THE MODEL WITH BEST PARAMETERS AND MAKE PREDICTIONS
            kfold = StratifiedKFold(n_splits=2, shuffle=True, random_state=seed)
            for train, test in kfold.split(X, y):
                if (role != 'P'):
                    # create the model
                    model = createNeuralNetworkAll(bestP['output_classes'], bestP['optimizer'], bestP['rate'],
                                                   bestP['activation'], bestP['init_mode'])
                else:
                    model = createNeuralNetworkGK(bestP['output_classes'], bestP['optimizer'], bestP['rate'],
                                                  bestP['activation'], bestP['init_mode'])

                model.fit(X, y, batch_size=bestP['batch_size'], epochs=bestP['epochs'], verbose=1)
                predictions = model.predict(X[test])
                final_y_t = y[test]

            path_inserted = path + 'neural_network_multi_class_' + role + '.joblib'
            dump(model, path_inserted)

            predict_class = np.argmax(predictions, axis=1)
            predict_class = predict_class.tolist()
            y_pred = le.inverse_transform(predict_class)
            y_true = le.inverse_transform(final_y_t)
            print(y_pred)
            print(y_true)
            print(len(y_pred))
            print(len(y_true))
            results[stringMatch]['r'] = pearsonr(y_true, y_pred)[0]
            results[stringMatch]['RSME'] = math.sqrt(mean_squared_error(y_true, y_pred))
            results[stringMatch]['Accuracy'] = accuracy_score(le.transform(y_true), le.transform(y_pred))
            results[stringMatch]['KS'] = ks_2samp(y_true, y_pred)[0]
            results[stringMatch]['r2'] = r2_score(y_true, y_pred)

            distributionPerNewspaper[newspaper]['true'].append(y_true)
            distributionPerNewspaper[newspaper]['pred'].append(y_pred)
    return distributionPerNewspaper, results


def plot_confusion_matrix(cm, classes, normalized=True, cmap='bone'):
    plt.figure(figsize=[7, 6])
    norm_cm = cm
    if normalized:
        norm_cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        sns.heatmap(norm_cm, annot=cm, fmt='g', xticklabels=classes, yticklabels=classes, cmap=cmap)


def compute_r_score(y_true, y_pred):
    return pearsonr(y_true, y_pred)[0]


def xgbboost_train(df, listMarks, listRoles, path):
    """
    Train XGBOOST Model to predict player marks

    Parameters
    ----------
    df : the dataset of player features.
    listMarks: the name of the field of the marks we want to train the different model
    listRoles: list of roles name ['A', 'C', 'D', 'P']
    path: where to store the models

    Returns
    -------
    results dictionaries:
        distributionPerNewspaper: a dictionary that for each newspaper has the true vales and predicted values
        results: a dictionary that syntetize for each newspaper for each roles some predictions metrics
    """

    forNeural = df.copy(deep=True)
    results = {}
    distributionPerNewspaper = {}
    # for each newspaper
    for newspaper in listMarks:
        distributionPerNewspaper[newspaper] = {}
        distributionPerNewspaper[newspaper]['true'] = []
        distributionPerNewspaper[newspaper]['pred'] = []
        # for each role
        for role in listRoles:
            if (newspaper != 'fantacalcio_score'):
                X = forNeural[forNeural['player_role_newspaper'] == role]
            else:
                X = forNeural[forNeural['player_role_fantacalcio'] == role]
            # extract and transfrom categorical values
            le_teams = preprocessing.LabelEncoder()
            X['contextual_against_club_name'] = le_teams.fit_transform(X['contextual_against_club_name'])
            X['contextual_club_name'] = le_teams.transform(X['contextual_club_name'])
            le_country = preprocessing.LabelEncoder()
            X['country'] = le_country.fit_transform(X['country'])

            if (newspaper == 'corriere_score'):
                X = X[X['corriere_score'] != 10]
            if (newspaper == 'corriere_score' and role == 'D'):
                X = X[X['corriere_score'] != 8]
                X = X[X['corriere_score'] != 3.5]
            if (newspaper == 'corriere_score' and role == 'P'):
                X = X[X['corriere_score'] != 9]

            # ectract and encode labels
            le = preprocessing.LabelEncoder()
            y = X[newspaper]
            le.fit(X[newspaper])
            y = le.transform(y)

            X.reset_index(drop=True, inplace=True)

            y = pd.DataFrame(data=y, columns=['marks'])
            values = y['marks'].value_counts().keys().tolist()
            counts = y['marks'].value_counts().tolist()
            valueToRemove = []
            for el in range(0, len(counts)):
                if (counts[el] < 2):
                    valueToRemove.append(values[el])
            indextoremove = []
            if (len(valueToRemove) > 0):
                for index, el in y.iterrows():
                    if (el[0] in valueToRemove):
                        indextoremove.append(index)

            X = X.drop(indextoremove)
            y = y.drop(indextoremove)

            y['marks'] = le.inverse_transform(y['marks'])

            # ectract and encode labels
            le = preprocessing.LabelEncoder()
            y = y['marks']
            le.fit(y)
            y = le.transform(y)

            for el in toRemove:
                del X[el]

            print(list(le.classes_))
            print(len(list(le.classes_)))

            # uncomment to train without contextual variables
            # for el in toRemoveWithoutContextual:
            #    del subDF[el]

            # uncomment to train without ratings variables
            # for el in toRemoveRatings:
            #    del subDF[el]

            stringMatch = newspaper + '_' + role

            results[stringMatch] = {}

            p = 0
            # remove player rank values for goalkeeper
            if (role == 'P'):
                for rat in ratings:
                    del X[rat]
                p = 1
            print(stringMatch)

            robust = RobustScaler()
            robust.fit(X)
            X = robust.transform(X)

            # splitting
            X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=17, stratify=y)

            '''
            #create DMatrix xgboost type
            dtrain = xgb.DMatrix(data=X_train, label=y_train)
            dtest = xgb.DMatrix(data=X_test)

            params = {'max_depth': 6,
                      'objective': 'multi:softmax',  # error evaluation for multiclass training
                      'num_class': len(list(le.classes_)),
                      'n_gpus': 0
                     }
            bst = xgb.train(params, dtrain)
            pred = bst.predict(dtest)
            pred = pred.astype('int')
            listr = [str(el) for el in list(le.classes_)]
            print(listr)
            print(len(listr))
            print(classification_report(y_test, pred))
            frog_cm = confusion_matrix(y_test, pred)
            plot_confusion_matrix(frog_cm, classes=listr)
            '''

            r = make_scorer(compute_r_score, greater_is_better=True)

            param_test1 = {'max_depth': range(3, 10, 2),
                           'min_child_weight': range(1, 6, 2),
                           'gamma': [i / 10.0 for i in range(0, 5)]
                           }
            gsearch1 = GridSearchCV(estimator=XGBClassifier(learning_rate=0.1, n_estimators=140, max_depth=5,
                                                            min_child_weight=1, gamma=0, subsample=0.8,
                                                            colsample_bytree=0.8,
                                                            objective='multi:softmax', seed=27),
                                    param_grid=param_test1, scoring=r, n_jobs=-1, iid=False, cv=2)
            gsearch1.fit(X_train, y_train)
            print(gsearch1.best_params_)
            print(gsearch1.best_score_)

            path_inserted = path + 'xgboost_multi_class_' + role + '.joblib'
            dump(gsearch1.best_estimator_, path_inserted)

            pred = gsearch1.predict(X_test)
            pred = pred.astype('int')
            listr = [str(el) for el in list(le.classes_)]
            print(listr)
            print(len(listr))
            print(classification_report(y_test, pred))

            results[stringMatch]['r'] = pearsonr(le.inverse_transform(y_test), le.inverse_transform(pred))[0]
            results[stringMatch]['RSME'] = math.sqrt(
                mean_squared_error(le.inverse_transform(y_test), le.inverse_transform(pred)))
            results[stringMatch]['Accuracy'] = accuracy_score(y_test, pred)
            results[stringMatch]['KS'] = ks_2samp(le.inverse_transform(y_test), le.inverse_transform(pred))[0]
            results[stringMatch]['r2'] = r2_score(le.inverse_transform(y_test), le.inverse_transform(pred))

            distributionPerNewspaper[newspaper]['true'].append(le.inverse_transform(y_test))
            distributionPerNewspaper[newspaper]['pred'].append(le.inverse_transform(pred))
    return distributionPerNewspaper, results


def compute_r_score(y_true, y_pred):
    return pearsonr(y_true, y_pred)[0]


def decision_tree_regressor_trainer(df, listMarks, listRoles, path):
    """
    Train Decision Tree Regressor to predict player marks

    Parameters
    ----------
    df : the dataset of player features.
    listMarks: the name of the field of the marks we want to train the different model
    listRoles: list of roles name ['A', 'C', 'D', 'P']
    path: math where to save models

    Returns
    -------
    results dictionaries:
        distributionPerNewspaper: a dictionary that for each newspaper has the true vales and predicted values
        resultPerRole: a dictionary that syntetize for each newspaper for each roles some predictions metrics
    """
    decisionTreeRegre = df.copy(deep=True)
    results = {}
    distributionPerNewspaper = {}
    # for each newspaper
    for newspaper in listMarks:
        distributionPerNewspaper[newspaper] = {}
        distributionPerNewspaper[newspaper]['true'] = []
        distributionPerNewspaper[newspaper]['pred'] = []
        # for each role
        for role in listRoles:
            if (newspaper != 'fantacalcio_score'):
                X = decisionTreeRegre[decisionTreeRegre['player_role_newspaper'] == role]
            else:
                X = decisionTreeRegre[decisionTreeRegre['player_role_fantacalcio'] == role]
            # extract and transfrom categorical values
            le_teams = preprocessing.LabelEncoder()
            X['contextual_against_club_name'] = le_teams.fit_transform(X['contextual_against_club_name'])
            X['contextual_club_name'] = le_teams.transform(X['contextual_club_name'])
            le_country = preprocessing.LabelEncoder()
            X['country'] = le_country.fit_transform(X['country'])

            if (newspaper == 'corriere_score'):
                X = X[X['corriere_score'] != 10]
            if (newspaper == 'corriere_score' and role == 'D'):
                X = X[X['corriere_score'] != 8]
                X = X[X['corriere_score'] != 3.5]
            if (newspaper == 'corriere_score' and role == 'P'):
                X = X[X['corriere_score'] != 9]

            # remove those marks that are too few in order to make a regression
            y = X[newspaper]

            for el in toRemove:
                del X[el]

            stringMatch = newspaper + '_' + role

            results[stringMatch] = {}

            p = 0
            # remove player rank values for goalkeeper
            if (role == 'P'):
                for rat in ratings:
                    del X[rat]
                p = 1
            print(stringMatch)

            # apply robust scaler to data
            robust = RobustScaler()
            robust.fit(X)
            X = robust.transform(X)

            print(len(y))

            # splitting
            X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=17)

            param_grid = {'max_depth': range(2, 16, 2),
                          'min_samples_split': range(2, 16, 2)}

            # create a regressor object
            regressor = DecisionTreeRegressor()

            r = make_scorer(compute_r_score, greater_is_better=True)

            clf = GridSearchCV(regressor,
                               param_grid,
                               scoring=r,
                               cv=5, n_jobs=1, verbose=1)
            _ = clf.fit(X_train, y_train)

            path_inserted = path + 'decision_tree_regressor_multi_class_' + role + '.joblib'
            dump(clf.best_estimator_, path_inserted)

            print(clf.best_params_)
            print(clf.best_score_)

            y_pred = clf.predict(X_test)
            y_true = y_test
            dat = pd.DataFrame()
            dat['y'] = y_pred
            y_pred = dat.y.mul(2).round().div(2)

            le = preprocessing.LabelEncoder()
            le.fit(y_true)

            results[stringMatch]['r'] = pearsonr(y_true, y_pred)[0]
            results[stringMatch]['RSME'] = math.sqrt(mean_squared_error(y_true, y_pred))
            results[stringMatch]['Accuracy'] = accuracy_score(le.transform(y_true), le.transform(y_pred))
            results[stringMatch]['KS'] = ks_2samp(y_true, y_pred)[0]
            results[stringMatch]['r2'] = r2_score(y_true, y_pred)

            distributionPerNewspaper[newspaper]['true'].append(y_test)
            distributionPerNewspaper[newspaper]['pred'].append(y_pred)
    return distributionPerNewspaper, results


def plot_radar_chart(features, colors, position):
    '''
    Create a radar chart plot

    Params:
            features - list of features importance in which in index there is the attribute name of the
    '''

    new_features = []
    new_indexes = []
    access = 0
    for ind in features.index:
        if (ind in select_features_for_radar):
            new_indexes.append(ind)
            new_features.append(features[access])
        access += 1

    max_ele = round(max(new_features) - 0.02, 2)
    # print(max_ele)

    # Each attribute we'll plot in the radar chart.
    labels = new_indexes

    # Let's look at the 1970 Chevy Impala and plot it.
    values = new_features

    # Number of variables we're plotting.
    num_vars = len(labels)

    # Split the circle into even parts and save the angles
    # so we know where to put each axis.
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()

    # The plot is a circle, so we need to "complete the loop"
    # and append the start value to the end.
    values += values[:1]
    angles += angles[:1]

    # ax = plt.subplot(polar=True)
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))

    # Draw the outline of our data.
    ax.plot(angles, values, color=colors, linewidth=2)
    # Fill it in.
    ax.fill(angles, values, color=colors, alpha=0.5)

    # Fix axis to go in the right order and start at 12 o'clock.
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    ax.annotate(position, (2, max_ele), fontsize=15)

    # Draw axis lines for each angle and label.
    ax.set_thetagrids(np.degrees(angles), labels, fontsize=8)

    # Go through labels and adjust alignment based on where
    # it is in the circle.
    for label, angle in zip(ax.get_xticklabels(), angles):
        if angle in (0, np.pi):
            label.set_horizontalalignment('center')
        elif 0 < angle < np.pi:
            label.set_horizontalalignment('left')
        else:
            label.set_horizontalalignment('right')

def locals_explanation_using_shap(mode, all_score, labels_training_set, labels_test_set, a, train_set,test_set, position, integral_test_set):
    '''

    :param mode: save or load, in order to access the already computed
    :param all_score: all the score from train set and test set
    :param labels_training_set:
    :param labels_test_set:
    :param a: alpha parameter for mord ordinal regression
    :param train_set:
    :param test_set:
    :paramn integral_test_set: test set without robust scaler application
    :return:
            shap explainer
            list of shap values
            list of predictions from test set (encoded)
            list of real prediction from test set (presents also intervals)
            list of motivation for each prediction
    '''
    if(mode == 'save'):
        le = preprocessing.LabelEncoder()
        le.fit(all_score)
        train_encoded_values = le.transform(labels_training_set)
        test_encoded_values = le.transform(labels_test_set)

        model_ordinal = LogisticAT(alpha=a)
        model_ordinal.fit(train_set.values, train_encoded_values)
        predictions = model_ordinal.predict(test_set)
        real_predictions = le.inverse_transform(predictions)

        # explain all the predictions in the test set
        explainer = shap.KernelExplainer(model_ordinal.predict_proba, train_set)

        shap_values = explainer.shap_values(test_set)

        with open("mord_shap_values_"+ position +"without_ratings.txt", "wb") as fp:
            pickle.dump(shap_values, fp)
    else:
        le = preprocessing.LabelEncoder()
        le.fit(all_score)
        train_encoded_values = le.transform(labels_training_set)
        test_encoded_values = le.transform(labels_test_set)

        model_ordinal = LogisticAT(alpha=a)
        model_ordinal.fit(train_set.values, train_encoded_values)
        predictions = model_ordinal.predict(test_set)
        real_predictions = le.inverse_transform(predictions)

        # explain all the predictions in the test set
        explainer = shap.KernelExplainer(model_ordinal.predict_proba, train_set)

        with open("mord_shap_values_"+ position +"without_ratings.txt", "rb") as fp:
            shap_values = pickle.load(fp)

    list_of_explanation = []
    for inde in range(0, len(predictions)):
        # extract predictions value
        importance_list = shap_values[predictions[inde]][inde, :]

        # extract the column index of positive increasing elements
        explanation = {}
        index = 0
        for el in importance_list:
            if (el > 0):
                explanation[index] = el
            index += 1
        exp = sorted(explanation.items(), key=lambda x: x[1], reverse=True)

        explanation = {}
        for el in exp:
            if (el[1] >= 0.01):
                explanation[el[0]] = el[1]
        newexp = {}
        for key in explanation.keys():
            newexp[key] = train_set.columns[key]

        explanation = {}
        for key in newexp.keys():
            explanation[newexp[key]] = integral_test_set.iloc[inde, key]
        list_of_explanation.append(explanation)

    return explainer, shap_values, predictions, real_predictions, list_of_explanation