def model_tuning_check(X, y, estimators, cv=None):
    model_df = pd.DataFrame(columns = ['mean_test_score', 'mean_train_score', 'std_test_score', 'std_train_score','params','mean_fit_time'])
    best_estimators =[]
    
    for est, param in estimators:

        gscv = GridSearchCV(
            est,
            param_grid= param,
            scoring='accuracy',
            cv=cv,
            #refit=False,
            return_train_score=True,
            n_jobs=-1).fit(X,y)

        model_df.loc[est.__class__.__name__] = pd.DataFrame(
            gscv.cv_results_).loc[
            gscv.best_index_][['mean_test_score', 'mean_train_score', 'std_test_score', 'std_train_score','params','mean_fit_time']]
        
        #save estimator for later use 
        best_estimators.append(gscv.best_estimator_)
        
    model_df.sort_values(by=['mean_test_score'],
                            ascending=False,
                            inplace=True)
    # Accuracy score DataFrame
    display(model_df.style.background_gradient(cmap='summer_r'))
    
    # accuracy score graph
    sns.barplot('mean_test_score',
                model_df.index,
                data=model_df,
                orient='h',
               **{'xerr': model_df['std_test_score']})
    plt.title('GridSearchCV Scores')
    plt.xlim([0.725, 0.88])
    plt.show()

    return best_estimators






ram = [
    (LogisticRegression(), {'max_iter' : [100],
                            'penalty' : ['l1', 'l2'],
                            'C' : np.logspace(-2, 2, 20),
                            'solver' : ['lbfgs', 'liblinear']}),
    (KNeighborsClassifier(), {'n_neighbors' : np.arange(3, 30, 1),
                              'weights': ['uniform', 'distance'],
                              'algorithm': ['auto'],
                              'p': [1, 2]}),
    (SVC(), [{'kernel': ['rbf'], 
              'gamma': [0.001, 0.01, 0.1, 0.5, 1, 2, 5],
              'C': [0.1, 0.5,  1, 2, 5]},
              {'kernel': ['linear'], 
              'C': [.1, 1, 2, 10]},
              {'kernel': ['poly'], 
              'degree' : [2, 3, 4, 5], 
              'C': [.1, 1, 10]}]),
    (DecisionTreeClassifier(random_state = 1), {'max_depth': [3, 5, 10, 20, 50],
                                                'criterion': ['entropy', 'gini'],
                                                'min_samples_split': [5, 10, 15, 30],
                                                'max_features': ['auto', 'sqrt', 'log2']}),
    (RandomForestClassifier(random_state = 42),{'n_estimators': [50, 150, 300, 450],
                                                'criterion': ['entropy'],
                                                'bootstrap': [True],
                                                'max_depth': [3, 5, 10],
                                                'max_features': ['auto','sqrt'],
                                                'min_samples_leaf': [2, 3],
                                                'min_samples_split': [2, 3]}),
    (XGBClassifier(random_state = 1, use_label_encoder=False,tree_method='gpu_hist', gpu_id=0),{'n_estimators': [15, 25, 50, 100],
                                                                                                'colsample_bytree': [0.65, 0.75, 0.80],
                                                                                                #'max_depth': [None],
                                                                                                'reg_alpha': [1],
                                                                                                'reg_lambda': [1, 2, 5],
                                                                                                'subsample': [0.50, 0.75, 1.00],
                                                                                                'learning_rate': [0.01, 0.1, 0.5],
                                                                                                'gamma': [0.5, 1, 2, 5],
                                                                                                'min_child_weight': [0.01],
                                                                                                'sampling_method': ['uniform']}),
    (LGBMClassifier(random_state=42),{#'num_leaves': sp_randint(6, 50), 
                                      #'min_child_samples': sp_randint(100, 500), 
                                      'min_child_weight': [1e-5, 1e-1, 1, 1e1, 1e2, 1e4],
                                      #'subsample': sp_uniform(loc=0.2, scale=0.8), 
                                      #'colsample_bytree': sp_uniform(loc=0.4, scale=0.6),
                                      'reg_alpha': [0, 1e-1, 1, 2, 5, 7, 10, 50],
                                      'reg_lambda': [0, 1e-1, 1, 5, 10, 20, 50, 10]}),
    (AdaBoostClassifier(base_estimator = DecisionTreeClassifier(random_state = 1), random_state=1),{'algorithm': ['SAMME', 'SAMME.R'],
                                                                                                    'base_estimator__criterion' : ['gini', 'entropy'],
                                                                                                    'base_estimator__splitter' : ['best', 'random'],
                                                                                                    'n_estimators': [2, 5, 10, 50],
                                                                                                    'learning_rate': [0.01, 0.1, 0.2, 0.3, 1, 2]})]

