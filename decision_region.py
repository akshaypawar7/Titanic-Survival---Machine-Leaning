

def dec_regs(X, y, estimators):
    from mlxtend.plotting import plot_decision_regions
    fig, axes = plt.subplots(3,2,figsize=(20, 30))

    axes = axes.flatten()
    
    for ax, estimator in zip(axes, estimators):
        estimator.fit(X, y.values)
        plot_decision_regions(X, y.values, clf=estimator,colors='#e00d14,#3ca02c',markers='x+', ax=ax)
        ax.set(title= estimator.__class__.__name__)

        ax.set_xlabel('PCA 1')
        ax.set_ylabel('PCA 2')

        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles,
                  ['Not Survived', 'Survived'],
                  framealpha=0.3, scatterpoints=1)

    plt.show()


def prob_regs(X, y, estimators):
    import math
    from matplotlib.colors import ListedColormap
    figure = plt.figure(figsize=(10, 10))
    h = .02
    i = 1

    # preprocess dataset, split into training and test part
    X_train, X_test, y_train, y_test = \
        train_test_split(X, y, test_size=.4, random_state=42)

    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    # Just plot the dataset first
    cm = plt.cm.RdYlGn
    cm_bright = ListedColormap(['#e00d14', '#3ca02c'])
    ax = plt.subplot(5, 2, i)

    # Iterate over estimators
    for clf in estimators:
        ax = plt.subplot(math.ceil(len(estimators) / 2), 2, i)
        clf.fit(X_train, y_train)
        score = clf.score(X_test, y_test)

        # Plot the decision boundary. For that, we will assign a color to each
        # point in the mesh [x_min, x_max]x[y_min, y_max].
        if hasattr(clf, "decision_function"):
            Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
        else:
            Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]

        # Put the result into a color plot
        Z = Z.reshape(xx.shape)
        ax.contourf(xx, yy, Z, cmap=cm, alpha=.8)

        # Plot the training points
        g = ax.scatter(X_train[:, 0],
                       X_train[:, 1],
                       c=y_train,
                       cmap=cm_bright,
                       edgecolors='k')
        # Plot the testing points
        ax.scatter(X_test[:, 0],
                   X_test[:, 1],
                   c=y_test,
                   cmap=cm_bright,
                   edgecolors='k',
                   alpha=0.6)

        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())

        ax.set_title(clf.__class__.__name__)
        #ax.text(xx.max() - .3,
        #        yy.min() + .3, (f'Score:{score:.2f}'),
        #        size=15,
        #        horizontalalignment='right')

        ax.set_xlabel('PCA 1')
        ax.set_ylabel('PCA 2')
        plt.legend(handles=g.legend_elements()[0],
                   labels=['Not Survived', 'Survived'],
                   framealpha=0.3,
                   scatterpoints=1)

        i += 1

    plt.tight_layout()
    plt.show()
