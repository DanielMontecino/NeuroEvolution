import pickle

from sklearn.metrics import confusion_matrix, balanced_accuracy_score
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import learning_curve, ShuffleSplit, GridSearchCV, StratifiedKFold, RepeatedStratifiedKFold
from sklearn.svm import SVR, SVC
from sklearn.kernel_ridge import KernelRidge
from sklearn.feature_selection import VarianceThreshold, SelectKBest, SelectPercentile, RFECV, RFE, SelectFdr, SelectFpr
from sklearn.feature_selection import chi2,f_classif, mutual_info_classif
from sklearn import neighbors

from utils.evaluation import *


def convert_to_classes(yi, divisions, binary=True, div_per_class_0=1):
    div = np.hstack([divisions.astype(np.float32)[1::], np.ones((1,))])
    class_yi = np.argmax(yi <= div)
    if binary:
        return 0 if class_yi < div_per_class_0 else 1
    else:
        return class_yi


def get_Xr_Y(table, divisions, fitness_lim=None, regression=False, binary=True, divisions_class_0=1):
    if fitness_lim is not None:
        table_ = table[table['error'] < fitness_lim]
    else:
        table_ = table
    Y = table_['error'].to_numpy()
    if not regression:
        Y = np.array([convert_to_classes(y, divisions, binary=binary, div_per_class_0=divisions_class_0) for y in Y])
    return get_Xr(table_), Y


def get_Xr(table_):
    table_ = table_.loc[:, table_.columns != 'id']
    table_ = table_.loc[:, table_.columns != 'error']
    table_ = table_.loc[:, table_.columns != 'code']
    Xr = table_.to_numpy() # [:, 3::]

    for i in range(Xr.shape[0]):
        for k in range(Xr.shape[1]):
            if Xr[i, k] is None:
                Xr[i, k] = -1
    Xr = Xr.astype(np.float32)
    return Xr


def get_estimators_dict():
    estimators_dict = {'SVC_rbf': SVC(kernel='rbf', gamma='auto'),
                       'SVC_linear': SVC(kernel="linear", gamma='auto'),
                       'KNeighbors': neighbors.KNeighborsClassifier(3, weights='uniform'),
                       #'DecisionTree': tree.DecisionTreeClassifier(),
                       'RandomForest': RandomForestClassifier(n_estimators=10)}
    # estimators_dict['GaussianNB'] = GaussianNB()
    return estimators_dict


class AbstractSelector:
    def __init__(self):
        self.scores_ = None
        self.k = None
        self.selector = None

    def func_(self, k, X_=None, svc=None):
        if k is None:
            self.scores_ = None
            return np.arange(1, X_.shape[1] + 1)
        self.k = k
        return self

    def fit_transform(self, X, y):
        raise NotImplementedError

    def get_x_from_score(self, X):
        order = np.argsort(self.scores_)[::-1]
        return np.array([X[:, order[i]] for i in range(self.k)]).T


class SelectorMI(AbstractSelector):
    def __init__(self,selector_builder):
        super().__init__()
        self.selector_builder = selector_builder

    def fit_transform(self, X, y):
        if self.scores_ is None:
            scores = []
            for i in range(100):
                self.selector = self.selector_builder(mutual_info_classif, 1)
                self.selector.fit(X, y)
                scores.append(np.array(self.selector.scores_))
            scores = np.array(scores)
            self.scores_ = np.mean(np.array(scores), axis=0)
        return self.get_x_from_score(X)


class SelectorRF(AbstractSelector):
    def fit_transform(self, X, y):
        if self.scores_ is None:
            self.selector = ExtraTreesClassifier(n_estimators=500,
                                          random_state=0)
            self.selector.fit(X, y)
            self.scores_ = self.selector.feature_importances_
        return self.get_x_from_score(X)


class SelectorH(AbstractSelector):
    def __init__(self, table_):
        super().__init__()
        self.table = table_
        self.features = table_.columns[3::].to_list()
        self.ranking_ = None
        self.ordered_features = ['is_there_prelu',
                                 'LR',
                                 'n_connections',
                                 'conv3_followed_by_conv5',
                                 'dropout_min',
                                 'filter_mul_max',
                                 'input_has_more_outputs',
                                 'kernel_5',
                                 'WU',
                                 'GR',
                                 'n_prelu',
                                 'n_skip_connections',
                                 'connections-joins',
                                 'n_layers',
                                 'filter_mul_min',
                                 'large_skip_c',
                                 'n_convs',
                                 'joins',
                                 'kernel_3',
                                 'kernel_1',
                                 'CELL',
                                 'dropout_max',
                                 'n_identities',
                                 'final_node_is_conv',
                                 'final_node_conv']
        self.orig_scores_ = [72, 36, 20, 13, 8, 3, 3, 2, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        self.complete_features_and_scores()

    def complete_features_and_scores(self):
        for f in self.features:
            if f not in self.ordered_features:
                self.ordered_features.append(f)
                self.orig_scores_.append(-1)
        self.scores_ = [self.orig_scores_[i] + (1 / (i + 1.)) for i in range(len(self.orig_scores_))]
        self.ranking_ = [self.ordered_features.index(f) for f in self.features]  # ranking/position of the i-th feature
        self.scores_ = [self.scores_[i] for i in self.ranking_]

    def fit_transform(self, X, y):
        self.features = self.features[0:X.shape[1]]
        self.complete_features_and_scores()
        assert np.sum(np.abs(np.argsort(self.ranking_) - np.argsort(self.scores_)[::-1])) == 0
        return self.get_x_from_score(X)


class SelectorUnivariate(AbstractSelector):
    def __init__(self, selector_builder, score_func):
        super().__init__()
        self.selector_builder = selector_builder
        self.score_func = score_func

    def fit_transform(self, X, y):
        if self.scores_ is None:
            self.selector = self.selector_builder(self.score_func, 1)
            self.selector.fit(X, y)
            self.scores_ = self.selector.scores_
        return self.get_x_from_score(X)


class SelectorVariance(AbstractSelector):
    def fit_transform(self, X, y):
        if self.scores_ is None:
            self.selector = VarianceThreshold()
            self.selector.fit(X, y)
            self.scores_ = self.selector.variances_
        return self.get_x_from_score(X)


class SelectorRFE(AbstractSelector):
    def func_(self, k, X_=None, svc=None):
        if k is None:
            self.scores_ = None
            return np.arange(1, X_.shape[1] + 1)
        if self.scores_ is None:
            self.selector = RFE(estimator=svc, n_features_to_select=1, step=1)
        self.k = k
        return self

    def fit_transform(self, X, y):
        if self.scores_ is None:
            self.selector.fit(X, y)
            self.scores_ = len(self.selector.ranking_) - self.selector.ranking_
        return self.get_x_from_score(X)


class SelectorRFECV(SelectorRFE):
    def func_(self, k, X_=None, svc=None):
        if k is None:
            self.scores_ = None
            return np.arange(1, X_.shape[1] + 1)
        if self.scores_ is None:
            self.selector = RFECV(estimator=svc,  step=1, cv=StratifiedKFold(2, shuffle=True), min_features_to_select=1)
        self.k = k
        return self


def get_selectors(name, score_func):
    if name == 'KBest':
        def func_(k, X_=None, svc=None):
            if k is None:
                return np.arange(1, X_.shape[1] + 1)
            return SelectKBest(score_func, k=k)
    elif name == 'SelectFdr':
        def func_(k, X_=None, svc=None):
            if k is None:
                return np.linspace(0.001, 0.1, 25)
            return SelectFdr(score_func, alpha=k)

    elif name == 'SelectFpr':
        def func_(k, X_=None, svc=None):
            if k is None:
                return np.linspace(0.001, 0.1, 25)
            return SelectFpr(score_func, alpha=k)
    else:
        def func_(k, X_=None, svc=None):
            if k is None:
                return np.linspace(1, 100, 25).astype(np.int32)
            return SelectPercentile(score_func, percentile=k)
    return func_


def get_VarianceThreshold(k, X_=None, svc=None):
    if k is None:
        return np.linspace(0.01, 0.99, 20)
    return VarianceThreshold(threshold=k)


def get_RFE(k, X_=None, svc=None):
    if k is None:
        return np.arange(1, X_.shape[1] + 1)
    rfe = RFE(estimator=svc, n_features_to_select=k, step=1)
    return rfe


def get_RFECV(k, X_=None, svc=None):
    if k is None:
        return np.arange(1, X_.shape[1] + 1)
    rfecv = RFECV(estimator=svc, step=1, cv=StratifiedKFold(2, shuffle=True), min_features_to_select=k)
    return rfecv


def get_mutual_info_classif(X, y):
    return mutual_info_classif(X, y, random_state=0, n_neighbors=5)


def get_selectors_dict(table_=None):
    score_func = {'chi2':chi2, 'F-value':f_classif}
    univariate = {'KBest': SelectKBest}  #  , 'Fpr':SelectFpr,  'Fdr':SelectFdr}
    selectors_dict = {}

    for s_name, s_builder in univariate.items():
        selectors_dict['%s (mutual information)' % s_name] = SelectorMI(s_builder).func_
        for sf_name, sf in score_func.items():
            selectors_dict['%s_%s' % (s_name, sf_name)] = SelectorUnivariate(s_builder, sf).func_

    #selectors_dict['RandomForest'] = SelectorRF().func_
    if table_ is not None:
        selectors_dict['selector_by_hand'] = SelectorH(table_).func_
    selectors_dict['VarianceThreshold'] = SelectorVariance().func_
    selectors_dict['RFE'] = SelectorRFE().func_
    selectors_dict['RFECV'] = SelectorRFECV().func_
    return selectors_dict


def get_best_selector_and_estimators():
    d0 = {'estimator': ('KNN', neighbors.KNeighborsClassifier(3, weights='uniform')),
          'selector': ('KBest (ANOVA)', get_selectors('KBest', f_classif))}
    d1 = {'estimator': ('SVM (Lineal)', SVC(kernel="linear", gamma='auto')), 'selector': ('RFE', get_RFE)}
    d2 = {'estimator': ('SVM (Lineal)', SVC(kernel="linear", gamma='auto')), 'selector': ('RFECV', get_RFECV)}
    d3 = {'estimator': ('SVM (Lineal)', SVC(kernel="linear", gamma='auto')), 'selector': ('RF', SelectorRF().func_)}
    return [d0, d1, d2, d3]


def shuffle(x, y):
    new_ids = np.arange(len(y))
    np.random.shuffle(new_ids)
    X = np.array([x[i,...] for i in new_ids])
    Y = np.array([y[i] for i in new_ids])
    return X, Y


def plot_learning_curve(estimator, title, X, y, axes=None, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 10), regression=False):
    """
    Generate 3 plots: the test and training learning curve, the training
    samples vs fit times curve, the fit times vs score curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    axes : array of 3 axes, optional (default=None)
        Axes to use for plotting the curves.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
          - None, to use the default 5-fold cross-validation,
          - integer, to specify the number of folds.
          - :term:`CV splitter`,
          - An iterable yielding (train, test) splits as arrays of indices.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : int or None, optional (default=None)
        Number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    train_sizes : array-like, shape (n_ticks,), dtype float or int
        Relative or absolute numbers of training examples that will be used to
        generate the learning curve. If the dtype is float, it is regarded as a
        fraction of the maximum size of the training set (that is determined
        by the selected validation method), i.e. it has to be within (0, 1].
        Otherwise it is interpreted as absolute sizes of the training sets.
        Note that for classification the number of samples usually have to
        be big enough to contain at least one sample from each class.
        (default: np.linspace(0.1, 1.0, 5))
    """
    if axes is None:
        _, axes = plt.subplots(1, 3, figsize=(20, 5))
    if regression:
        score = "neg_mean_squared_error"
    else:
        score = None

    axes.set_title(title)
    if ylim is not None:
        axes.set_ylim(*ylim)
    axes.set_xlabel("Training examples")
    axes.set_ylabel("Score")

    train_sizes, train_scores, test_scores, fit_times, _ = \
        learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs,
                       train_sizes=train_sizes,
                       scoring=score,
                       return_times=True)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    # Plot learning curve
    axes.grid()
    axes.fill_between(train_sizes, train_scores_mean - train_scores_std,
                      train_scores_mean + train_scores_std, alpha=0.1, color="r")
    axes.fill_between(train_sizes, test_scores_mean - test_scores_std,
                      test_scores_mean + test_scores_std, alpha=0.1, color="g")
    axes.plot(train_sizes, train_scores_mean, 'o-', color="r",
              label="Training score")
    axes.plot(train_sizes, test_scores_mean, 'o-', color="g",
              label="Cross-validation score")
    axes.legend(loc="best")
    return plt


def show_curve(Xr, Y, regression=False):
    if regression:
        X, y = shuffle(Xr, Y)  # load_digits(return_X_y=True)
        svr = GridSearchCV(SVR(kernel='rbf', gamma=0.1),
                           param_grid={"C": [1e0, 1e1, 1e2, 1e3],
                                       "gamma": np.logspace(-2, 2, 5)})

        kr = GridSearchCV(KernelRidge(kernel='rbf', gamma=0.1),
                          param_grid={"alpha": [1e0, 0.1, 1e-2, 1e-3],
                                      "gamma": np.logspace(-2, 2, 5)})
        estimators = [svr, kr]
        titles = ["Learning Curves (SVR)", "Learning Curves (KR)"]
        ylim = None  # (-7.5, 0.1)
    else:
        X, y = shuffle(Xr, Y)  # load_digits(return_X_y=True)

        estimators = [GaussianNB(), SVC(gamma=0.001), neighbors.KNeighborsClassifier(3, weights='uniform')]
        titles = ["Learning Curves (Naive Bayes)", r"Learning Curves (SVM, RBF kernel, $\gamma=0.001$)", 'KNN_3']
        ylim = (0.7, 1.01)

    cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)
    fig, axes = plt.subplots(1, len(estimators), figsize=(6 * len(estimators), 5))
    for i in range(len(estimators)):
        plot_learning_curve(estimators[i], titles[i], X, y, axes=axes[i], ylim=ylim,
                            cv=cv, n_jobs=4)

    plt.show()


def evaluate(X_, Y_, estimator, selector=None, reps=10, confusion_m=False):
    w0, w1 = np.sum(Y_ == 0), np.sum(Y_ == 1)
    w = w0 + w1
    rskf = RepeatedStratifiedKFold(n_splits=2, n_repeats=reps)
    train_scores, test_scores = [], []
    confusion_matrices = []
    if selector is not None:
        X_ = selector.fit_transform(X_, Y_)
    for train_index, test_index in rskf.split(X_, Y_):
        X_train, X_test = X_[train_index], X_[test_index]
        y_train, y_test = Y_[train_index], Y_[test_index]
        try:
            weights = [w / w0 if y == 0 else w / w1 for y in y_train]
            estimator.fit(X_train, y_train, weights)
        except TypeError:
            estimator.fit(X_train, y_train)
        y_test_pred = estimator.predict(X_test)
        y_train_pred = estimator.predict(X_train)
        if w0 != w1:
            train_scores.append(balanced_accuracy_score(y_train, y_train_pred))
            test_scores.append(balanced_accuracy_score(y_test, y_test_pred))
        else:
            train_scores.append(estimator.score(X_train, y_train))
            test_scores.append(estimator.score(X_test, y_test))
        if confusion_m:
            y_pred = estimator.predict(X_test)
            confusion_matrices.append(confusion_matrix(y_test, y_pred))
    if confusion_m:
        cm = np.mean(np.array(confusion_matrices), axis=0)

        return (np.mean(train_scores), np.std(train_scores)), (np.mean(test_scores), np.std(test_scores)), cm, estimator, selector
    return (np.mean(train_scores), np.std(train_scores)), (np.mean(test_scores), np.std(test_scores))


def evaluate_range_values(Xr, Yc, estimator, get_selector, reps=10, plot=False):
    train_scores_mean = []
    train_scores_std = []
    test_scores_mean = []
    test_scores_std = []
    n_features = get_selector(k=None, X_=Xr, svc=None)
    print("features :[", end='')
    for i in n_features:
        selector = get_selector(k=i, svc=estimator)
        (train_m, train_s), (test_m, test_s) = evaluate(Xr, Yc, estimator, selector, reps=reps)
        train_scores_mean.append(train_m)
        train_scores_std.append(train_s)
        test_scores_mean.append(test_m)
        test_scores_std.append(test_s)
        print("=", end='')
    print("]")

    train_scores_mean = np.array(train_scores_mean)
    train_scores_std = np.array(train_scores_std)
    test_scores_mean = np.array(test_scores_mean)
    test_scores_std = np.array(test_scores_std)
    if plot:
        fig = plt.figure(figsize=(10,10))
        plt.grid()
        plt.fill_between(n_features, train_scores_mean - train_scores_std,
                             train_scores_mean + train_scores_std, alpha=0.1,
                             color="r")
        plt.fill_between(n_features, test_scores_mean - test_scores_std,
                             test_scores_mean + test_scores_std, alpha=0.1,
                             color="g")
        plt.plot(n_features, train_scores_mean, 'o-', color="r",
                     label="Training score")
        plt.plot(n_features, test_scores_mean, 'o-', color="g",
                     label="Cross-validation score")
        plt.legend(loc="best")
        plt.show()
    return test_scores_mean, test_scores_std, n_features


def evaluate_estimators(X_, Y_, estimators_dict, selector, reps=100, title=''):
    evaluated_dict = {}
    plt.figure(figsize=(10, 10))
    for estimator_name, estimator in estimators_dict.items():
        try:
            test_scores_mean, test_scores_std, n_features = evaluate_range_values(X_, Y_, estimator, selector, reps=reps)
            evaluated_dict[estimator_name] = {}
        except RuntimeError:
            continue
        evaluated_dict[estimator_name]['test_mean'] = test_scores_mean
        evaluated_dict[estimator_name]['test_std'] = test_scores_std
        evaluated_dict[estimator_name]['n_features'] = n_features

        plt.fill_between(n_features, test_scores_mean - test_scores_std,
                             test_scores_mean + test_scores_std, alpha=0.1)
        plt.plot(n_features, test_scores_mean, 'o-', label=estimator_name)
    plt.legend()
    plt.xlabel('N° features')
    plt.ylabel("Score")
    plt.title(title)
    plt.show()
    return evaluated_dict


def plot_estimators(final_dict, estimators_dict, selectors_dict, savefig=None):
    for e_name in estimators_dict.keys():
        print(e_name)
        fig = plt.figure(figsize=(10, 10))
        for s_name in selectors_dict.keys():
            if s_name not in final_dict.keys():
                continue
            ev_dict = final_dict[s_name]
            if e_name in ev_dict.keys() and len(ev_dict[e_name]) > 0:
                test_scores_mean = ev_dict[e_name]['test_mean']
                test_scores_std = ev_dict[e_name]['test_std']
                n_features = np.linspace(1, 39, len(test_scores_mean))
                plt.fill_between(n_features, test_scores_mean - test_scores_std,
                                 test_scores_mean + test_scores_std, alpha=0.1)
                if 'chi' in s_name:
                    s_name = 'KBest (Chi2)'
                elif 'f_class' in s_name:
                    s_name = 'KBest (ANOVA F-value)'
                elif 'mutual' in s_name:
                    s_name = 'KBest (mutual information)'
                plt.plot(n_features, test_scores_mean, 'o-', label=s_name)
        plt.legend()
        plt.xlabel('N° features')
        plt.ylabel("Score")
        plt.title(e_name)
        # plt.ylim([0.91, 0.95])
        ylim = list(plt.gca().get_ylim())
        ylim[0] = max(ylim[0], 0.8)
        ylim[1] = min(ylim[1], 0.96)
        plt.yticks(np.arange(0.5, 0.97, 0.005))
        plt.ylim(ylim)
        plt.grid()
        if savefig is not None:
            plt.savefig("%s_%s" % (savefig, e_name))
        plt.show()


def plot_selectors(final_dict, savefig=None):
    for s_name, s_dict in final_dict.items():
        print(s_name)
        if s_name not in final_dict.keys():
            continue
        plt.figure(figsize=(10, 10))
        for e_name, ev_dict in s_dict.items():
            if len(ev_dict) < 1:
                continue
            test_scores_mean = ev_dict['test_mean']
            test_scores_std = ev_dict['test_std']
            n_features = ev_dict['n_features']
            plt.fill_between(n_features, test_scores_mean - test_scores_std,
                             test_scores_mean + test_scores_std, alpha=0.1)
            plt.plot(n_features, test_scores_mean, 'o-', label=e_name)

        if 'chi' in s_name:
            s_name = 'KBest (Chi2)'
        elif 'f_class' in s_name:
            s_name = 'KBest (ANOVA F-value)'
        elif 'mutual' in s_name:
            s_name = 'KBest (mutual information)'
        plt.legend()
        plt.xlabel('N° features')
        plt.ylabel("Score")
        plt.title(s_name)
        ylim = list(plt.gca().get_ylim())
        ylim[0] = max(ylim[0], 0.8)
        ylim[1] = min(ylim[1], 0.96)
        plt.yticks(np.arange(0.5, 0.97, 0.005))
        plt.ylim(ylim)
        plt.grid()
        if savefig is not None:
            plt.savefig("%s_%s" % (savefig, s_name))
        plt.show()


class ModelFilter:

    def __init__(self, file_to_save, TwoLevelGaObject, data_path='../../experiments/test_validation3', one_level=False, divisions=5, fitness_div=None, n_features=8):
        self.file_to_save = file_to_save
        self.fitness_lim = None if fitness_div is not None else 0.1   # 0.107984 if serie == 1 else 0.1
        self.data_path = data_path
        self.one_level = one_level
        self.divisions = divisions
        self.k = n_features
        self.estimator = None
        self.selector = None
        self.table = None
        self.fitness_div = fitness_div
        self.twolevel = TwoLevelGaObject
        self.create_system()

    def get_data(self, dataset, parent_dir, runs=None):
        #one_level = '54' in parent_dir
        runs = os.listdir(parent_dir) if runs is None else runs
        sorted_h, sorted_ph = get_sorted_individuals_from_evolutions(dataset, parent_dir, runs, self.twolevel)
        table_h, table_ph = get_sorted_tables(sorted_h=sorted_h, sorted_ph=sorted_ph)
        self.table = table_h if self.one_level else table_ph
        sorted_ph = sorted_h if self.one_level else sorted_ph
        divisions = self.get_divisions(sorted_ph)
        Xr, Yc = get_Xr_Y(self.table, divisions, fitness_lim=self.fitness_lim, regression=False, binary=True)
        print(np.sum(Yc), 'positive examples.', len(Yc) - np.sum(Yc), 'negative examples')
        return Xr, Yc

    def get_divisions(self, sorted_):
        if self.fitness_lim is not None:
            div_mode = 'fit'
            h = [s for s in sorted_ if s[1] < self.fitness_lim]
            percentiles_dict, divisions = make_percentiles(h, self.divisions, div_mode)
            return divisions
        else:
            return np.array([0, self.fitness_div])

    def evaluate(self, X_, Y_, estimator, selector):
        print(X_.shape)
        w0 = np.sum(Y_ == 0)
        w1 = np.sum(Y_ == 1)
        w = w0 + w1
        # x_train, y_train = shuffle(X_, Y_)
        # x_train = selector.fit_transform(x_train, y_train)

        rskf = RepeatedStratifiedKFold(n_splits=2, n_repeats=100)
        X_ = selector.fit_transform(X_, Y_)
        for train_index, test_index in rskf.split(X_, Y_):
            x_train, x_test = X_[train_index], X_[test_index]
            y_train, y_test = Y_[train_index], Y_[test_index]
            try:
                weights = [w / w0 if y == 0 else w / w1 for y in y_train]
                estimator.fit(x_train, y_train, weights)
            except TypeError:
                estimator.fit(x_train, y_train)
        return estimator, selector

    def create_system(self):
        k = self.k
        Xr, Yc = self.get_data('MRDBI', self.data_path)
        estimator = SVC(kernel="linear", gamma='auto')
        selector = SelectorRFE()
        #selector = SelectorH(self.table)
        _ = selector.func_(k=None, X_=Xr, svc=estimator)
        selector = selector.func_(k=k, X_=Xr, svc=estimator)
        self.estimator, self.selector = self.evaluate(Xr, Yc, estimator, selector)
        self.save(self.file_to_save)

    def is_model_ok(self, model):
        x = self.get_features_from_model(model)
        x = self.selector.get_x_from_score(x)
        return self.estimator.predict(x)[0] == 0
    
    def get_model_prob(self, model):
        x = self.get_features_from_model(model)
        x = self.selector.get_x_from_score(x)
        return self.estimator.decision_function(x)[0]

    def get_features_from_model(self, model):
        b = -1
        features_dict = {b: {}}
        for value, function in get_util_functions().items():
            features_dict[b][value] = function(str(model))
        c = pd.DataFrame(features_dict, index=self.table.columns).T
        x = get_Xr(c)
        return x

    def save(self, filename):
        outfile = open(filename, 'wb')
        pickle.dump(self, outfile)
        outfile.close()

    @staticmethod
    def load(file):
        infile = open(file, 'rb')
        obj = pickle.load(infile)
        infile.close()
        return obj
