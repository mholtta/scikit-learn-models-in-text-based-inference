import sklearn
from sklearn.metrics import confusion_matrix
import progressbar

import numpy as np
import pandas as pd
import seaborn as sns
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure


def test_different_models(X_train, X_test, y_train, y_test, modelsToTest, average_type='binary', show_learning_curve=False, random_state = 42):
  """
  Function for testing different models.

  Train and test data and a list of models (with their name and function call) should be given as parameters.

  Outputs a dataframe with various measures on performance of the model with the given data as columns.

  """
  
  predictionTable = pd.DataFrame()
  predictionTable["True value"] = y_test.reshape(-1)
  
  with progressbar.ProgressBar(max_value=len(modelsToTest)) as bar:
    i=0
    for m in modelsToTest:
      #print(m[0])
      model=m[1]
      model.fit(X_train, y_train)
      
      predictions = model.predict(X_test)
      if len(m) > 2:
        predictions = predictions > m[2]
      predictionTable[m[0]] = predictions
  
      if show_learning_curve:
        #plot_learning_curve -toimii (ainakin osassa algoritmeja), mutta se on todella hidas!
        cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=random_state)
        plot_learning_curve(m[1], m[0], X_train, y_train, cv=cv, n_jobs=4) #ylim=(0.7, 1.01)
        plt.show()

      #plot_validation_curve EI TOIMI, pitäisi jalostaa...
      #plot_validation_curve(m[1], m[0], X, y)
      bar.update(i)
      i=i+1
  return predictionTable

def get_metrics_for_algorithm(model_name, true_labels, predictions):
  """
  Helper function for obtaining metrics for classification.

  """

  accuracy = sklearn.metrics.accuracy_score(true_labels, predictions)
  precision = sklearn.metrics.precision_score(true_labels, predictions)
  recall = sklearn.metrics.recall_score(true_labels, predictions)
  AUC = float('nan')
  try:
    AUC = sklearn.metrics.roc_auc_score(true_labels, predictions)
  except ValueError:
    pass
  matthews_corrcoef = sklearn.metrics.matthews_corrcoef(true_labels, predictions)

  return pd.Series({
      "Model":model_name, 
      "Accuracy":accuracy, 
      "Precision":precision, 
      "Recall":recall,
      "MCC": matthews_corrcoef,
      "AUC":AUC
      })

def metrics_classification(df):
  """
  Function for obtaining metrics for classification of a dataframe containing true values and predictions for different models.

  """
  true_labels = df['True value']
  predictions = df.drop('True value', axis=1)

  scoreDf = pd.DataFrame(columns=["Model", "Accuracy"])
  scoreDf.set_index("Model")
  for model_name, predictions in predictions.iteritems():
    scoreDf = scoreDf.append(get_metrics_for_algorithm(model_name, true_labels, predictions), ignore_index=True)
  return scoreDf


def add_or_replace_to_dataframe(name, dataframe, series):
  """
  A convenience function for adding or replacing new model scores to a dataframe.

  """

  if(name in dataframe["Model"].values or name in dataframe.index):
    # print("replace")
    # replace
    dataframe.update(series)
  else:
    # print("add")
    #add
    dataframe = dataframe.append(series, ignore_index=True)
  return dataframe



def metrics_regression(df):
  """
  For obtaining metrics for regression problems.

  """

  y_true = df['True value']
  predictions = df.drop('True value', axis=1)

  scoreDf = pd.DataFrame(columns=["Model"])
  scoreDf.set_index("Model")

  for label, content in predictions.iteritems():

      mean_squared_error = sklearn.metrics.mean_squared_error(y_true, content)
      r2_score = sklearn.metrics.r2_score(y_true, content)
      explained_variance = sklearn.metrics.explained_variance_score(y_true, content)

      scoreDf = scoreDf.append(pd.Series({"Model":label, "Mean squared error":mean_squared_error, "R2 score":r2_score, "Explained variance":explained_variance}), ignore_index=True)
  return scoreDf




# ottaa syötteeksi Pandas dataframen, ja piirtää siitä kuvaajia
def create_graphs_from_scores(scores, modelsToTest):
  """
  A function for creating a summary graph of different measures for a classification algorithm.

  """

  _, axs = plt.subplots(1, 4, sharey='row', gridspec_kw={'hspace': 0, 'wspace': 0}, figsize=(10,len(modelsToTest)*0.8)) #sharex='col',
  (ax1, ax2, ax3, ax4) = axs

  scores.plot.barh(x='Model', y='Accuracy', legend=False, ax=ax1)
  ax1.invert_yaxis() 
  ax1.set_title('Accuracy')

  scores.plot.barh(x='Model', y='Precision', legend=False, ax=ax2)
  ax2.invert_yaxis() 
  ax2.set_title('Precision')

  scores.plot.barh(x='Model', y='Recall', legend=False, ax=ax3)
  ax3.invert_yaxis() 
  ax3.set_title('Recall')

  scores.plot.barh(x='Model', y='MCC', legend=False, ax=ax4)
  ax4.invert_yaxis() 
  ax4.set_title('Matthews Correlation Coefficient')
  plt.show()



def print_scoretable(scores, columns=["Accuracy", "AUC", "MCC", "Precision", "Recall"]):
  """
  A convenience function styling a dataframe such that text contains bars in the background."

  """


  return scores.style.set_precision(2).bar(
      subset=columns, color='#ddd'
      ).hide_index().set_properties(**{'width':'8em', 'text-align':'center'})

# Helper functions


def draw_confusion_matrix_for_good_poor(truevalues, predictions, normalize=True, title=""):
  """
  Function for drawing a confusion matrix.

  """

  res = pd.DataFrame()
  res['y_true'] = np.where(truevalues, "good", "poor")
  res['y_pred'] = np.where(predictions, "good", "poor")
  cm = confusion_matrix(res['y_true'], res['y_pred'])
  if normalize:
    cm = 100 * cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
  columns=["good", "poor"]
  df_cm = pd.DataFrame(cm, index=columns, columns=columns)
  fmt = '.0f' if normalize else 'd'
  ax = sns.heatmap(df_cm, annot=True, cmap='Oranges', fmt=fmt)
  for t in ax.texts: t.set_text(t.get_text() + " %")

  # fix for mpl bug that cuts off top/bottom of seaborn viz
  b, t = plt.ylim() # discover the values for bottom and top
  b += 0.5 # Add 0.5 to the bottom
  t -= 0.5 # Subtract 0.5 from the top
  plt.ylim(b, t) # update the ylim(bottom, top) values
  if title:
    plt.title(title)
  plt.ylabel('true values')
  plt.xlabel('predictions')
  plt.show()

def color_func_green(word, font_size, position, orientation,random_state=None, hue=100,  **kwargs):
    return("hsl({},50%, {}%)".format(np.random.randint(hue-5,hue+5), np.random.randint(20,51)))

def color_func_red(word, font_size, position, orientation,random_state=None, hue=5,  **kwargs):
    return("hsl({},50%, {}%)".format(np.random.randint(hue-5,hue+5), np.random.randint(20,51)))

def color_func_blue(word, font_size, position, orientation,random_state=None, hue=230,  **kwargs):
    return("hsl({},50%, {}%)".format(np.random.randint(hue-5,hue+5), np.random.randint(20,51)))

def draw_word_cloud(words, title, numberofWords = 100, color="blue", titlefontsize=12):
  """
  Function for drawing a wordcloud.

  Colour choices are red, green and blue.

  """

  wordCounts = words.sum(axis = 0, skipna = True)
  topFrequentWords = wordCounts.sort_values(ascending=False)[:numberofWords]
  wc = WordCloud(background_color='white') #, colormap="Blues"
  wc.fit_words(topFrequentWords)
  cf= color_func_red
  if(color=="green"):
    cf = color_func_green
  if(color=="blue"):
    cf = color_func_blue
  wc.recolor(color_func = cf)
  figure(figsize=(12,8))
  plt.imshow(wc, interpolation='bilinear')
  plt.axis("off")
  plt.title(title, fontsize=titlefontsize)
  plt.show()

def draw_roc_curve(fpr, tpr):
  """
  Draws a ROC-curve based on sklearn.metrics.roc_curve.

  """

  plt.plot(fpr, tpr, color='orange', label='ROC')
  plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
  plt.xlabel('False Positive Rate')
  plt.ylabel('True Positive Rate')
  plt.title('Receiver Operating Characteristic (ROC) Curve')
  plt.legend()
  plt.show()

def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None, n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Generate a simple plot of the test and training learning curve.

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

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
          - None, to use the default 3-fold cross-validation,
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
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt