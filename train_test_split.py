from sklearn.model_selection import train_test_split

def train_test_split_as_requested(X, y, portion = 0.8):
  """
  Returns X_train, X_test, y_train, y_test

  """
  print("Returns X_train, X_test, y_train, y_test")
  return train_test_split(X, y, test_size=1-portion, random_state=None, shuffle=False)

def get_training_portion_of_the_data(dataframe, training_portion = 0.8):
  """
  Returns train-test split requested in the project description.

  Takes training_portion amount of lines from the beginning of dataframe as the training set.

  """
  train_size = int(len(dataframe) * training_portion)
  print("returns {} lines in total {} out of lines (portion {})".format(train_size, len(dataframe), training_portion))
  return dataframe[:train_size]

