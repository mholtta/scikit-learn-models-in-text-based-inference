# Testing prediction with scikit-learn models based on textual data

This is the course project for CS-C3160 Data Science. The project was done in pairs, with equal contributions from both. I completed the course in 12/2019 and refactored the project in 8/2020. In the course we were given three datasets to choose from, we chose a dataset consisting of Yelp-reviews. The project met all the criteria set in grading and earned a grade 5/5 (maximum grade).

**We were asked to perform the following tasks in the project:**
- exploratory data analysis (including answering some questions) 
- classify the reviews to good and bad based on the review text
- predict the usefulness of a review based on the review text and other information such as number of stars given

**The notebooks are structured as follows:**
- notebook 1 contains loading in data and dumping dataframes with pickle
- notebook 2 contains the exploratory data analysis
- notebook 3.1 contains the work regarding predicting review classification to good or bad (above or below 3.5 stars) based on review text in bag-of-words format
- notebook 3.2 contains the work for predicting usefulness of reviews (how many Yelp users have marked the review as useful) based on review text and other data

### Learnings from the project:
- During refactoring:
    - Jupyter Notebooks are handy for visualizations and for some quick, interactive scripting. However, for anything more serious, the global nature of all variables leads quickly to issues even if functions are used extensively. Splitting all functions to separate .py files and importing from those makes it easy to make sure that no global variables not passed as arguments to a function get passed to function due to spelling errors. Having solved these issues a few times, this is an improvement.
  - Refactoring even ones own code from past takes unnecessarily long when the original code was at times repetitive and badly documented.
- After completion of the project:
	- There are much more advanced ways to represent text than bag-of-words that may also improve performance. Starting from word2vec and doc2vec type of approaches to pre-trained language models such as BERT.
- During project:
	- Hyperparameter tuning makes a difference. With badly chosen hyperparameters, the results are not great. We did not perform much hyperparameter tuning.
	- The train-test split to be used given in project description was not great (first 80% of reviews as training set, remaining as test set). Essentially this leads to fitting the model and its hyperparameters to the test set – the true level of performance on new data cannot be inferred using this strategy. A better strategy would have been to randomly split the data to two pieces, e.g. one consisting of 80% of data and other 20%. Then running k-folds cross validation cycle in the 80% of data. And when the best model and hyperparameter combination has been found, test the performance in 20% of the data.
	- The evaluation metrics are important to gain a good picture of model performance. Matthews Correlation Coefficient turned out be a good way to summarize the classification performance.
	- We could have employed TF-IDF transformation to the bag of words.

### Data utilized in project:
The dataset consisted of Yelp-reviews, both the files random_reviews.csv and bow1000.csv were given. Please see five first lines of the dataframes below.

### Some results:
In classification it turned out that the simple Logistic Regression was on par with more advanced methods such as Random Forest or Multilayer Perceptron. On the other hand, in predicting usefulness of the review linear regression was confused from the highly collinear data with more attributes than reviews. 

Additionally, we noted that obtaining additional data would have benefitted at least Random Forest in classification – which is no surprise. Furthermore, we noted that Decision Trees overfit easily when the length of the tree is increased – as expected.


**Summary of results from classification**


**Summary of results from predicting usefulness of review**


**Learning curve for Random Forest in classification**

**Tree max depth vs generalization error for Decision Trees and AdaBoost based on trees**
