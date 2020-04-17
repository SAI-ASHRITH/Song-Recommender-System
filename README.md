# Song-Recommender-System
CMPE 256 Programming Assignment
Song Recommender System
Name: Sai Ashrith Aduwala
SJSU Id: 014427725
Kaggle Id: Ashrith
RMSE Score: 15.70972

Data Loading and Pre-processing[1]:
The first step is to import the necessary libraries. The surprise library contains the algorithms used for the training and testing of the data and for evaluating the built model. The train_test_split library contains packages used for splitting the train data into train set and test set. The cross_validate and GridSearchCV libraries contain packages used for parameter tuning.
Once the libraries are imported, the data is read from the files into the dataframes using the packages from the pandas library. This read data is checked for any null values.
As there are no null values, the data is converted to the reader format using the Reader function. After the data is converted, we use the train_test_split function to split the data into train set and test set of required proportions.
Data Pre-processing with Artist data:
Import the required package and read the training data and the artist data into corresponding dataframes. Check the meta-data of training data and artist data for any useful information. We can see that the count of the artist data is less than the training data. First sort the artist data in ascending order of the track ids. Then drop any repeating or duplicate values in the artist data. Create a dictionary with track ids as keys and artists as values. For every track in the training data create a list of artists in the same order. Then append this list to the dataframe as a new column and save it in a file. 
Upon multiple screening and testing with the new training data, the RMSE score would not go below 40. Thus, this data was not considered for the program.

Algorithms[2][3]:
For the initial analysis, I have considered the following algorithms, with their default parameters; BaselineOnly(), KNNBasic(), KNNWithMeans(), KNNWithZScore(), KNNBaseline(), SVD(), SlopeOne(), CoClustering(). After building the respective models with the training set, and testing them with the test set, the following RMSE scores were recorded; BaselineOnly() - 19.6346, KNNBasic() - 18.8269, KNNWithMeans - 17.6878, KNNWithZScore – 17.6097, KNNBaseline – 17.5664, SVD – 16.2095, SlopeOne – 18.5940, CoClustering – 19.6638.
Of all the above algorithms, SVD has the least RMSE score. Thus, I chose SVD for predicting the ratings for the songs. Singular Value Decomposition (SVD) is a matrix-factorization based algorithm from the scikit-surprise library.
Matrix Factorization is a collaborative based filtering method where matrix m*n is decomposed into two matrices of size m*k and k*n, such that if we multiply the factorized matrices, we get back the original matrix with most of the missing ratings filled. Here, the k-values are called factors and can range up to hundreds. The original utility matrix m*n is a sparse matrix, with not many ratings available. But after performing the factorization and multiplying the factorized matrices, the resultant utility matrix will be populated with the predicted rating values for most of the users and items.
To improve the RSME score, I performed offline parameter tuning using cross-validation and hyperparameter tuning using GridSearchCV.
Grid Search is a process of performing hyper parameter tuning in order to determine the optimal parameter values for a given model. The param_grid parameter of the GridSearchCV is used to pass the list of parameters used by the model and the range of values for each parameter. Cross validation is performed in order to obtain the parameter value that gives the best RMSE score.
Parameters used for tuning:
•	n_factors : The number of factors to be used for the matrix factorization i.e value of ‘k’. Default value is 100.
•	reg_all : The regularization value for all the parameters to avoid overfitting. Default value is 0.02
•	n_epochs : The number of iterations for the SGD procedure. Default is 20.
Once the best parameters are obtained, the model is retrained using the entire dataset and the rating values in the test data are predicted and stored in a .csv file in the required format.

References:
[1] https://towardsdatascience.com/building-and-testing-recommender-systems-with-surprise-step-by-step-d4ba702ef80b
[2]The in-class demo files: Surprise_demo.ipynb and Surprise_demo_MF.ipynb
[3] https://surprise.readthedocs.io/en/stable/prediction_algorithms_package.html
