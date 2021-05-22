***Cybersecurity Machine Learning Final Project***
***University of Valencia, Spring 2021***
***by Marc Mestre Cañón and Yana Veitsman***

1. Goal:
Compare 3 types of classifiers - Decision Tree, KN-neighbors, and Support Vector Machine - and their
ability to correctly classify between an example of benign and malicious traffic

2. Dataset:
https://www.unb.ca/cic/datasets/ids-2017.html
Wednesday, July 5, 2017

3. Initial dataset structure:
- 80 features per each sample
- Traffic:
  - 440.031 	Benign Traffic samples
  - 252.673 	Attack samples DdoS
  - 231.073 	Attack DoS Hulk
  -  10.293 	Attack DoS GoldenEye
  -   5.796 	Attack DoS SlowLoris
  -   5.499 	Attack DoS SlowHttpTest
  - 11   Heartbleed
  - 692.703 	All samples

4. Modified dataset:
As our primary goal was to distinguish between an example of benign and malicious traffic,
we artificially modified the original problem according to our needs. That is:
We selected 2 stratified samples of 5.000 samples each and saved them as train.csv and test.csv files respectively
to later use for training and testing.
Then, using the script BenignVSRestScript.py, we relabled the data in a fashion that each sample of 
Benign traffic was given a label value of 1 and each sample of attack traffic was given a label value
of 0

5. Preprocessing method:
In KN-neighbors and SVC we use MinMaxScaler as a preprocessing method, since our data features have a very big range of values
(for example, some of them are the quantity of bits/packages flowing). 
According to the sklearn library's manual, MinMaxScaler "transforms features by scaling each feature to a given range.
This estimator scales and translates each feature individually such that it is in the given range on the training set, 
e.g. between zero and one."

6. Decision Tree:
For a decision tree classifier we selected the most basic decision tree available in sklearn library.
No preprocessing or any other preparation of data was done for this classifier.
Corresponding file: DecisionTree.py

7. KN-neighbors:
We used the KN-neighbors classifier available in sklearn library. We considered a number of neighbors varying from
1 to 20, as well as preprocessing and no preprocessing of data with MinMaxScaler.
Corresponding file: KNeighbors.py

8. SVC:
We used a basic SVM - SVC - available in sklearn library. This type of classifier, according to sklearn library,
works worse with datasets that have more than 10.000 samples, but our datasets were fixed to 5.000 samples for training
and testing. 
- Firstly, we evaluated the model with no preprocessing.
- Secondly, we introduced preprocessing with MinMaxScaler.
- Thirdly, we looked for the best pair of values of C and gamma for the classifier.
Corresponding files: SVM.py and SVM_gamma_and_C_selection.py
