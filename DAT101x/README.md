Microsoft: DAT101x Introduction to Data Science
 
Read  : https://docs.microsoft.com/en-us/azure/machine-learning/team-data-science-process/ 


# Table of Contents
1. [Module 1: An Introduction to Data](#module-1-an-introduction-to-data)
2. [Module 2: Data Analysis Fundamentals](#module-2-data-analysis-fundamentals)
3. [Module 3: Getting Started with Statistics](#module-3-getting-started-with-statistics)
4. [Module 4: Machine Learning Basics](#module-4-machine-learning-basics)




## Module 1: An Introduction to Data

### Exploring Data :
#### Getting Started with Data :

Temporal Data
Text 
    Categorical
Numerical Varable 
	Continuous 
    Discrete
#### Sorting and Filtering Data
    Ascending 
    Descending

    Outlier Value : Exteme Value 

#### Derived Data 
    Example  : Revenue = Sales * Price 

#### Highlighting Data 
    Higlight comparative numeric value 
        Using color intentiity  - also called Heat Map 
        Using relative size uisng Data bar 
        Using certain criteria 

## Module 2: Data Analysis Fundamentals  
### Analyzing and Visualizing Data   
#### Aggregating Data
Aggregating data is usually the first kind of analysis you perform in order to summarize the data and get a feel for it as a whole .

* How many records are in the Data Set  - Count 
* For categorical data, you might want to calculate something called the distinct count; which counts how many distinct values that are.
    
    - Sum 
    - Average
    - Minimum 
    - Maximum 


#### Grouping and Summarizing Data
When your dataset includes categorical values, it's common to group the data by categories and calculate subtotals for numeric values

When there are multiple categories, you can group them into hierarchies to get multi-level subtotals.

#### Visualizing Data
It's generally easier to identify and explore trends and relationships in your data by creating visualizations - variously referred to as graphs plots or charts. 

* Line Chart  :
    - X axis 
    - Y axis 
    - Legend 
    - Tick Labels
    - Title 

* Pie Chart :
* Bar Chart /Column Chart  - Bar charts are often used to compare numeric values across categories. You can compare across multiple
* Scatter Plot : To compare two numeric values, you can use a scatter plot 



## Module 3: Getting Started with Statistics
### An Introduction to Statistics
#### Measuring Central Tendency
    Descriptive Statistics
    Range (Maximum -Minimum)  
    Mean 
    Sample Mean (Bar over )
    Median - Middle Value
    Mode - Most Common Value

    Frequecy of Value  (Histogram) 
    Bell Curve - Normal Distribution 
    Box and Whisker Plot 

#### Measuring Variance
    Population variance as Sigma-squared
    Variance  : We calculate the sum of the squared differences between each value and the mean, and then we divide that by the number of observations.
    * Excel : VAR.P()

    For a sample of data :
    We calculate the sum of the squared differences between each value and the Sample mean, and then we divide that by the number of observations - 1 
    * Excel : VAR.S()

    Standard Deviation  of Population  :  sigma
    Square root of the variance 
    * Excel : STDEV.P()

    Standard Deviation  of Sample  : s 
     * Excel : STDEV.S()


In a normal distribution 

    * the mean is in the middle and

    * Around 68.26% of all observations lie within one standard deviation above or below the mean. 

    * Around 95.45% of observations lie within two standard deviations of the mean; 

    * Around 99.73% of observations lie within three standard deviations of the mean.

#### Skewed Distributions

centrality of the mean

the shape of  chart is a curve with a long tail on the right, and we therefore call this a *right-skewed distribution*

it's also possible
for the data sample to be skewed by some low values, creating a tail to the left;
and naturally enough we call this a *left-skewed distribution* .

#### Working with Samples
* working with a full population of data 
* working with a samples of data that are representative of the data, and you use sample statistics such as the mean and standard


In practice 
* it's best to get as large a sample as you can. The larger the
sample, the better it will approximate the distribution and parameters of the  full population

* Another thing you can do is to take multiple random samples.
Each sample has a sample mean, and you can record these to form what's called a sampling distribution.

    With enough samples, two things happen. 
        
        One is that, thanks to something called the central limit theorem, the sampling distribution takes on a normal shape regardless of the shape of the population distribution;
            
        the second thing is that the mean of the sampling distribution, in other words the mean of all the sample means,will be the same as the population mean.


#### Correlation

Correlation is a value between negative 1 and positive 1 that indicates the strength and direction of a relationship. 
* A value near positive 1 indicates a positive relationship in which an increase in one value generally correlates with an increase in the other. 
* A value near negative 1 on the other hand indicates a negative correlation in which an increase in one value tends to be associated with a decrease in the other value.

Note: correlation is not the same thing as causation

#### Hypothesis Testing
Null Hypothesis
Alternative Hypothesis

Significance level (alpha) : that indicates the probability threshold under which we
will reject the null hypothesis in favor of the alternative hypothesis 

if we know the standard deviation of the full population we conduct something called a Z test, 
but if we don't have that information (and typically we don't) we conduct a T-test.

Z test 
    * p-value 
T-test 


the p-value we get from our test is 0.02, which is smaller than our
significance level of 0.05 or 5% . 
That means that the probability of observing a mean as extreme as our sample mean by chance is less than the threshold we set; 
so in this case we would reject the null hypothesis in favor of the alternative hypothesis


#### A Handy Handout


## Module 4: Machine Learning Basics
### Introduction to Machine Learning
#### What is Machine Learning?
Machine learning is a way of using data to train a computer model so that it can predict new data values based on known inputs. 

Supervised : 
f(features value) = label   
Unsupervised 

#### Regression
Supervised Learning 

f(features value) = label 

Train Data :

Evaluation Data  : 
    Predict Y (Prediction or Scored Label) for given set of X
 

Errors or Residual :  Y predicted Label - Y Actual label 

Measure of Error 

Absolute Measure 

    root mean squared error (absolute measure)
    RMSE = sqrt[Summantion(score-label)^2]


    mean absolute error (absolute measure)
    MAE = 1/n * Summantion[abs(score-label)]

Relative Measure

    * Relative absolute error is the MAE relative to the mean value of the label
    RAE : Summantion[abs(score-label)] /Summantion [label]

    a relative value for the error within a scale of zero and one;with zero being a perfect model.


    * relative squared error, which is the RMSE divided by the sum of the squares of the label
    RSE : sqrt[Summantion{abs(score-label)}^2 /Summantion {label}^2]

    * The coefficient of determination, which is also known as the R-squared of the model, represents the predictive power of the model as a value between 0 and 1. 
    A value of 0 means that the model is random and has learned nothing about the data. A perfect fit would result with a value of 1.
    CoD(R^2) = 1-var(score-label)/var(label)



#### Classification

Suprevised Learning 

Binary Classification (f([x1,X2,X3])) = Y[1|0]

Threshold Value help us to detemine :

The function won't actually calculate an absolute value of
1 or 0, it will calculate a value between 1 and 0 and we'll use a threshold value to decide whether the result should be counted as a 1 or a 0. 

When you use the model to predict values, the resulting value is classed as a 1 or 0 depending on which side of the threshold it falls


* *True Positive*  : Cases where the model predicts a 1 or a test observation that actually has a label of 1 are considered true positives. 
* *True Negatives* : cases where the model predicts 0 and the actual label is 0, these are true negatives. 
* *False positive*: If the model predicts 1 but the actual label is 0, well that's a false positive;
* *False Negative* : if the model predicts 0 but the actual label is 1, that's a false negative. 


Confusion Matrix 

* Accuracy = (TP + TN) /(TP+FP+TN+FN)

* Precision = TP /(TP+FP)

* True Postive Rate or  Recall = TP /(TP+FN)

* False Positive Rate = FP /(FP+TN)

* receiver operating characteristic, or ROC Chart  - FPR vs TPR for all threshold value 

* Area Under the curve (AUC) of ROC chart

        A perfect classifier would go straight up the left-hand side and then along the top giving an AUC of 1

#### Clustering
Unsupervised Learning 

* In unsupervised learning techniques, you don't have a known label with which to train the model. You can still use an algorithm that finds similarities in the data observations in order to group them into clusters.

K-means clustering 

To measure this :

* we can compare the average distance between the cluster centers.And the average distance between the points in the cluster and their centers.

Clusters that maximize this ratio have the greatest separation.

* We can also use the ratio of the average distance between clusters, and the maximum distance between the points and the centroid of the cluster.

* Now another way we could evaluate the results of a clustering algorithm is to use a method called principal component analysis, or PCA.

        in which we decompose the points in the cluster in two directions. We can represent the firsttwo components of the PCA decomposition as an ellipse. The first principal component is the direction along the maximum variance, or major axis of theellipse; 

        and the second is along the minor axis of the ellipse. If we decompose another cluster that's perfectly separate from the first cluster, we would get an ellipse with the major axis of the ellipse perpendicular to the first cluster. 
        
        If our second cluster is reasonably well, but not perfectly, separated; we'll have a second ellipse with the major axis no longer perpendicular, but certainly in a different direction from the first ellipse; 
        
        and finally if the second cluster is very poorly separated from the first, the major axes of both ellipses will be nearly collinear. Notice also that the ellipse for the second cluster is more like a circle because it's less well defined.

#### Some Useful Resources
* https://stattrek.com/ 

* https://docs.microsoft.com/en-us/azure/machine-learning/studio/data-science-for-beginners-the-5-questions-data-science-answers

