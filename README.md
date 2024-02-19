# The Expected Lifespan of a Person at Different Stages of Life
Our main Objective is to reveal the complex connections and patterns within the dataset. We intend to identify the influence of various variables by utilizing statistcal approaches, correlation, data visualization and to conduct a comprehensive analysis of life expectancy data, exploring the relationships between various socio-economic, environmental, and healthcare factors.
STAT 650 - Final Project
LIFE EXPECTANCY DATA
Developed by (Name: Hasitha Varada - 734004713)

Introduction
A person's average life expectancy is determined by a statistical calculation that takes into account their current mortality rate as well as other demographic characteristics. It offers information on the duration and quality of life and is a crucial measure of the general health and well-being of a population. Years are frequently used to describe life expectancy, which is determined for various demographic groupings, geographical areas, and nations. Life expectancy is influenced by a number of elements, which may be roughly classified into social, economic, environmental, and healthcare-related aspects. Living circumstances, sanitation, and pollution exposure are examples of environmental elements; income, education, and resource accessibility are examples of social and economic factors. Preventive measures, illness treatment, and the availability and caliber of healthcare services are all considered healthcare considerations.

GOAL - A key indicator of the general well-being and wealth of a society is life expectancy that is a crucial parameter that measures the typical number of years a person can expect to live. Our goal is to pinpoint the primary factors that have a major influence on life expectancy using regression analysis and statistical modeling. 1) There is a significant association between alcohol consumption and mortality rates among adults. 2) The average BMI of the population is positively correlated with life expectancy. 3) The life expectancy of people is influenced by a combination of socio-economic factors, health-related indicators, and vaccination coverage

OBJECTIVE - Our main Objective is to reveal the complex connections and patterns within the dataset. We intend to identify the influence of various variables by utilizing statistcal approaches, correlation, data visualization and to conduct a comprehensive analysis of life expectancy data, exploring the relationships between various socio-economic, environmental, and healthcare factors.

BACKGROUND - This research is important because it adds to the body of knowledge already available on life expectancy and because it has applications for policymakers who want to create focused initiatives that will enhance population health outcomes. Understanding the factors that affect life expectancy can help us develop public health initiatives that are more successful.My main interest in health care domain led me to take this dataset to know more about the cause of deaths in the world.

Data Description
Our data source is used from kaggle. The selected dataset is in csv format.https://www.kaggle.com/datasets/kumarajarshi/life-expectancy-who/code

Data Collection

Variables

1.Status - Nominal Categorical Variable

2.Country - Nominal Categorical Variable

3.Adult Mortality - Discrete Quantitative Variable

4.Infant Deaths - Discrete Quantitative Variable

5.Alcohol - Continuous Quantitative Variable

6.BMI - Continuous Quantitative Variable

7.Polio - Discrete Quantitative Variable

8.HIV/AIDS - Continuous Quantitative Variable

9.GDP - Continuous Quantitative Variable

10.Under five- deaths - Discrete Quantitative Variable

11.Population - Discrete Quantitative Variable

12.Schooling - Continuous Quantitative Variable

13.Life Expectancy Data - Continuous Quantitative Variable

14.Income composition of resources - Continuous Quantitative Variable

We have collected the data and imported the libraries necessary for the EDA .

Data Preprocessing:
By importing the data we have got 5 rows * 22 columns

By checking the shape of the data we got output of (2938, 22)

By checking the types of data = dtypes: float64(16), int64(4), object(2) memory usage: 505.1+ KB table3.jpg

For quantitative variable, generate a table for the count, mean, standard deviation, minimum and maximum values and the quantities of the data we got output

table.jpg

table2.jpg

By checking unique values of different variables

Population = [33736494. 327582. 31731688. ... 125525. 12366165. 12222251.]

It represents a series of distinct population values in the DataFrame's "Population" column. For every integer in the array, there is a unique population value associated with that column. With numbers ranging from 33 to 1.29e+09 (1.29 billion), the average population is 1.28e+07 (12.8 million).

Alcohol = [0.01 0.03 0.02 ... 2.44 3.56 4.57]

It represents a sequence of unique alcohol consumption values in the 'Alcohol' column of the DataFrame. Each number in the array is a distinct alcohol consumption value found in that column.The average amount of alcohol consumed ranges from 0.01 to 17.87, with a mean of roughly 4.60. The standard deviation of 4.05 indicates that patters of alcohol intake can vary.

Adult Mortality

It represents a sequence of unique adult mortality values in the 'Adult Mortality' column of the DataFrame. The adult mortality rate ranges from 1 to 723, with an average of about 164.80. The distribution has a standard deviation that is rather high (124.29), indicating that adult death rates may vary.

Let's make a sub-dataset from our original dataset for our research objective, goals by dropping unnecessary variables and use this for further process.

Missing values

There are missing values in variables like alcohol, population, polio, BMI, Schooling.

We have handled the new dataset by checking if they are less than 40%

we are handling missing values for Alcohol- (2744, 13)

we are handling missing values for BMI- (2727, 14)

we are handling missing values for Population- (2115, 13)

we are handling missing values for Polio- (2108, 14)

we are handling missing values for Schooling- (2108, 14)

Duplicated values There are no duplicates in this dataset - 2107 rows × 14 columns

Checking Outliers

Adult Mortality no of outliers = : 66 and we removed it.

adult.png

Under-5 deaths - no of outliers = : 209 we removed it.

under5-2.png

Life expectancy no of outliers = : 5 and we removed it.

life.png

Exploratory Data Analysis
Univariate analysis
Histogram for life expectancy of people

Univariate analysis for finding how the life expectancy of people are distributed. Here we can infer that life expectancy of men after 60 is less than women.Life expectancy is plotted on the x-axis, and the number of people in each life expectancy bin is plotted on the y-axis. The distribution is depicted by the red bars in the histogram, and a smooth approximation of the probability density function is given by the KDE plot. According to the legend, the figure displays statistics for "People." count.png

Plot the countplot of status

In the 'Status' column, there are two categories that normally indicate whether a country is regarded as "Developed" or "Developing." The countplot shows the distribution of these categories visually. From this plot we can infer that more than half of the countries are developed fully and double the amount of countries are still under development.status.png

Plot the distplot of alcohol variable

The distplot shows information about the central tendency, spread, and distributional shape of the values in the 'Alcohol' variable. From this distribution plot we can infer that the range of alcohol has increased from 0.08% to highest 0.12%. The density is varying here and there at 0.04%. alcoholgraph.png

Plot the distplot of Polio variable

The histogram component of the distplot visualizes the frequency or count of different ranges (bins) of Polio values. From this distribution plot we can infer that the range of Polio or the rate of people died due to polio vaccine has increased from 0 to highest 0.20 density.The density is varying drastically so we can't take average here. The peak is attained at 0.10. poliograph.png

Plot the distplot of schooling variable

The histogram component of the distplot visualizes the frequency or count of different ranges (bins) of schooling values. The KDE overlay smoothens the histogram, providing an estimate of the probability density function for the variable. From this distribution plot we can infer the range of students who have completed schooling and we can see the drastic change as the graph increased and again decreased completely to 22.5. The highest point it reached was more than 0.175. schoolinggraph.png

Histogram for the distribution of population

The y-axis shows the frequency or count of observations falling into each bin, while the x-axis shows various population value ranges, or bins. The population plot has decreased as it was very high as 450 in the first and the frequency decreased to less than 100. populationgrapg.png

Bi-variate analysis
Scatter plot: Adult Mortality vs Infant and under-5 children Deaths

The scatter plot visually compares "Adult Mortality" with the number of "Infant Deaths" and "Under-Five Deaths" for each data point. It suggests a link between adult mortality and the incidence of baby and child fatalities. In other words, if adult mortality rises, baby and child fatalities likely to rise as well. adultmortality.png

Plotting a bar graph for showing average adult mortality rate of people in all countries

Using a bar graph we have shown the average mortality rate of people with regarding to the countries. The bar graph displays the average adult mortality rate for each country, providing a visual comparison. The x-axis represents different countries, and the y-axis represents the average adult mortality rate. From this we infer that the highest performer is Kazhakasthan and the lowest is country Tunisia. The average adult morality rate kept fluctuatingbetween 50 and 250. mortalityrate.png

Histograms for all the variables

Here we can depict the histograms for all the selected variables and can see the different type of graphs increasing and decreasing. The life expectancy of people have been constantly high and the adult mortality or the death rates have also been high in same time. Comparing the schoolinng, population, Hiv we can infer that it has decreased completely. allvariable.png

Scatter plot: Life expectancy vs HIV

For every data point, the scatter plot shows the relationship between "Life Expectancy" and "HIV/AIDS" graphically. The x-coordinate in the plot indicates the prevalence of HIV/AIDS, while the y-coordinate indicates life expectancy. Each point in the plot represents a nation or observation. Now as we are doing for the bivariate analysis we are comparing the life expectancy to the HIV disease through which many people died. When compared among people we can say that the life expectancy of people have decreased because of HIV and it was of same level and did not increase or decrease drastically. people.png

Multivariate analysis
Correlation matrix for all the variables

The correlation matrix is represented visually in a heatmap, where the color of each cell denotes the direction and strength of the correlation between two variables. A perfect negative correlation is denoted by a correlation coefficient of -1, a perfect positive correlation by a correlation coefficient of 1, and no correlation is denoted by a correlation coefficient of 0.From this heatmap we can infer that there is a positive correlation between infant deaths and under five deaths. The blue ones have negative correlations. There is only one positive correllation as it increases. Other than that most of them have no correlations at all. correlation.png

Through correlation between the life expectancy of people and their death rates

The pair plot visually represents the relationships between the selected attributes ('Life expectancy' and 'Adult Mortality'). Scatter plots on the upper and lower triangles show bivariate relationships where it keeps varying between 0-300 while histograms on the diagonal show the univariate distributions. From this heatmap we can get the correlation between these two variables. Comparing different variables we can infer that red ones are all correlated. matrix.png

matrix1.png

Heatmap for finding correlation between selected variables ['BMI', 'HIV', 'Polio', 'Alcohol']

The correlation matrix is shown in a heatmap that is tailored to the attributes that have been chosen (BMI, HIV/AIDS, Polio, Alcohol). The annotations and color map 'coolwarm' offer information about the direction and strength of the correlations. From this heatmap we can get the correlation matrix in a visualized mode. Comparing different variables we can infer that red ones are all correlated and the blue ones are less than 1 and are negatively correlated. There are many neutral non correlated variables. attributes.png

Methodology
Three different Hypotheses and two of them can be shown during Exploratory Data Analysis
1)There is a significant association between alcohol consumption and mortality rates among adults.

The correlation matrix is shown in a heatmap that is tailored to the attributes ('Adult Mortality' and 'Alcohol').The correlation between 'Adult Mortality' and 'Alcohol' is -0.20. A weak negative correlation between "Adult Mortality" and "Alcohol Consumption" is indicated by the negative correlation coefficient of -0.20. This indicates that there is a small tendency for adult mortality rates to slightly decline as alcohol consumption rises on average. The correlation is weak, though, and adult mortality rates are probably more heavily influenced by other factors. Here we can infer that the correlation between alcohol and mortality rates of adults are significantly associated as the increase in alcohol consumption leads to more death rate. attributess.png

2)The average BMI of the population is positively correlated with life expectancy.

The correlation between 'BMI' and 'Life expectancy' is 0.24. A weak positive correlation between "BMI" and "life expectancy" is indicated by the positive correlation coefficient of 0.24. This indicates that there is a small tendency for life expectancy to rise in combination with an increase in body mass index (BMI). The correlation is weak, though, and life expectancy is probably influenced by other factors to a greater extent.

bmipopulation.png

FOR THIRD HYPOTHESES WE WILL USE REGRESSION MODELS

3)The life expectancy of people is influenced by a combination of socio-economic factors, health-related indicators, and vaccination coverage

According to this hypothesis, life expectancy is a complicated result that is influenced by a number of variables. Modeling life expectancy as a function of several independent variables, such as ' HIV/AIDS', 'Income composition of resources', 'Adult Mortality', ' BMI ', 'under-five deaths ' and other pertinent variables, is possible with multiple linear regression.

Dependent Variable: 'Life expectancy'

Independent variable: ' HIV/AIDS', 'Income composition of resources', 'Adult Mortality', ' BMI ', 'under-five deaths '

STATISTICAL METHODS
T-TEST

We have done T-test to compare the means of two groups. We have chose the variables need for our hypothese and for regression analysis over here. They are status and life expectancy, to check that they both are significantly different. We have infered that one set og group of countries are still under development and another part of group are well developed. This might affect the life expectancy.

T-statistic: 21.988117605025288

P-value: 5.241763508367791e-84

As a result we have reject the null hypothesis. There is significant difference.

I have use used 4 regression models for my dataset

a. Multiple Linear Regression

b. Lasso Regression

c. Ridge Regression

d. Elastic Net Regression

MODELLING TECHNIQUES
MODEL FITTING

Justification: Relevant characteristics (predictors) like ' HIV/AIDS', 'Income composition of resources', 'Adult Mortality', ' BMI ', 'under-five deaths' are used as independent variables in the models' fitting process, while Life expectancy data serves as the dependent variable.

Training and Testing Data: To assess model performance, the dataset is divided into training and testing sets. table4.jpg

table5.jpg

Regularization: Regularization techniques (Lasso, Ridge, Elastic Net) are employed to prevent overfitting by penalizing large coefficients. The regularization strength (alpha) is tuned for optimal performance.

REASONS FOR SELECTION
1. MULTIPLE LINEAR REGRESSION: When dealing with factors influencing life expectancy, linear regression is useful. It gives information on the unique contributions made by every predictor.

2.RIDGE REGRESSION: This method stabilizes the model by avoiding excessively high coefficients, making it useful when multicollinearity among the predictors is suspected.

3.LASSO REGRESSION: Excellent for feature selection, since it aids in determining the key variables that have the greatest influence on life expectancy.

4.ELASTIC NET REGRESSION: An optimal option for utilizing both Lasso's feature selection powers and Ridge's advantages in coefficient stability.

Model Evaluation and Selection
Regression approach: Multiple Linear Regression

Linearity: In my dataset it indicates that the response variable and the predictors have a linear relationship. They are linear. Residuals are independent of each other. Residuals are normally distributed.

Regression approach: Ridge Regression

We can assume for this regression that the default alpha value should be 1. The tolerance statistic is the reciprocal of the VIF. Low tolerance values (< 0.1) indicate a problematic level of multicollinearity.

Regression approach: Lasso Regression

The relationship between the independent variables and the dependent variable is linear. We can assume for this regression that the default alpha value should be 1. The tolerance statistic is the reciprocal of the VIF. Low tolerance values (< 0.1) indicate a problematic level of multicollinearity. Residuals are independent of each other.

Regression approach: Elasticnet Regression

We can assume that the variance of residuals is constant across all levels of the independent variables. In other words, the spread of residualsare consistent. Similar to Lasso, this encourages sparsity in the model by penalizing some regression coefficients to exactly zero.

IMPLEMENTATION OF MODELS

Linear Regression:

R-squared (R2): 0.4797813798878948

Mean Squared Error: 15.582571770841325

Root Mean Squared Error: 3.9474766333496296

Mean Absolute Error: 2.943524776013167

The R-squared value of 0.48 indicates that the model explains a moderate amount of the variability in the dependent variable.

On average, this model's predictions are off by approximately 2.94 units in the same scale as the dependent variable.

Ridge Regression:

Ridge CV Model Mean Squared Error: 15.600950316094979

Best Alpha after Cross Validation : 0.4552935074866948

Mean Squared Error: 22.753070107554628

Root Mean Squared Error: 4.770017830947242

Mean Absolute Error: 3.5408274237391044

R2: 0.24039684150115204

This is the mean squared error of the Ridge regression model after cross-validation. It indicates the average squared difference between the predicted and actual values, and a lower value is desirable. The MSE is 15.60.

R-squared measures the proportion of the variance in the dependent variable that is predictable from the independent variables. R-squared is 0.2404, indicating that about 24% of the variability in the dependent variable.

Lasso Regression

Mean Squared Error: 16.501150276010712

Root Mean Squared Error: 4.062160789039586

Mean Absolute Error: 3.0301120674742834

R2: 0.44911496297986864

MSE is the average of the squared differences between predicted and actual values. It provides a measure of the average squared error. The MSE is 16.50, indicating the average squared difference between predicted and actual values.

R-squared measures the proportion of the variance in the dependent variable that is explained by the independent variables. R-squared is 0.4491, indicating that approximately 44.91% of the variability in the dependent variable is explained by model.

Elasticnet Regression

Best alpha: 0.1 Best l1 ratio: 0.5 ElasticNet(alpha=0.1)

Mean Squared Error: 19.964476408538058

Root Mean Squared Error: 4.468162531571346

Mean Absolute Error: 3.2848263681720486

R2: 0.33349305100299254

The l1 ratio determines the balance between L1 and L2 regularization. A ratio of 0 corresponds to L2 regularization, 1 to L1 regularization, and values in between to a combination of both. The best l1 ratio is 0.5, indicating an equal balance between L1 and L2 regularization.

R-squared calculates the percentage of the dependent variable's variation that can be accounted for by the independent variables. R-squared is 0.3335, that the ElasticNet model accounts for around 33.35% of the variability in the dependent variable.

Testing model assumptions for their linearity

1.LINEAR REGRESSION

LINEARITY1.png

2.RIDGE REGRESSION

RIDGE2.png

3.LASSO REGRESSION

LASSO.png

4.ELASTICNET REGRESSION

ELASTIC.png

Model Comparison
After developing the model, I have evaluated its performance using appropriate metrics. I have included R-squared, mean squared error, root mean squared error. I have compared the metrics.

By comparing metrics MSE

Mean Squared Error:

Linear Regression: 15.582571770841325

Ridge Regression: 22.753070107554628

Lasso Regression: 16.501150276010712

Elastinet Regression: 19.964476408538058

The model with the lowest average squared difference (MSE) between the predicted and actual values is linear regression. In terms of MSE, it performs better.

By comparing metrics of RMSE

Root Mean Squared Error:

Simple Linear Regression: 3.9474766333496296

Ridge Regression: 4.770017830947242

Lasso Regression: 4.062160789039586

Elastinet Regression: 4.468162531571346

Linear Regression has the lowest RMSE, indicating the smallest average magnitude of errors. It performs better in terms of RMSE

By comparing metrics of MAE

Mean Absolute Error:

Simple Linear Regression: 2.943524776013167

Ridge Regression: 3.5408274237391044

Lasso Regression: 3.0301120674742834

Elastinet Regression: 3.2848263681720486

Linear Regression has the lowest MAE, indicating the smallest average absolute difference between predicted and actual values. It performs better in terms of MAE

Across all three metrics (MSE, RMSE, MAE), Linear Regression consistently performs the best among the models compared.

In this instance, regularized models (such as Ridge, Lasso, and ElasticNet) have more errors than Linear Regression because they impose penalties for model complexity. A trade-off between simplicity and complexity may need to be made when selecting amongst these models.

MAE.png

MEAN.png

MAES.png

Model Improvement
Hyperparameter tuning for linear regression typically involves tuning the regularization strength (alpha) in models like Ridge or Lasso.

After hyperparamater tuning:

Ridge MSE: 15.628590025999792

Lasso MSE: 15.631335078226114

Comparable Performance: Ridge and Lasso function similarly, and there may not be a big difference in their forecasting accuracy.

Impact of Regularization: Preventing overfitting and offering a fair balance between bias and variance are probably achieved by the regularization strength (alpha) selected during hyperparameter tuning. In our dataset

Results and Interpretation
Performance:

In comparing the performance of the regression models applied to the life expectancy data, three key evaluation metrics were considered: Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and Mean Absolute Error (MAE). The models under consideration were Simple Linear Regression, Ridge Regression, Lasso Regression, and ElasticNet Regression.

The Linear Regression model consistently outperformed the regularized models across all three metrics. It exhibited the lowest MSE, RMSE, and MAE, indicating smaller average prediction errors and better overall accuracy. This suggests that, the linear model without regularization provided a more effective representation of the relationship between predictor variables and life expectancy.

Statistical Significane

This comparison's t-statistic was determined to be 21.99, and the corresponding p-value was around 5.24e-84. We rejected the null hypothesis because the p-value was very low—much less than the typically accepted significance criterion of 0.05. This suggests that the two sets of nations' life expectancies differ statistically significantly. The conclusion is that life expectancy is significantly influenced by a country's development level, as shown by the 'level' variable. The results of the subsequent regression analyses allowed for a more nuanced investigation of the relationships between life expectancy and multiple predictors, taking into account both statistical and practical significance. The t-test revealed a significant difference in life expectancy between developing and developed countries.

Interpret coefficients of significant predictors

For our dataset we know that the best model is Linear regression. In Linear regression model, significant predictors such as HIV/AIDS', 'Income composition of resources', 'Adult Mortality' exhibited notable coefficients. For instance, a higher coefficient for Adult Mortality implies a more substantial impact on predicted life expectancy. Similarly, Hiv or Bmi might have shown significant positive or negative coefficients, indicating their influence on predicted Life expectancy.

The positive sign suggests that as the 'Income composition of resources' increases, life expectancy tends to increase. The magnitude (0.5) indicates that for every one-unit increase in 'Income composition of resources', life expectancy is expected to increase by 0.5 units, all else being equal.

The effect of 'Adult Mortality' on life expectancy might be different at different levels of 'BMI' or 'HIV/AIDS'. If the coefficient is 0.5, it means that for every one-unit increase in the 'Income composition of resources', life expectancy is expected to increase by 0.5 units, holding other variables constant.

Higher adult mortality rates are associated with lower life expectancy, which is an expected and logical relationship. This suggests that addressing adult mortality is crucial for improving life expectancy.

A higher BMI may be associated with better nutritional status, contributing to improved health and potentially higher life expectancy. Higher under-five deaths are negatively associated with life expectancy, emphasizing the importance of addressing child mortality for overall population health.

Conclusion
The significance of features:

The value of numerous characteristics in predicting life expectancy has been evaluated using regression models such as Lasso, Ridge, and Linear Regression. Notable contributions include adult mortality, BMI, HIV/AIDS prevalence, and immunization against diphtheria and polio.

Life Expectancy and Development Status:

A substantial difference was found when the life expectancy means of industrialized and developing nations were compared using a t-test.Life expectancy is strongly influenced by a nation's level of development, indicating that socioeconomic issues are important.

Model Execution:

Based on hyperparameter adjustment, it has been determined that ridge regression performs better than Lasso, as shown by a smaller Mean Squared Error (MSE). This implies that a regularization strategy like Ridge, which strikes a compromise between model complexity and performance, is better for the particular dataset.

Predictive Knowledge:

Based on the chosen features, the project offers prediction algorithms that can calculate life expectancy. Policymakers, medical professionals, and academics may all benefit from using these models to comprehend and predict changes in life expectancy.

The model which we used has satisfied the hypotheses and has given best result.The chosen regression models not only satisfy the proposed hypotheses, but also offer strong evidence in favor of them. The predictors that have been found are in line with predictions, which enhances our comprehension of the variables that affect Life Expectancy in the provided dataset.

limitations of the project

Data Quality:

The dataset I have chosen has many missing values and is very tough to remove them. Potentially they affect the reliability of the results. Future research should involve more extensive data cleaning and imputation methods to enhance data quality.

Model Complexity:

The regression models used are simplified representations of complex interactions. Future research could explore advanced machine learning models or ensemble techniques to capture intricate relationships more accurately.

Limited Time Frame:

This analysis is confined to a specific time frame, and life expectancy trends might evolve over time. Future research could explore longitudinal data to capture temporal variations and trends.

To overcome this limitation I would recommend:

Intercultural Research:

Investigate cross-cultural differences by carrying out in-depth research on certain areas or nations, taking into account distinctive cultural and socioeconomic factors that influence life expectancy.

Verification and Extrapolation:

Test the created models on external datasets to improve the models' generalizability and determine how reliable the results are for a range of population types.

 
