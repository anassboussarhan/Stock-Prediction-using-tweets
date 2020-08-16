
                                                            #Stock Predictions using Tweets

# 1 Loading the data :
I have decided to load the stock prices  using  the Yahoo finance API (via Pandas) and would suggest Tweepy to load the tweets instead of using a predefined CSV file ,for the AAPL stock as an example (due to it's popularity) .
NB:1-As my Tweeter programmer account does have some restrictions in regard of the number of tweets to load ,I have decided to generate a uniform sentiment score (between -1 and 1 ) and use it as an input for my models .
2-Due to the fact that the tweets data set doesn't contain the tweets for the first two years (only 5 years of data) ,I would suggest to drop the first two years in the price data set or to fill the missing sentiment scores .
3-If I had to work with CSV files , I would have to merge the data sets (With Pandas) using the dates as an index .
2 Extracting the sentiment polarity score :
To extract the sentiment score I would suggest the use of the Vader sentiment analyzer , it's a rule based sentiment analyzer , and it has shown good results on social media data because it takes into account emoticons ,emojis and punctuations which plays an important role in how people express there opinions in social media as you can explore in more details in the article behind Vader http://comp.social.gatech.edu/papers/icwsm14.vader.hutto.pdf , then aggregate the polarity score for each day(by averaging).
NB : No pre-pocessing has to be done on the tweets such as stop words removal or lemmatization because as stated before ,the Vader analyzer takes into account this information .

                                                                                              


# 2 Data Exploration :
Plotting the daily closing price data proves that the price time series shows some trend , the Augmented dicky fuller test confirm it as the test statistic is superior to the critical value ,which implies that I failed to reject the null hypothesis and the non stationarity of the time series .After differencing the closing prices for the first and second degree , I succeeded to reject the Null hypothesis and then was able to find the p and q parameters for the ARIMAX model (ARIMA with exogenous variable , the sentiment score ) using the autocorrelation and partial autocorrelation plots .Beside that the Granger causality test supports the use of the exogenous variable ,as I have failed to reject the null hypothesis ,then was able to state that sentiment score granger cause the price movement significantly .(Doesn't imply a strict causality)
NB:1- A cointegration test(Johansen test) may be added to this arsenal to confirm a long term equilibrium between the sentiment score and the price closes ,  it may suggest a contrarian move in case of a divergence between the two variables but it's out of the scope of our problem.
2- I have used statsmodels library to fit the ARIMAX model.



                                                           
# 3 Prepocessing : 
To test the machine learning models against unseen data , I have decided to split my data set into two train-test subsets ( using the 80 -20 rule) , I fitted my  models on the train set , and did the tests on the test set, I have also created  a new stationary feature ( closes(t)-closes(t-2) ) to fit the ARIMAX model .
Supervised learning modelling :
To train the supervised learning models , I created a label by shifting the closing prices by 1 step ,and choosed the new feature created to the train the ARIMAX model as an input feature because it is scaled and LSTM models tend to be sensitive to non scaled data , I would highlight that sentiment scores are already scaled.
The shape of our new data set for supervised learning modelling :
X= (Differentiated Closing price(t) , sentiment(t))
Y=Differentiated Closing price (t+1)

# 4 Training the models : 
I fitted the ARIMAX model using the Box Jenkins method, and trained  the LSTM models family:Vanilla, Bidirectionnal and StackedLSTMs using Keras ( LSTM models are suited for sequence data such as time series because they can retain long term interrelations using memory gates)  .I have also used XGBOOST  being inspired by some papers that showcase it superiority to the ARIMA model . 
More models could be used such as Nonlinear autoregressive exogenous model that are inspired from control theory ,LSTMCNN  and LSTMConvLSTM, we could also finetune the models hyperparameters using Grid search for example. 
The results on the test set shows that the stacked LSTM model , gives the best results in term of the mean squared error as it was suggested by many scientific papers.

# 5 Implementation :
To implement the model , we need to predict the transformed label each day, then apply an inverse transformation (inverse diffirentiation) to predict the future closing price , the model needs also periodical re-training as the distribution of time series tend to shift during time , the periodicty of the re-training needs to be optimized also.

# 6 Other factors that may improve our model:
As an hobbyist trader I would use my domain expertise and brainstorm some factors that may or not have an impact on future price mouvements , feature selection methods such  as L1 regularization can be used to select the most predictive ones.
Data related to the company :Technical data : smoothing factors (moving averages )  , RSI ,MACD …,Fundamental data: Price to earning ratio , EBDIT..
Data related to the Sector :Sector growth ,key ratios and even EFT prices relative to the sector ...
Macro economic data :Exchange rate , inflation , unemployement rate etc
Financial news related to the stock
