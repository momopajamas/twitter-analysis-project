# Business Understanding
Microsoft wants to be able to factor in sentiments about our products and services expressed on social media platforms like Twitter to supplement the feedback we received through official review channels like Amazon, Yelp, etc. 

Specifically, Microsoft wants to look at Positive feedback produced organically on social media about our products and services to know what we should develop further, expand upon, and keep from undermining through budget cuts and the like. We want to be able to filter tweets carrying Positive sentiments so that we are able to assess what's working well.

The way we will go about this is by creating a **binary classification model** that is trained and tested on a dataset containing tweets where users expressed various emotions or reactions to other products, as the essential issue is how well our model is able to parse through and differentiate between Positive tweets on the one hand, and Negative or Neutral tweets on the other hand.

In evaluating out work, we need to ascertain how effective we were at correctly identifying Positive tweets, keeping an eye out for **False Positives** in particular as non-Positive sentiments seeping into our Positive class would muddy our analysis and cause problems down the road.

False Negatives are important to keep an eye out for as well, however not as important as False Positives. While missing out on Positive sentiments that fell through the cracks is not ideal, it is more detrimental for our purposes for our Positive category to carry Negative or Neutral sentiments.

# Data Understanding
As this problem is about analyzing and categorizing sentiments expressed through text, we will need to build a model capable of Natural Language Processing, or NLP for short. This model needs to be adept at processing and parsing through text, and categorizing the text as 'Positive', 'Not Positive' (ie: 'Negative' or 'Neutral').

We will engage in NLP to build a binary classifier that is capable of differentiating between 'Positive' and 'Negative' or 'Neutral' sentiments.

## Dataset
For these purposes, we will use a [dataset from data.world](https://data.world/crowdflower/brands-and-product-emotions) which contains more than 9,000 tweets expressing Positive, Negative, or Neutral sentiments towards Apple or Google products.

This dataset contains information across three columns:
1. `tweets_text`, containing the text of the collected tweets. This will serve as our **Features** or X variable in modeling, and will use this column to generate **TF-IDF scores**, which assigns numeric values for key terms by weighing their frequency within a certain text against their frequency across different texts. This will help our model in gaining signals from significant words and reduce noise from frequent, insignificant words. These will be our features, at least in the initial baseline model.
2. `emotion_in_tweet_is_directed_at`, which indicates which brand or product the tweet is addressing. We will not need this column in our modeling and will drop it.
3. `is_there_an_emotion_directed_at_a_brand_or_product`, which categorizes the tweets according to Positive, Negative, Neutral, or I Can't Tell. We will drop the latter entries as they don't serve us. We will also combine Neutral and Negative under a new category, Not Positive, since our purpose is to find Positive tweets and not differentiate between Negative, Neutral, and Positive.

## Class Imbalance
There is also strong **class imbalance** in our data, with only 33% of the data representing Positive sentiments. We will deploy a number of strategies to minimize the impact of this class imbalance, but with a dataset of this size it will be difficult to neutralize entirely the negative impact of this class imbalance on our models' abilities to correctly differentiate between Positive and Not Positive tweets.

## Success Metrics
As described above, we need to pay extra mind to our model's ability to correctly identify Positive tweets, placing a higher importance on the rates of False Positives. However, we should also keep an eye on False Negatives to minimize the number of Positive tweets falling through the cracks.

For this reason, we will rely on the following metrics:

1. **Precision Score**, which evaluates how accurate we were in actually identifying Positive tweets (telling us the rates of False Positives). The higher the precision score, the lower the rate of False Positives. This will be our primary metric.
2. **F1 Score**, which weighs the rate of False Positives and False Negatives, since we also want to minimize the rate at which we misclassify Positive tweets as Not Positive. This will be a seconadry metric for us.
3. **Recall Score**, which is essentially the inverse of Precision in that it measures the rate of False Negatives and True Negatives. It will tell us how many True Positives are falling through the cracks and can supplement Precision.

## Model Selection
We will start with a simple baseline model as an initial performance check before moving on to more complex models. Since we have some class imbalance, we will deploy **Multinomial Naive Bayes (Multinomial NB)**, which can help compensate for this imbalance using a weighted approach. This model is also better at producing the metrics we outlined above.

After establishing a baseline, we will then move on to testing out more complex models.

## Data Preparation
In preparing the data, we carry out the following:
1. Dropping `emotion_in_tweet_is_directed_at` column
2. Dropping null values
3. Cleaning the Target values by dropping **I can't tell**
4. Collapsing **Neutral** and **Negative** values under Target column together as **Note Positive** and making these values binary (0=Not Positive, 1=Positive)
5. Splitting the data into training, validation, and testing sets

We create a custom transformer, `TextPreprocessor()`, which preprocessed our text data by lowercasing it, removing special characters, tokenizing the text, removing standard stop words, and lemmatizing the text.

We compile a list of **custom stop words** that includes both standard English stop words in addition to terms we identify as having low significance in the context of social media, such as 'rt' and 'u'.

# Modeling

# Conclusion

## Limitations

## Recommendations

## Next Steps

# Appendix 
