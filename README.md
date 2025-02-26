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
We prepare initial Pipelines to use with our baseline model, and create a function `store_results()` that stores the models and results for our work so we can easily access the results and compare later.

After running initial MultinomialNB and and LogisticRegression models with modest results, we attempt hypertuning the parameters to see if we can produce better success metrics. We saw meager progress in this avenue, so we brought in a couple more models to experiment with, namely **LightGBM** and **RandomForestClassifier**.

Still struggling to produce sufficient progress in terms of success metrics, we attempt to enhance our models by integrating **Feature Engineering** such as:
- TF-IDF Vectorization
- Bigrams and Trigrams
- Word Embeddings

We create another transformer, `FeatureEngineer()`, which essentially carries out the previous preprocessing tasks in addition to the feature engineering techniques described.

Despite best efforts, we were unable to entirely surmount the obstacle of class imbalance we described earlier, with our models either performing well in terms of Precision but very poorly in terms of Recall and F1, or performing average across all three.

Across all the iterations of the various models we ran, we identified three final models to use on our test data:
1. **Baseline MultinomialNB**, which had the highest Precision score but scored poorly in Recall and F1
2. **Logistic Regression, Tuned and Enhanced**, which was slightly more balanced than the Baseline but still skewed in favor of Precision
3. **Random Forest Classifier, Tuned and Enhanced**, which though it did not excel at any single metric, was by far the most balanced among all three metrics

## Visualizations
### Evaluating Performance
![Heatmap](https://github.com/momopajamas/twitter-analysis-project/blob/535cc3a3015f647d6f3ee5490a8b261d52858eb8/images/model_heatmap.png?raw=true)
Let's look at it from another perspective.

![Barchart](https://github.com/momopajamas/twitter-analysis-project/blob/main/images/model_group_barchart.png?raw=true)
What this tells us is the following, in terms of evaluating the strengths and weaknesses of our models:
1. The final RFC model is the most balanced model, with BaselineMNB the most unbalanced.
2. All of the models struggle to varying degrees with capturing all Positive cases, with many Positives falling through the cracks, indicating that despite our efforts to account for the class imbalance our models still failed to fully absorb Positive cases in their training.
3. All three models perform best at correctly keeping from misclassifying Not Positive as Positive, the issue lies in accidentally classifying Positives as Not Positives.

### Feature Importances

While we have a better idea at how the models compare to each other, let's see what our findings can tell us about Feature Importances, or which words were calculated to have significance in determining whether a tweet was Positive or Not Positive.

![Top5Words](https://github.com/momopajamas/twitter-analysis-project/blob/main/images/top_5_features.png?raw=true)
What this can lead us to conclude is that, despite the limitations and constraints we faced, we were able to successfully get our models to learn the relative significance of certain terms in the context of social media and Twitter.

# Conclusion

## Limitations

## Recommendations

## Next Steps

# Appendix 
