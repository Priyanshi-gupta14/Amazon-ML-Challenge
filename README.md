# Amazon ML Challenge:
## Introduction
    Our aim is to predict the browser node ID for Amazon products, so we used the NLP approach
    to solve the problem.
## Libraries and Modules
    Major libraries imported were Pandas for data manipulation, NumPy for mathematical
    manipulation, Scikit-learn for importing transformers and model structures, nltk, and re for
    cleaning and modifying text data.
## Importing Data
    Data was pretty huge with around ~3M rows in train and ~0.1 M in test data. The import was
    done using pandas and then the train data was distributed into batches
## Functions
    We used three custom functions in our notebook:
        1. create_product(DataFrame):
            After analyzing the data, we concluded that details in Title, Description and Bullet
            Points are clashing so, we concluded merging the Title, Description, and Bullet points
            into one single article which can be tokenized and used for training, so this function
            creates a new column “Products” using the existing 3 columns
            Also, we replaced the Null values in Title, Description, and Bullet Points with the empty
            string(“”)
        2. clean_data(data)
            This is the most important helper function in the notebook, it is used for cleaning the
            text data as it is important to send uniform data to our model for better training. This
            function lowercase all the text removes unnecessary punctuations and whitespaces,
            stems the words to their basic form, and removes stop words.
        3. create(index, model):
            This is a testing function that takes an index as an input. It runs the model over the test
            data and predict browse_node_ids and creates a CSV file of labels and predictions
## Stratified Data sampling
    We had a HUGE!! Dataset (~ 3M rows). So, instead of training the whole Dataset, we used
    Stratified sampling of the given Data. It gave us a sample Dataset that best represents the
    entire Dataset under study. For this, we used Scikit-Learn’s StratifiedShuffleSplit class.
## Cleaning
    From stratified sampling, we got the required representative training data, so before applying
    any Machine Learning algorithm on this data we need to first clean it. For cleaning the main
    text data containing column(‘Product’) we used the clean_data function on training as well as
    testing data.
## Models used
    To make the classifier, we can create a pipeline of all these estimators and transformers we
    want to use. To apply any machine learning model to the text data we have to first convert it
    into numerical representation, for that we used Count Vectorizer and TfidfTransformer.
    Now, the numeric vectors are given as input machine algorithms. We applied MultinomialNB,
    SGD, and LinearSVC, to the chunk of data and checked their accuracy.
    We found that LinearSVC was having the highest accuracy among all three estimators.
## Training and Validation
    We’ll fit our model over the training data’s Products_clean column and training data
    Browse_node_ids column.
## Testing and Final Result
    On testing data, our model accuracy came out to be 64.14%, which came to be increased by the
    increasing complexity of our transformers and estimators. Trying the model with a different
    set of hyperparameters and then comparing best among them. Using state-of-art
    Transformers models, deep learning models, but all took so much time and memory to train.
    
# Resources:
## How did you deal with the Imbalanced dataset?
 
## How did you clean the text data?
[Text Cleaning](https://monkeylearn.com/blog/text-cleaning/)
 
## Why did you use stemming instead of lemmatization? 
[Reference](https://monkeylearn.com/blog/text-cleaning/#:~:text=Stemming%2C%20the%20simpler,past%2C%20and%20indefinite.)
#### Stemming:
    It groups words by their root stem. This allows us to recognize that ‘jumping’ ‘jumps’ and ‘jumped’ are all rooted in the same 
    verb (jump) and thus are referring to similar problems.
#### Lemmatization:
    - It groups words based on root definition, and allows us to differentiate between present, past, and indefinite.
    - In our case, we were given the product descriptions. Hence the present, past, or future doesn't have any significance.
 
## Converting Tokenized words into numerical vectors:
1. [Tf-IDF](https://towardsdatascience.com/tf-idf-for-document-ranking-from-scratch-in-python-on-real-world-dataset-796d339a4089)
2. [Tf-IDF](https://machinelearningmastery.com/prepare-text-data-machine-learning-scikit-learn/)
## Different Word embedding techniques:
[ord embedding techniques](https://medium.com/analytics-vidhya/text-classification-using-word-embeddings-and-deep-learning-in-python-classifying-tweets-from-6fe644fcfc81)

## Support Vector Machine:
1. [Linear Kernel: Why is it recommended for text classification ?](https://www.svm-tutorial.com/2014/10/svm-linear-kernel-good-text-classification/)
2. [Naïve Bayes vs. SVM For Text Classification](https://medium.com/analytics-vidhya/na%C3%AFve-bayes-vs-svm-for-text-classification-c63478229c33)
3. [Which one is better: LinearSVC or SVC?](https://intellipaat.com/community/19783/which-one-is-better-linearsvc-or-svc)


# Description:
    This project was a part of a hackathon conducted by Amazon. We were given the product's browse node ID along with their textual description in the form of 3 columns, i.e. Title, Description, and bullet points. Our task was to find the Browse node ID of a product given its description. We were a team of 4.
    It was a supervised learning classification problem. The dataset contained ~3M rows and 9k categories. The data was huge and the categories were highly imbalanced. So, we opt for stratified sampling based on browse node id to get a representative of data.
    
    All of the feature columns contained some information about the product and some of the information was mentioned in the description as well as in the bullet points.  So, we thought of combining all the feature columns into one column by just concatenating all the text. Resulting in one feature column and one target.

    Now the next step was data cleaning. We did the typical cleaning steps used in any text-based problem. i.e. Tokenization then removing punctuations & stopwords, then stemming.

    The next step was to convert this vector of words to numeric vectors to feed it into our model. We chose CountVectorizer and Tf-IDF. 

    There are models like Naive Bayes, logistic regression, and SVM which work well with text data. We tried all the algorithms and found that SVM performed better resulting in an accuracy of 64.14% in the final submission.


# Why Stemming?
    Lemmatization is closely related to stemming. The difference is that a stemmer operates on a single word without knowledge of the context, and therefore cannot discriminate between words that have different meanings depending on part of speech. However, stemmers are typically easier to implement and run faster, and the reduced accuracy may not matter for some applications.

# Why Stratified sampling?
    It is also called proportional random sampling.
    It allows researchers to obtain a sample population that best represents the entire population being studied.
    It involves dividing the entire population into homogeneous groups called strata.
    It differs from simple random sampling, which involves the random selection of data from an entire population, so each possible sample is equally likely to occur. Stratified sampling is used to select a sample that is representative of different groups. If the group is of different sizes, the number of items selected from each group will be proportional to the number of items in that group.


# What are CountVectorizer and Tfidf doing? Why not any other transformation?
    CountVectorizer fits on the training data and gives a number to each unique word and then each word will have a vector that has a non-zero value at index(which is equal to the number it has in vocabulary formed before). This value is equal to the frequency of its occurrence in the document.
    TF-IDF has a value equal to frequency*tfidf_term, this TF-IDF term normalizes the value if it's occurring so many times in the document. If a word is occurring in the whole document so many times then this word will no longer have importance wrt to a particular instance, so better to remove it or decrease its importance. 


	Wij = tfij * log(N/dfi)
    Wij = tf idf weight for token i in document j
    Tfij = number of occurrence of token i in document j
    Dfi = No. of document containing token i
    N= Total no. of documents

    Word2Vec or Glove embedding can be used to transform text data which uses a neural network to give us word embedding. Word2Vec uses the proximity of other words to give context to a particular word. Whereas glove embedding with the help of dot products keeps similar words in similar space. 

# How does logistic regression work with text data?


# Multinomial NB vs SVM?
    NB assumes all features as independent of each other and it makes its prediction on the basis of this assumption. But in our dataset, this assumption might fail because each text token of a particular example has a context related to the description, which will help the model predict the browse node ID.
    That's why SVM should work better than NB as it doesn’t assume the independence of features and try to place all the similar vector representations in the same plane.
    SVM works fine with binary classification, but here we had a multi-class classification task, SVM here either can choose one vs one algorithm (in scikit- learn it can opt by SVC(kernel=linear) ) or can opt for one vs rest( LinearSVC() )
    In one vs one algorithm, we’ll choose 2 classes and one classifier will only make predictions for these classes and will ignore the rest of the classes, in this way, we’ll end up having n*(n-1)/2 classifiers for n classes
    In the "one vs rest" algorithm, the model will classify for one class and assume 0 for the rest of the classes, in this way we’ll end up having classifiers for n classes.
    In our case we had a 9k categories, choosing one vs one classifier will be very time consuming and will crash our model in between, so we chose one vs all classifier

# Why only ML algorithms, not any deep learning algorithm?

