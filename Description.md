# Description:
	This project was a part of a hackathon conducted by Amazon. 
	We were given the product's browse node ID along with their textual description in the form of 3 columns, i.e. Title, Description, and bullet points. Our task was to find the Browse node ID of a product given its description. We were a team of 4.
	It was a supervised learning classification problem. 
	The dataset contained ~3M rows and 9k categories. 
	The data was huge and the categories were highly imbalanced. So, we opt for stratified sampling based on browse node id to get a representative of data.

	All of the feature columns contained some information about the product and some of the information was mentioned in the description as well as in the bullet points.
	So, we thought of combining all the feature columns into one column by just concatenating all the text. Resulting in one feature column and one target.
	
	Now the next step was data cleaning. We did the typical cleaning steps used in any text-based problem. 
	i.e. Tokenization then removing punctuations & stopwords, then stemming.

	The next step was to convert this vector of words to numeric vectors to feed it into our model. 
	We chose CountVectorizer and Tf-IDF. 

	There are models like Naive Bayes, logistic regression, and SVM which work well with text data. 
	We tried all the algorithms and found that SVM performed better resulting in an accuracy of 64.14% in the final submission.


# Why Stemming?
	Lemmatization is closely related to stemming. 
	The difference is that a stemmer operates on a single word without knowledge of the context, and therefore cannot discriminate between words that have different meanings depending on part of speech. 
	However, stemmers are typically easier to implement and run faster, and the reduced accuracy may not matter for some applications.

# Why Stratified sampling?
	It is also called proportional random sampling.
	It allows researchers to obtain a sample population that best represents the entire population being studied.
	It involves dividing the entire population into homogeneous groups called strata.
	It differs from simple random sampling, which involves the random selection of data from an entire population, so each possible sample is equally likely to occur.


# What are CountVectorizer and Tfidf doing? Why not any other transformation?
	CountVectorizer fits on the training data and gives a number to each unique word and then each word will have a vector that has a non-zero value at index(which is equal to the number it has in vocabulary formed before). 
	This value is equal to the frequency of its occurrence in the document.
	TF-IDF has a value equal to frequency*tfidf_term, this TF-IDF term normalizes the value if it's occurring so many times in the document. 
	If a word is occurring in the whole document so many times then this word will no longer have importance wrt to a particular instance, 
	so better to remove it or decrease its importance. 

	Wij = tfij * log(N/dfi)
	Wij = tf idf weight for token i in document j
	Tfij = number of occurrence of token i in document j
	Dfi = No. of document containing token i
	N= Total no. of documents

	Word2Vec or Glove embedding can be used to transform text data which uses a neural network to give us word embedding.
	Word2Vec uses the proximity of other words to give context to a particular word. 
	Whereas glove embedding with the help of dot products keeps similar words in similar space. 

# How does logistic regression work with text data?


# Multinomial NB vs SVM?
	1. NB assumes all features as independent of each other and it makes its prediction on the basis of this assumption. 
	But in our dataset, this assumption might fail because each text token of a particular example has a context related to the description, 
	which will help the model predict the browse node ID.
	2. That's why SVM should work better than NB as it doesn’t assume the independence of features and try to place all the similar vector representations in the same plane.
	3. SVM works fine with binary classification, but here we had a multi-class classification task.
	SVM here either can choose one vs one algorithm (in scikit- learn it can opt by SVC(kernel=linear) ) or can opt for one vs rest( LinearSVC() )
	4. In one vs one algorithm, we’ll choose 2 classes and one classifier will only make predictions for these classes and will ignore the rest of the classes, in this way, we’ll end up having n*(n-1)/2 classifiers for n classes
	5. In the "one vs rest" algorithm, the model will classify for one class and assume 0 for the rest of the classes, in this way we’ll end up having classifiers for n classes.
	6. In our case we had a 9k categories, choosing one vs one classifier will be very time consuming and will crash our model in between, so we chose one vs all classifier

# Why only ML algorithms, not any deep learning algorithm?

