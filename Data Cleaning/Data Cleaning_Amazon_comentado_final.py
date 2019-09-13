
import pandas as pd
import numpy as np
import re
import json
from textblob.classifiers import NaiveBayesClassifier
import nltk
from sklearn.model_selection import train_test_split
from autocorrect import spell
from datetime import datetime, timedelta
#from symspellpy.symspellpy import SymSpell, Verbosity 
from sklearn import metrics
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.pipeline import Pipeline


from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer,TfidfTransformer
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report 
from sklearn.linear_model import LogisticRegression, SGDClassifier

from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
import time
from sklearn.metrics import matthews_corrcoef


'''
   Method to execute an initial cleaning of the data colected.
   - extract the stars number given from a text ex: "A 4 out of 5 start" extracts the number 4
   - removes characters in the beggining of the text from the title and the review column
   - count the number of words in each review
   - save this modified dataframe in a new excel
'''
def initial_clean_data_set(file_name):

    print('start reading: %s' % (file_name))
    
    df = pd.read_excel(file_name) #Read Excel file as a DataFrame

    df['stars'] = df['stars'].str[2] # retirar apenas o nr de rating
    df['title'] = df['title'].str[2:-1] # retirar o b do titulo e as pelicas
    df['review'] = df['review'].str[2:-1] # retirar o b da review e as pelicas

    df.reset_index(drop=True) #reset do index sem uma nova coluna

    df['word_count'] = df['review'].apply(lambda x: len(x.split(' '))) # conta o nr de palavras de cada review
     
    df.to_excel("amazon_final_ok.xlsx") #Write DateFrame back as Excel file

    return 0

def select_review_with_x_words(file_name, number_words):

    '''
        Method to remove the data set with less than the words given in the input number_words
        example: remove the data set reviews with less then 50 words
        save this filtered dataframe in an excel file
    '''

    print('start reading: %s' % (file_name))
    
    df = pd.read_excel(file_name)                   # Read Excel file as a DataFrame

    initialSize = df.shape[0]                       # store the df size before the filtering

    df = df.loc[df['word_count'] >= number_words]   # filter the dataframe where the review word count is grater or equal to number given in the input

    afterSize = df.shape[0]                         # store the df size after the filtering

    print('number of lines with below %s words: %s' % (number_words, initialSize - afterSize))  # calculate de number of words removed

    df.reset_index(drop=True)                        #reset do index sem uma nova coluna

    df.to_excel("amazon_final_ok_below_%s.xlsx" % (number_words)) #Write DateFrame back as Excel file

def get_df_with_x_review_each_star(file_name, number_of_reviews = 50000):
    '''
        method to obtain a dataframe with a number of reviews where each star classification has the number of the review given.
    '''

    print('start reading: %s' % (file_name))
    
    df = pd.read_excel(file_name) #Read Excel file as a DataFrame

    number_star_1 = df.loc[df['stars'] == 1]

    # check if there are enough samples to create a sample. If there is less samples from the begining, then do not create a smaller sample and assign all its samples
    if number_star_1.shape[0] >= number_of_reviews:
        sampled_number_star_1 = number_star_1.sample(random_state=1, n = number_of_reviews)
    else:
        sampled_number_star_1 = number_star_1.copy()


    number_star_2 = df.loc[df['stars'] == 2]

    #same thing as the if above, but shorter in code
    # check if there are enough samples to create a sample. If there is less samples from the begining, then do not create a smaller sample and assign all its samples
    sampled_number_star_2 = number_star_2.sample(random_state=1, n = number_of_reviews) if number_star_2.shape[0] >= number_of_reviews else number_star_2.copy()
    
    number_star_3 = df.loc[df['stars'] == 3]
    sampled_number_star_3 = number_star_3.sample(random_state=1, n = number_of_reviews) if number_star_3.shape[0] >= number_of_reviews else number_star_3.copy()
    
    number_star_4 = df.loc[df['stars'] == 4]
    sampled_number_star_4 = number_star_4.sample(random_state=1, n = number_of_reviews) if number_star_4.shape[0] >= number_of_reviews else number_star_4.copy()
    
    number_star_5 = df.loc[df['stars'] == 5]
    sampled_number_star_5 = number_star_5.sample(random_state=1, n = number_of_reviews) if number_star_5.shape[0] >= number_of_reviews else number_star_5.copy()           

    frames = [sampled_number_star_1, sampled_number_star_2, sampled_number_star_3, sampled_number_star_4, sampled_number_star_5]

    result = pd.concat(frames)
    print("final contatenation result ",result.shape[0])

    result.to_excel('sampled_df_with_%s_each_star.xlsx' % (number_of_reviews))



def count_number_stars(file_name):
    '''
        method to count the number of each star classification.
        Example:  1 star: 3 review, 2 star: 27 reviews ... 5 star: 1599 reviews 
    '''
    start = time.time()

    print('start reading: %s' % (file_name))  
    
    df = pd.read_excel(file_name)                   #Read Excel file as a dataframe                         

    df['stars'] = pd.to_numeric(df['stars'])

    df.set_index('stars', inplace=True)                      # create an index with the column stars. Created this index so we can do a group by with the stars column 
    df['stars count'] = df.groupby(['stars'], sort=False)['review'].count() # group the reviews by star and count the number of occurrence 

    df_stars_unique = df.drop_duplicates(subset=['stars count'], keep='first') # keep only one line for each type of stars. this is useful to only have 1 entry for each star classification
    

    df_stars_unique.reset_index( inplace=True) # reset index so we can select the column stars

    # create a dictionary where the key is a star number, and the value the number of occurrence of that star
    # example  1(star one): 432 (number of times that star exists in the df) 
    dict_stars = {1:df_stars_unique[df_stars_unique['stars']==1]['stars count'].item(),
                2:df_stars_unique[df_stars_unique['stars']==2]['stars count'].item(),
                3:df_stars_unique[df_stars_unique['stars']==3]['stars count'].item(),
                4: 0 if df_stars_unique[df_stars_unique['stars']==4]['stars count'].empty else df_stars_unique[df_stars_unique['stars']==4]['stars count'].item(),
                5: 0 if df_stars_unique[df_stars_unique['stars']==5]['stars count'].empty else df_stars_unique[df_stars_unique['stars']==5]['stars count'].item() }
    print(dict_stars)
    print ("it took_count_number", time.time() - start, "seconds.")
    print(datetime.strftime(datetime.today() , '%d/%m/%Y-%Hh/%Mm'))

# def split_df_in_training_and_test(file_name):
#     '''
#     method that reads a dataframe from an excel file and create two files:
#         - a train df containing 70% of the received df
#         - a test df contianing 30% of the received df
#     '''
#     print('start reading: %s' % (file_name))
    
#     df = pd.read_excel(file_name) #Read Excel file as a DataFrame

#     msk = np.random.rand(len(df)) < 0.7 # create a mask that contains the values for the df_train in the positive values (msk) and the values in for the df_test in the negative values (~msk). NOTE the ~ is similarly as a negative in this case

#     train = df[msk]         # use the mask to obtain the respective df values for the training set
#     print('train df size: %s ' %(train.shape[0]))
#     train.to_excel("df_train.xlsx") #Write DateFrame back as Excel file

#     test = df[~msk]         # use the mask to obtain the respective df values for the testing set
#     print('test df size: %s ' %(test.shape[0]))
#     test.to_excel("df_test.xlsx") #Write DateFrame back as Excel file

#     ##algoritms used, already filter by train/test

def convert_to_numeric_column(file_name, column_name):
    start = time.time()
    print('start reading: %s' % (file_name))
    
    df = pd.read_excel(file_name) #Read Excel file as a DataFrame


    df['stars'] = pd.to_numeric(df['stars'])

def remove_special_and_stop_words(file_name, is_to_remove_stopwords=True):
    '''
    method to remove special characters and stop words.
    The stopwords remotion is optional and can be passed as input

    '''
    start = time.time()
    print('start reading: %s' % (file_name))
    
    df = pd.read_excel(file_name) #Read Excel file as a DataFrame


    REPLACE_NO_SPACE = re.compile("(\;)|(\:)|(\!)|(\n)|(\r)|(\?)|(\")|(\()|(\))|(\[)|(\])")  # regex used to find these special characters, to later be removed
    REPLACE_WITH_SPACE = re.compile("(\-)|(\/)")                                             # regex used to find these special characters, to later be swapped with a space

    df['review'] = df['review'].astype(str)             # change the review column type to be a string. This is to ensure all reviews are of a string type

    df = df.replace(r'\\n',' ', regex=True)             # replace all the "\n" characters with a space. This was required because for some reason the regex rule above wouldn't recognise
    df = df.replace(r'\\','', regex=True)               # find any "\" character and remove them. This was required because for some reason the regex rule above wouldn't recognise

    df['review'] = df['review'].apply(lambda x: REPLACE_NO_SPACE.sub("", x.lower()) ) # apply the regex defined above to remove all special characters in the column review. Replaced by blank and all of the review in lowercase. 

    df['review'] = df['review'].apply(lambda x: REPLACE_WITH_SPACE.sub(" ", x) ) # apply the regex defined above to replace all special characters with a space in the column review # already in lowercase.
                                                                                 # replace all hifens by space.
    if (is_to_remove_stopwords):                    
        df['review'] = df['review'].apply(lambda x: remove_stopwords(x))        # apply the remotion of stop words for all rows in the column review
    
    df.to_excel(file_name) #Write DateFrame back as Excel file
    print ("it took_remove special words", time.time() - start, "seconds.")
    print(datetime.strftime(datetime.today() , '%d/%m/%Y-%Hh/%Mm'))


def get_classifier_with_cross_validation(file_name, subset=0, use_binomial=False):
    '''
    method that receives a file , loads its dataframe and run 3 classifiers:
        - naive bayes
        - SVM
        - Logistic Regression

    in the end prints the classification report for each classifier
    this method receives an optional subset field, which is used to select the number of rows to use in the df

    '''

    print('start reading: %s' % (file_name))

    
    df = pd.read_excel(file_name) #Read Excel file as a DataFrame
 

    if subset :                 # in case of a subset value is passed
        df = df.iloc[0:subset]  # select the number of rows to use, passed in the input
    

    X = df['review']            # set the variable X with the list of the review column values
    if use_binomial:
        Y = df['starEquivalent']             # use the respective true /false classification
    else:
        Y = df['stars']             # set the variable Y with the list of the stars column values

    # this splits the df into 2 groups, the train values and the test values
    start = time.time()

    # create a pipeline to execute the methods, CountVectorizer, TfidfTransformer, MultinomialNB sequentialy for each value received
    nb = Pipeline([('vect', CountVectorizer()),
                   ('tfidf', TfidfTransformer()),
                   ('clf', MultinomialNB()),
                  ])

    # run the pipeline defined above with the train values. This teaches the naive bayes algorithm
    scores = cross_val_score(nb, X, Y, cv=10, scoring='accuracy')
    print(scores)
    print(scores.mean())
    print ("it took NAIVE BAYES", time.time() - start, "seconds.")

    start = time.time()
    logreg = Pipeline([('vect', CountVectorizer()),
                    ('tfidf', TfidfTransformer()),
                    ('clf', LogisticRegression(n_jobs=1, C=1e5)),
                   ])

    scores = cross_val_score(logreg, X, Y, cv=10, scoring='accuracy')
    print(scores)
    print(scores.mean())
    print ("it took LogisticRegression", time.time() - start, "seconds.")     

    start = time.time()
    sgd = Pipeline([('vect', CountVectorizer()),
                ('tfidf', TfidfTransformer()),
                ('clf', SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, random_state=42, max_iter=5, tol=None)),
                ])

    scores = cross_val_score(sgd, X, Y, cv=10, scoring='accuracy')
    print(scores)
    print(scores.mean())
    print ("it took SVM", time.time() - start, "seconds.")                

def get_classifier(file_name, subset=0, use_binomial=False):
    '''
    method that receives a file , loads its dataframe and run 3 classifiers:
        - naive bayes
        - SVM
        - Logistic Regression

    in the end prints the classification report for each classifier
    this method receives an optional subset field, which is used to select the number of rows to use in the df

    '''

    print('start reading: %s' % (file_name))

    
    df = pd.read_excel(file_name) #Read Excel file as a DataFrame
 

    if subset :                 # in case of a subset value is passed
        df = df.iloc[0:subset]  # select the number of rows to use, passed in the input
    

    X = df['review']            # set the variable X with the list of the review column values
    if use_binomial:
        Y = df['starEquivalent']             # use the respective true /false classification
    else:
        Y = df['stars']             # set the variable Y with the list of the stars column values

    # this splits the df into 2 groups, the train values and the test values
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3,random_state=42)
    start = time.time()

    # create a pipeline to execute the methods, CountVectorizer, TfidfTransformer, MultinomialNB sequentialy for each value received
    nb = Pipeline([('vect', CountVectorizer()),
                   ('tfidf', TfidfTransformer()),
                   ('clf', MultinomialNB()),
                  ])

    # run the pipeline defined above with the train values. This teaches the naive bayes algorithm
    nb.fit(X_train, y_train)

    # with a taught naive bayes, it tries to predict the values from the test list
    y_pred = nb.predict(X_test)

    # create a list of tags for result display purpose 
    if use_binomial:
        my_tags = ['Negative', 'Positive']  

    else:
        my_tags = ['1','2','3','4','5']

    print('results for the Naive Bayes')
    print('accuracy %s' % accuracy_score(y_pred, y_test))
    print(classification_report(y_test, y_pred,target_names=my_tags)) # show the precision    recall  f1-score   support values
    print(confusion_matrix(y_test, y_pred))
    print("Erro quadrado NB", np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
    print('Matthews correlation coefficient %s' % (matthews_corrcoef(y_test, y_pred)))
    print ("it took NAIVE BAYES", time.time() - start, "seconds.")
    print(datetime.strftime(datetime.today() , '%d/%m/%Y-%Hh/%Mm'))
    

    start = time.time()
    # create a pipeline to execute the methods, CountVectorizer, TfidfTransformer, LogisticRegression sequentialy for each value received
    logreg = Pipeline([('vect', CountVectorizer()),
                    ('tfidf', TfidfTransformer()),
                    ('clf', LogisticRegression(n_jobs=1, C=1e5)),
                   ])
    # run the pipeline defined above with the train values. This teaches the Logistic regression algorithm
    logreg.fit(X_train, y_train)

    # with a taught logistic regression, it tries to predict the values from the test list
    y_pred = logreg.predict(X_test)

    print('results from the Logistic Regression')
    print('accuracy %s' % accuracy_score(y_pred, y_test))
    print(classification_report(y_test, y_pred,target_names=my_tags)) # show the precision    recall  f1-score   support values
    print(confusion_matrix(y_test, y_pred))
    print("Erro quadrado Logistic", np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
    print('Matthews correlation coefficient %s' % (matthews_corrcoef(y_test, y_pred)))
    print ("it took Log", time.time() - start, "seconds.")
    print(datetime.strftime(datetime.today() , '%d/%m/%Y-%Hh/%Mm'))
   

    start = time.time()
    # create a pipeline to execute the methods, CountVectorizer, TfidfTransformer, SGDClassifier sequentialy for each value received
    sgd = Pipeline([('vect', CountVectorizer()),
                    ('tfidf', TfidfTransformer()),
                    ('clf', SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, random_state=42, max_iter=5, tol=None)),
                   ])

    # run the pipeline defined above with the train values. This teaches the SVM algorithm
    sgd.fit(X_train, y_train)

    # with a taught SVM, it tries to predict the values from the test list
    y_pred = sgd.predict(X_test)

    print('results from the SVM')
    print('accuracy %s' % accuracy_score(y_pred, y_test))
    print(classification_report(y_test, y_pred,target_names=my_tags)) # show the precision    recall  f1-score   support values
    print(confusion_matrix(y_test, y_pred))
    print("Erro quadrado SVM", np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
    print('Matthews correlation coefficient %s' % (matthews_corrcoef(y_test, y_pred)))
    print ("it took SVM", time.time() - start, "seconds."  )
    print(datetime.strftime(datetime.today() , '%d/%m/%Y-%Hh/%Mm'))


def set_df_to_binomial(file_name):
    '''
        method to convert the stars classification into the respective negative and positive binomial classification

    '''
    print('start reading: %s' % (file_name))

    
    df = pd.read_excel(file_name) #Read Excel file as a DataFrame
 
    starEquivalent = {
        1: 1,
        2: 1,
        3: 2,
        4: 2,
        5: 2
     }

    df['starEquivalent'] = df['stars'].apply(lambda x : starEquivalent[x]) # usa o dicionario das estrelas para atribuir as novas classes as estrelas

    df.to_excel('binomial_' + file_name) #Write DateFrame back as Excel file


def remove_stopwords(phrase):
    '''
    method to remove stop words from a phrase
    '''
    stopwords = nltk.corpus.stopwords.words('english')  # load the stop word list for the english dictionary
    word_tokens = word_tokenize(phrase)                 # tokenize the phrase into words list
    filtered_sentence = [w for w in word_tokens if not w in stopwords] # filter the words list and remove any word that is contained in the stopwords list

    return " ".join(filtered_sentence)                  # join the words list separated by spaces and return it

def get_score_equivalent(number):
    '''
        method to return a classification of neutral, negative and positive given the number received
    '''

    if (number == 0) :    
        return 'Neutral'
    elif (number < 0) :
        return 'Negative'
    else:  
        return 'Positive'


def apply_stemming_to_df(file_name, subset=0):
    ''' 
    method that applies the stemming process to the review column in the dataframe

    '''    
    print('start reading: %s' % (file_name))
    df = pd.read_excel(file_name) #Read Excel file as a DataFrame  
    
    if subset :                 # in case of a subset value is passed
        df = df.iloc[0:subset]  # select the number of rows to use, passed in the input

    start=datetime.now()

    # with autocorrect spell checker it takes 0:06:55.094294 for 1000 samples
    df['review'] = df['review'].apply(set_stemming_to_phrase)     # for each row in the review column apply the stemming process
    df.to_excel("stemming_" + file_name)                          # store in an excel file

    print (datetime.now()-start)        # print the time that was used to process the stemming in all review column


def set_stemming_to_phrase(phrase):
    '''
        method that receives a phrase and apply the stem process
    '''
    ps = PorterStemmer()              #initialize the stemmer library
    words = word_tokenize(phrase)     # tokenize the phrase into a word list

    new_phrase = []                    # initialize a new list that will contain the stemmed words. 
    for w in words:
        new_phrase.append(ps.stem(w))  # apply the stem process for the word and store it in the list
        
    return ' '.join(new_phrase) # join the words list separated by spaces and return it

def get_emotional_lexicon(file_name, emotion_dic, negative_connotation = False):

    ''' 
        method that uses the emotional lexicon to classify the received df into 3 classes: positive, negative and neutral
        the reviews with 1 and 2 stars are classified as negative
        the reviews with 3 stars are classified as neutral
        the reviews with 4 and 5 stars are classified as positive
        in the end it calculates the accuracy
     ''' 
    print('start reading: %s with negative_connotation = %s' % (file_name, negative_connotation))

    df = pd.read_excel(file_name) #Read Excel file as a DataFrame


    emolex_df = pd.read_csv(emotion_dic,  usecols=["English", "Positive", "Negative"], sep=';')  # load the emotion lexicon into a df

    # create a dictionary for each lexicon word which has a value 1 in case of a positive word and -1 in case of a negative word
    lexicon_dic = {row['English']:(row['Positive'] if row['Positive'] else 0-row['Negative']  )for index, row in emolex_df.iterrows()}

    # caso seja para aplicar a conotacao negativa
    if(negative_connotation):
        df['review'] = df['review'].apply(set_negative_connotation ) # for each review , apply the negative connotation || ############################33change column


    # aplica o lexicon a review e guarda o resultado da soma dos pesos na coluna score
    df['score'] = df['review'].apply(lambda x: sum([lexicon_dic[word] for word in x.split() if word in lexicon_dic]) )

    # dicionario com classificacao para cada estrela
    starEquivalent = {
        1: 'Negative',
        2: 'Negative',
        3: 'Neutral',
        4: 'Positive',
        5: 'Positive'
    }

    df['scoreEquivalent'] = df['score'].apply(get_score_equivalent) # por cada linha le o score e atribui caso seja negativo, positivo ou neutro numa nova coluna
    df['starEquivalent'] = df['stars'].apply(lambda x : starEquivalent[x]) # usa o dicionario das estrelas para atribuir as novas classes as estrelas

    df['prediction'] = df['scoreEquivalent'] == df['starEquivalent'] # por cada linha, verifica se a predicao foi acertada ou nao e guarda numa nova coluna

    correct_prediction = df.loc[df['prediction'] == True].shape[0]  # conta as predicoes correctas, para calcular no fim a accuracy

     # if apenas para efeitos de prints na consola
    if(negative_connotation):
        print('results from emotion lexicon with negative treatment')
    else:
        print('results from emotion lexicon without negative treatment')

    # accuracy print
    print('total numbers of entries: %s\nnumber of correct prediction: %s\naccuracy: %s' % (df.shape[0], correct_prediction, correct_prediction/df.shape[0]))

    # if apenas para mudar o nome do ficheiro caso a negacao tenha sido aplicada
    if(negative_connotation):
        df.to_excel("with_negative_" + file_name)
    else:    
        df.to_excel("without_negative_" + file_name)
 

def use_autocorrect_on_dataframe(file_name, subset=0):
    '''
    method to apply the spell checker for each review in the df review column

    '''
    
    print('start reading: %s' % (file_name))
    
    df = pd.read_excel(file_name) #Read Excel file as a DataFrame
  
    if subset :                 # in case of a subset value is passed
        df = df.iloc[0:subset]  # select the number of rows to use, passed in the input

    start=datetime.now()

    # with autocorrect spell checker it takes 0:06:55.094294 for 1000 samples
    df['review'] = df['review'].apply(set_autocorrect_spell) # for each review, apply the spell checker
    df.to_excel("spellchecker_" + file_name)                # write to file the new df spell checked

    print (datetime.now()-start)

def set_autocorrect_spell(phrase):
    '''
        method that receives a phrase and apply the spell checker
    '''

    # remove all punctuation, adding a space for each punctuation found
    with_space_between_punctuation = re.sub(r"(\;)|(\,)|(\?)|(\!)|(\:)|(\.)", r' ', phrase)

    new_phrase = []     # initialize a new list that will contain the spell checked words. 
    words_changed = 0
    for word in with_space_between_punctuation.split():     # the split is the same as tokenize
        new_phrase.append(spell(word))                      # apply the spell checker and then store in the list


    return ' '.join(new_phrase) # join the words list separated by spaces and return it

def set_negative_connotation(phrase):
    '''
        method that receives a phrase and apply a negative connotation.
        In a phrase, when a negation word is found, it will add a prefix "NOT_" to each word after the negation word, 
        until a punctuation delimit is found
    '''

    negation_words = ["no", "not","rather", "couldn't", "wasn't", "didn't", "wouldn't", "shouldn't",
                      "weren't","don't","doesnt'","haven't","hasn't","won't", "wont", "hadn't",
                      "never", "none","nobody","nothing","neither","nor","nowhere", "isn't","can't",
                      "cannot","mustn't", "mightn't", "shan't","without","needn't"]

    diminuisher_words= ["hardly", "less", "little", "rarely", "scarcely", "seldom"]         
    

    punctuation_delimits = [',', '.', '?', '!',':',';']

    # add the negation words and diminuisher words in the same list.
    negative_words_list = negation_words + diminuisher_words

    # swap all special characters with a dot with spaces after and before.
    # this is required to separate punctuations from words. ex: "car." becomes "car . " A tokenizer was not used because we need the punctiation marks
    with_space_between_punctuation = re.sub(r"(\;)|(\,)|(\?)|(\!)|(\:)|(\.)", r' . ', phrase)

    new_phrase =        [] # initialize a new list that will contain the negated words if applied. 
    start_negate_words = False          # variable to control when the negations is to be applied
    for word in with_space_between_punctuation.split():

        if (word in punctuation_delimits):          # if we reach a punctuation, stops negating any word
            start_negate_words = False

        elif(start_negate_words):                   # if the negation was activated, then negate the word
            new_phrase.append("NOT_" + word)
        elif(word in negative_words_list):          # if the words is contained in the negation words, then start negate the next words
            start_negate_words = True               # activate the flag start_negate_words to start negate the next words 
            new_phrase.append(word)                 
        else:
            new_phrase.append(word) 

    return ' '.join(new_phrase) # join the words list separated by spaces and return it

  
def remove_emptys(file_name):
    '''
    method to remove all the NA rows in the column review
    this is required because the classifiers won't work with empty rows

    '''
    print('start reading: %s' % (file_name))
    
    df = pd.read_excel(file_name) #Read Excel file as a DataFrame

    print('before df size %s'% (df.shape[0]))
    df.dropna(subset=['review'], inplace=True)
    print('after df size %s'% (df.shape[0]))

    df.to_excel(file_name) #Write DateFrame back as Excel file

###########################

def main():
    #initial_clean_data_set('amazon_final.xlsx')
    #initial_clean_data_set('mergedReviewsFirstAndSecondAndThirdBatch.xlsx')

    #select_review_with_x_words( 'final_review_v2.xlsx', 50)

    ###########################
    #count_number_stars("final_review_v2.xlsx")

    #get_df_with_x_review_each_star('final_review_748k.xlsx')
    #count_number_stars("sampled_df_with_50000_each_star.xlsx")
    #remove_special_and_stop_words("sampled_df_with_50000_each_star_without_stopwords.xlsx", is_to_remove_stopwords=True)

    # get classification for the 500k reviews
    #remove_emptys("sampled_df_with_50000_each_star_without_stopwords.xlsx")
    #count_number_stars("sampled_df_with_50000_each_star_without_stopwords.xlsx")
    #get_classifier("sampled_df_with_50000_each_star_without_stopwords.xlsx", subset=0)
    #get_classifier_with_cross_validation("sampled_df_with_50000_each_star_without_stopwords.xlsx", subset=0)
    # get classification for binomial
    set_df_to_binomial("sampled_df_with_50000_each_star_without_stopwords.xlsx")
    get_classifier("binomial_sampled_df_with_50000_each_star_without_stopwords.xlsx", use_binomial= True)
    '''

    # get classification for the words stemmed
    apply_stemming_to_df("amazon_final_ok_repaired.xlsx", subset=0)    
    get_classifier("stemming_amazon_final_ok_repaired.xlsx", subset=0)

    # get classification for auto correct
    use_autocorrect_on_dataframe("amazon_final_ok_repaired.xlsx", subset=0)
    get_classifier("spellchecker_amazon_final_ok_repaired.xlsx")

    # get classification for binomial
    set_df_to_binomial("amazon_final_ok_repaired.xlsx")
    get_classifier("binomial_amazon_final_ok_repaired.xlsx", use_binomial= True)
    '''

    #remove_special_and_stop_words('amazon_final_ok_repaired_deia.xlsx', is_to_remove_stopwords=False)
    #get_emotional_lexicon('amazon_final_ok_repaired.xlsx','NCR-lexicon.csv')
    #get_emotional_lexicon('df_train.xlsx','NCR-lexicon.csv',negative_connotation= True)
if __name__== "__main__":
    main()