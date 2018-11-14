import json
import pandas as pd
import nltk
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.corpus import wordnet
import sklearn
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import train_test_split

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('omw')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

class Preprocessor:
    # Constants
    minimalFreq = 0
    maximalFreq = 1000

    def preprocessDataset(self, data):
        return data.apply(self.__processText)
        
    def preprocessData(self, text):
        return text.apply(self.__processText)

    def __processText(self, text):
        # Lowercase
        text = text.lower()

        # Tokenize
        token = nltk.tokenize.word_tokenize(text)

        # Lemmatizer        
        tagged_token = pos_tag(token)
        lemma = WordNetLemmatizer()
        lemmatized_token = []
        for t in tagged_token:
            word, tag = t
            l = lemma.lemmatize(word, self.__get_wordnet_pos(tag))
            lemmatized_token.append(l)
        token = lemmatized_token  

        # Remove Stopwords
        stops = stopwords.words('english')
        token = [t for t in token if t not in stops]

        text = ' '.join(token)
        return text
      
    def __get_wordnet_pos(self, tag):
        if tag.startswith('J'):
            return wordnet.ADJ
        elif tag.startswith('V'):
            return wordnet.VERB
        elif tag.startswith('N'):
            return wordnet.NOUN
        elif tag.startswith('R'):
            return wordnet.ADV
        else:
            return wordnet.NOUN

class genreClassifier():    
    # Constants
    minimalFreq = 0
    maximalFreq = 1000
    
    
    def fit(self, X, y):
        self.preprocessor = Preprocessor()
        preproc = self.preprocessor.preprocessDataset(X)
        preproc = self.__makeFeatures(preproc)
        self.classifier = OneVsRestClassifier(LogisticRegression(solver='sag'), n_jobs=-1)
        self.classifier.fit(preproc, y)
        
    def predict(self, data):
        preproc = self.preprocessor.preprocessData(data)
        features = self.__remakeFeatures(preproc)
        result = self.classifier.predict_proba(features)
        return result
        
    def __makeFeatures(self, data):        
        self.vectorizer = CountVectorizer(stop_words = None, min_df=self.minimalFreq, max_df=self.maximalFreq, token_pattern=r'[A-Za-z]{2}[A-Za-z]*')
        self.transformer = sklearn.feature_extraction.text.TfidfTransformer()
        
        train_count = self.vectorizer.fit_transform(data)
        train_tf = self.transformer.fit_transform(train_count)
        return train_tf
    
    def __remakeFeatures(self, data):
        vector = self.vectorizer.transform(data)
        tf = self.transformer.transform(vector)
        
        return tf

class ResultProcessing():    
    lib = {
        12 : "Adventure",
        14 : "Fantasy",
        16 : "Animation",
        18 : "Drama",
        27 : "Horror",
        28 : "Action",
        35 : "Comedy",
        36 : "History",
        37 : "Western",
        53 : "Thriller",
        80 : "Crime",
        99 : "Documentary",
        878 : "Science Fiction",
        9648 : "Mystery",
        10402 : "Music",
        10749 : "Romance",
        10751 : "Family",
        10752 : "War",
        10770 : "TV Movie"
    }
    
    def idx_to_cat(self, result_df):    
        result = []
        for key in self.lib.keys():
            if(result_df[key] == 1):
                result.append(self.lib[key])

        return result

class KeywordExtractor:
    def __init__(self):
        self.__max_keyword_length = 3
        self.__words_scores = {}
        self.__words_degrees = {}
        self.__words_frequencies = {}
        self.__candidates_scores = []
    
    def __generate_candidate_keyword(self, original_text):
        self.__set_of_words = set([])
        self.__keyword_candidates = []
        keywords = []
        document = original_text.lower()
        
        # tokenize text
        important_one_char = ['a', 'i', 'o', '.', ',', '?', '!']
        tokens = nltk.tokenize.word_tokenize(document)
        temp_tokens = []
        for t in tokens:
            if len(t) > 1 or (len(t) == 1 and t in important_one_char):
                #remove trailing dots
                if(len(t) > 1 and t[0] == '.'):
                    dot_count = 1
                    for i in range(1, len(t)):
                        if t[i] == t[i-1]:
                            dot_count += 1
                        else:
                            break
                    if dot_count == len(t):
                        temp_tokens.append('.')
                else:
                    dot_position = t.find('.')
                    if dot_position == -1 or dot_position == (len(t) - 1):
                        temp_tokens.append(t)
                    else:
                        if (ord(t[dot_position + 1]) < ord('A') or ord(t[dot_position + 1]) > ord('Z')) and (not t[dot_position + 1] == ' '):
                            new_token = t[:dot_position]
                            temp_tokens.append(new_token)
                            temp_tokens.append(".")
                            new_token = t[(dot_position + 1):]
                            temp_tokens.append(new_token)
                            
        temp_tokens.append(".")
        tokens = temp_tokens
        
        # group candidate keywords
        elimination_list = stopwords.words('english') + ["almost", "even", ",", ".", "'s", "?", "!", "``"]
        keyword_candidates = []
        current_keyword = []
        for i in range (0, len(tokens)):
            if (tokens[i] not in elimination_list):
                current_keyword.append(tokens[i])
                # list unique words document
                self.__set_of_words.add(tokens[i])
            else:
                if (len(current_keyword) > 0):
                    keyword_candidates.append(current_keyword)
                    current_keyword = []
                    
        self.__keyword_candidates = keyword_candidates
    
    def extract_keywords(self, original_text):
        self.__generate_candidate_keyword(original_text)
        self.__calculate_words_scores()
        self.__candidates_scores = []
        self.__candidates_scores_oneword = []
        for i in range(0, len(self.__keyword_candidates)):
            sum_of_score = 0
            for word in self.__keyword_candidates[i]:
                self.__candidates_scores_oneword.append({'key': word, 'score': self.__words_scores[word]})
                sum_of_score += self.__words_scores[word]
            key = ' '.join(self.__keyword_candidates[i])
            self.__candidates_scores.append({'key': key, 'score': sum_of_score})
        
        self.__candidates_scores = sorted(self.__candidates_scores, key=lambda k: k['score'])
        self.__candidates_scores_oneword = sorted(self.__candidates_scores_oneword, key=lambda k: k['score'])
        
    def __calculate_words_scores(self):
        # initialize words scores
        self.__words_scores = {}
        self.__words_degrees = {}
        self.__words_frequencies = {}
        temp_dictionary = {}
        for word in self.__set_of_words:
            word_count = 0
            word_degree = 0
            for i in range(0, len(self.__keyword_candidates)):
                for j in range(0, len(self.__keyword_candidates[i])):
                    if (word == self.__keyword_candidates[i][j]):
                        word_count += 1
                        word_degree += len(self.__keyword_candidates[i])
            
            self.__words_scores[word] = word_degree/word_count
            self.__words_frequencies[word] = word_count
            self.__words_degrees[word] = word_degree
    
    def get_keywords(self, num_of_keywords):
        top_n_keywords = []
        idx = len(self.__candidates_scores) - 1
        while (len(top_n_keywords) < num_of_keywords and idx > -1):
            if (len(self.__candidates_scores[idx]['key'].split(" ")) <= 3):
                top_n_keywords.append(self.__candidates_scores[idx]['key'])
            idx -= 1
        
        idx = len(self.__candidates_scores_oneword) - 1
        while (len(top_n_keywords) < num_of_keywords and idx > -1):
            if (len(self.__candidates_scores_oneword[idx]['key'].split(" ")) <= 3):
                top_n_keywords.append(self.__candidates_scores_oneword[idx]['key'])
            idx -= 1
        return top_n_keywords



if __name__ == "__main__":
    moviedatas = pd.read_json("out4.json", orient='records')
    moviedatas = moviedatas[(moviedatas.astype(str)['genre'] != '[]') & (moviedatas.details != "")]
    # Process result frame
    mlb = MultiLabelBinarizer()
    genredatas = pd.DataFrame(mlb.fit_transform(moviedatas.genre), columns=mlb.classes_)

    # Split data
    train_x, test_x, train_y, test_y = train_test_split(moviedatas.details, genredatas, test_size=0.33, random_state=42)

    # Training for threshold
    clf = genreClassifier()
    clf.fit(train_x, train_y)

    # Test data
    predictions = pd.DataFrame(clf.predict(test_x), columns=mlb.classes_)

    # Threshold calculation
    threshold = {}
    for i in mlb.classes_:
        print(i)
        tprs = []
        fpr, tpr, thresholds = sklearn.metrics.roc_curve(test_y[i], predictions[i])
        delta = tpr - fpr
        max_delta = delta.argmax(axis=0)
        best_th = thresholds[max_delta]
        threshold[i] = best_th
        print("Max delta %f, threshold %f" % (delta.max(), best_th))  
        roc_auc = sklearn.metrics.auc(fpr,tpr)
    
    def custom_threshold(df):
        col = df.columns.values
        for i in col:        
            df[i] = df[i] >= threshold[i]
        
        return df
            
    predictions = custom_threshold(predictions)
    
    # Report score
    rp = ResultProcessing()
    report = metrics.classification_report(test_y, predictions)
    print(report)

    for c in mlb.classes_:
        print("Score for category %d (%s)" % (c,rp.lib[c]),metrics.accuracy_score(test_y[c], predictions[c]))
        print(metrics.confusion_matrix(test_y[c],predictions[c]))
        print()
        
        
    # Full training
    clf = genreClassifier()
    clf.fit(moviedatas.details, genredatas)    
    
    import dill
    with open("clf.pkl", "wb") as f:
        dill.dump(clf.classifier, f)

    with open("vect.pkl", "wb") as f:
        dill.dump(clf.vectorizer, f)

    with open("trf.pkl", "wb") as f:
        dill.dump(clf.transformer, f)

    with open("mlb.pkl", "wb") as f:
        dill.dump(mlb, f)
        
    with open("thr.pkl", "wb") as f:
        dill.dump(threshold, f)
    print("Classifier saved!")
    
    