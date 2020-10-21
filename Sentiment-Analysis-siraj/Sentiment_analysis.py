from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from numpy import *

train = pd.read_csv('train.csv', encoding='latin-1')
test = pd.read_csv('test.csv', encoding='latin-1')

train.drop("ItemID", axis=1)
x = train["SentimentText"]

# data has contains number+string so using regular expression removing intigers
import re

patt = re.compile(r'[a-z]*\s+', re.I)
u = list(map(lambda s: ''.join(patt.findall(s)), x[:35000]))   # selecting only 21000 string
u = list(map(lambda s: s.strip(), u))  # removing space from front and last

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk

nltk.download('stopwords')
nltk.download('punkt')
stop_words = set(stopwords.words('english'))

# removing stop word using natural language tool kit nltk
arr = []
for m in u:
    word_tokens = word_tokenize(m)

    arr.append(" ".join([w for w in word_tokens if not w in stop_words]))

# removing repeat word which have repeat char in string
final_arr = []
for string in arr:
    for word in string.split(" "):
        if len(re.findall(r"((\w)\2{2,})", word)) != 0:
            string = string.replace(word, "")
    final_arr.append(string)

# removing empty string in list
final_arr = list(filter(None, final_arr))

# encoding string to train machine learning model.
feature_extra = TfidfVectorizer()
feature_extra.fit(arr)

print(feature_extra.vocabulary_)
print(feature_extra.get_feature_names())
print(feature_extra.idf_)

l = feature_extra.transform(arr).toarray()
print("encoded string data:")
print(l)

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import r2_score

# training Logistic Regression machine learning model
y = train["Sentiment"]
model = LogisticRegression()
model.fit(l[:34000], y[:34000])

print("Logistic Regression model intercepts: {} ".format( model.intercept_))
print("Logistic Regression model coefficient: {} ".format(model.coef_))

prediction = model.predict(l[34000:35000])

print("Accuracy: {} ".format(r2_score(prediction, y[34000:35000])))



while True:
    string = input("enter string: ")
    t = feature_extra.transform([string])
    pre = model.predict(t.toarray())
    if pre[0] == 1:
        print("positive string")
    else:
        print("negative string")
    if string == "exit":
        break
