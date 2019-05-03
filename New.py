import pandas as pd 
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from textblob import Word
from gensim.models import Word2Vec
import glob

def addSentenceID(data):

    list0Value = data.index[data['cycle'] == 0].tolist()
    listLabelNotNull = data[data['y'].notnull()].index.tolist()
    # print(list0Value)
    # print(listLabelNotNull)
    index=0
    for i in list0Value:
        firstIndex = i
        lastIndex = listLabelNotNull[index]
        index = index + 1
        for j in range(firstIndex, lastIndex+1):
            data.at[j, 'sentence'] = index
    return data

def getOnlyTextWithChatID(data):

    formattedData = data[data['sentence'].notnull()]
    formattedData = formattedData[['text','sentence','y', 'dialogueID']]
    newFormattedData = pd.DataFrame()

    for i in set(formattedData['sentence']):
            textFrame = formattedData[formattedData['sentence'] == i]
            labelFrame = textFrame.loc[textFrame['y'].notnull(), ['y']]
            dialogueID = list(textFrame['dialogueID'].unique())[0]
            label = float(labelFrame['y'])
            textValue = ""
            for value in list(textFrame['text']):
                textValue = textValue + " " + str(value)
            textValue = textValue.strip()
            newFormattedData = newFormattedData.append({'ChatID':dialogueID,'Chat':textValue,'y':label}, ignore_index=True)

    return newFormattedData

def preprocessingData(dataFrame):

    # Removing URL
    dataFrame['Chat'] = dataFrame['Chat'].replace(r'http\S+', '', regex=True).replace(r'www\S+', '', regex=True)
    # Converting to lowercase
    dataFrame['Chat'] = dataFrame['Chat'].apply(lambda x: " ".join(x.lower() for x in x.split()))
    # Removing punctuation
    dataFrame['Chat'] = dataFrame['Chat'].str.replace('[^\w\s]','')
    # Removing stopwords
    stop2 = stopwords.words('english')
    dataFrame['Chat'] = dataFrame['Chat'].apply(lambda x: " ".join(x for x in x.split() if x not in stop2))
    # Removing common words
    freq = pd.Series(' '.join(dataFrame['Chat']).split()).value_counts()[:10]
    freq = list(freq.index)
    dataFrame['Chat']= dataFrame['Chat'].apply(lambda x: " ".join(x for x in x.split() if x not in freq))
    # Removing rare words
    freq2 = pd.Series(' '.join(dataFrame['Chat']).split()).value_counts()[-10:]
    freq2 = list(freq2.index)
    dataFrame['Chat'] = dataFrame['Chat'].apply(lambda x: " ".join(x for x in x.split() if x not in freq2))
    # Removing numerics
    dataFrame['Chat'] = dataFrame['Chat'].str.replace(r'\d+','')
    # Stemming
    st = PorterStemmer()
    dataFrame['Chat'][:5].apply(lambda x: " ".join([st.stem(word) for word in x.split()]))
    # Lemmatization
    dataFrame['Chat'] = dataFrame['Chat'].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))
    
    return dataFrame

def getOnlySentences(dataFrame):
    dataFrame['words']= dataFrame.Chat.str.split('[\W_]+')
    sentences = list(dataFrame['words'])
    return sentences


def getWord2VecModel(sentences):
    model = Word2Vec(sentences, min_count=1) 
    model.train(sentences, total_examples=len(sentences), epochs=10)
    return model


def getWordVectors(model, sentences):
    for sentence in sentences:
        for  word in sentence:
            vector = model.wv[word]
            return vector

def getSentVec(vectorList):
    tempDf = pd.DataFrame(vectorList)
    tempDf.loc['mean'] = tempDf.mean(axis=0)
    return list(tempDf.loc['mean'])



# Loading data from file pre3.csv

data = pd.concat([pd.read_csv(f) for f in glob.glob('*.csv')], ignore_index = True)



data = data[['y', 'chatter','cycle', 'date', 'dialogueID', 'folder', 'from1', 'text', 'to']]


#fileName = input('Enter the filename along with full path (if code path is different to file path else only file name):')
#data = pd.read_csv(fileName,index_col=0)

# Adding column 'sentence' to that dataframe which shows the sentence ID against each row
data = addSentenceID(data)


#  create a new dataframe with columns Chat (text after removing blank spaces) and ChatID (a integer to each chat)
formattedData = getOnlyTextWithChatID(data)

# preprocessing data 
# Removing URL, punctuation, stopwords, common words, rare words, numerics
# Converting to lower case
# Stemming and Lemmatization
formattedData = preprocessingData(formattedData)

# getOnlySentences
sentences = getOnlySentences(formattedData)

# Word2Vec model for processed data
model = getWord2VecModel(sentences)

formattedData = formattedData[['ChatID','y']]

# get sentence vector each sentence:
formattedData['message_vector'] = 'Not Available'
formattedData['role'] = 'role'


index = 0
for sentence in sentences:
    vectorList = []
    for  word in sentence:
        listValues = model.wv[word].tolist()
        vectorList.append(listValues)
    sentenceVector = getSentVec(vectorList)
    formattedData.at[index,'message_vector'] = sentenceVector
    index = index + 1


# word vectors
#getWordVectors(model, sentences)
# Change this path
formattedData.to_csv('data with vector.csv')








