import numpy as np
import nltk
import pandas as pd
import re
import requests
from bs4 import BeautifulSoup
import traceback
import textstat
import pyphen
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.tokenize import RegexpTokenizer
tokenizer = RegexpTokenizer(r'\w+')
# nltk.download('all')


dic = pyphen.Pyphen(lang='en')

clean = re.compile('<.*?>')

df = pd.read_excel('./Input.xlsx')
df.head()

# get data function to  read the content and save it locally in files name as url_ids
def get_data(df):
    links = df['URL']
    ids = df['URL_ID']
    j = 0
    text = []
    for id in ids:
        # print(i)
        result = requests.get(links[j], headers={
            'Accept': "*/*",
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'User-Agent': 'User'
        })
        content = result.content
        content = content.decode(
            'utf-8').encode('cp850', 'replace').decode('cp850')
        soup = BeautifulSoup(content, 'html.parser')
        try:
            header = soup.find('h1', class_='entry-title').string

            try:
                soup.find('pre', class_='wp-block-preformatted').decompose()
            except:
                pass
            div = soup.find_all('div', {'class': 'td-post-content'})

            divStr = str(div[0])
            cleanText = re.sub(clean, '', divStr)
            totalText = header + '. ' + cleanText
            temp = re.sub('\n|\xa0|(|)|"|,|', '', totalText)
            with open('./TextFiles/'+str(int(id)) + '.txt', 'w') as f:
                f.write(temp)

        except Exception:
            traceback.print_exc()
            print(j, '-->', id)
            with open('./TextFiles/' + str(int(id)) + '.txt', 'w') as f:
                f.write(' ')
        j = j+1
    return text

# executing get data
get_data(df)


# extracting stop words from stopword directory and CAPITALIZE the words
auditor = ''
currencies = ''
dates = ''
gen = ''
genlong = ''
geo = ''
names = ''

with open('./StopWords/StopWords_Auditor.txt', 'r') as f:
    auditor = f.read()

with open('./StopWords/StopWords_Currencies.txt', 'r') as f:
    currencies = f.read()

with open('./StopWords/StopWords_DatesandNumbers.txt', 'r') as f:
    dates = f.read()

with open('./StopWords/StopWords_Generic.txt', 'r') as f:
    gen = f.read()

with open('./StopWords/StopWords_GenericLong.txt', 'r') as f:
    genlong = f.read()


with open('./StopWords/StopWords_Geographic.txt', 'r') as f:
    geo = f.read()

with open('./StopWords/StopWords_Names.txt', 'r') as f:
    names = f.read()

auditor = auditor.split('\n')
c = currencies.split('\n')
temp = []
for i in c:
    j = i.split('|')
    temp.append(j[0].strip())
temp
currencies = temp
currencies[:-1]
dates = dates.replace('\n', '*').replace('|', '*')
dates = dates.split('*')
dates
temp = []
for d in dates:
    j = d.split('|')
    temp.append(j[0].strip())
dates = temp
dates
gen = gen.split('\n')
gen

genlong = genlong.upper().split('\n')
# print(genlong)
geo = geo.split('\n')
geo
temp = []
for i in geo:
    j = i.split('|')
    temp.append(j[0].strip())
geo = temp
geo
names = names.split('\n')
temp = []
for i in names:
    j = i.split('|')
    temp.append(j[0].strip())
names = temp
names
stopwords = auditor + currencies + dates + gen + genlong + geo + names


#extracting negative and positive words
positive = open('./MasterDictionary/positive-words.txt', 'r').read()
positive = positive.upper().split('\n')
negative = open('./MasterDictionary/negative-words.txt', 'r').read()
neagtive = negative.upper().split('\n')

POSITIVE_SCORE = []
NEGATIVE_SCORE = []
POLARITY_SCORE = []
SUBJECTIVITY_SCORE = []
AVG_SENTENCE_LENGTH = []
PERCENTAGE_OF_COMPLEX_WORDS = []
FOG_INDEX = []
AVG_NUMBER_OF_WORDS_PER_SENTENCE = []
COMPLEX_WORD_COUNT = []
WORD_COUNT = []
SYLLABLE_PER_WORD = []
PERSONAL_PRONOUN = []
AVG_WORD_LENGTH = []


for id in df['URL_ID']:
    text = open('./TextFiles/' + str(int(id))+'.txt').read()
    text = text.upper()
    sentances = sent_tokenize(text)
    words = tokenizer.tokenize(text)

    def filter_stopwords(word):
        return True if word not in stopwords else False

    filtered_words = tuple(filter(filter_stopwords, words))
    filtered_words

    p_count = 0
    n_count = 0
    t_count = len(filtered_words)
    #count positive negative
    for word in filtered_words:
        if word in positive:
            p_count = p_count + 1
        if word in negative:
            n_count = n_count + 1
    #print(p_count,n_count, t_count)
    POSITIVE_SCORE.append(p_count)
    NEGATIVE_SCORE.append(n_count)
    # polarity score and subjectivity score
    polarity_score = (p_count - n_count) / (p_count + n_count + 0.000001)
    POLARITY_SCORE.append(polarity_score)
    subjectivity_score = (p_count + n_count) / (t_count+0.000001)
    subjectivity_score
    SUBJECTIVITY_SCORE.append(subjectivity_score)
    average_sen_len = len(words) / (len(sentances) + 0.000001)
    AVG_SENTENCE_LENGTH.append(average_sen_len)

    def com_filter(word):
        temp = dic.inserted(word)
        temp = temp.split('-')
        return True if len(temp) > 2 else False
    cw = tuple(filter(com_filter, filtered_words))
    COMPLEX_WORD_COUNT.append(len(cw))
    pcw = len(cw) / (len(filtered_words) + 0.000001)
    PERCENTAGE_OF_COMPLEX_WORDS.append(pcw)

    fog_index = 0.4 * (subjectivity_score+pcw)
    #fog index
    FOG_INDEX.append(fog_index)
    WORD_COUNT.append(len(filtered_words))
    char_count = textstat.char_count(text, ignore_spaces=True)
    char_count
    sly_count = 0
    for w in filtered_words:
        temp = dic.inserted(w)
        temp = temp.split('-')
        sly_count = sly_count + len(temp)
    (sly_count/(len(filtered_words) + 0.000001))
    SYLLABLE_PER_WORD.append((sly_count/(len(filtered_words) + 0.000001)))
    avg_word_length = (char_count/(len(words) + 0.000001))
    avg_word_length
    awps = len(words) / (len(sentances) + 0.000001)
    AVG_NUMBER_OF_WORDS_PER_SENTENCE.append(awps)
    AVG_WORD_LENGTH.append(avg_word_length)
    # print(avg_word_length)
    #pronouns
    pronounRegex = re.compile(r'I|we|my|ours|us', re.I)
    pronouns = pronounRegex.findall(text)
    PERSONAL_PRONOUN.append(len(pronouns))

df['POSITIVE SCORE'] = np.array(POSITIVE_SCORE).tolist()
df['NEGATIVE SCORE'] = np.array(NEGATIVE_SCORE).tolist()
df['POLARITY SCORE'] = np.array(POLARITY_SCORE).tolist()
df['SUBJECTIVITY SCORE'] = np.array(SUBJECTIVITY_SCORE).tolist()
df['AVG SENTENCE LENGTH'] = np.array(AVG_SENTENCE_LENGTH).tolist()
df['PERCENTAGE OF COMPLEX WORDS'] = np.array(
    PERCENTAGE_OF_COMPLEX_WORDS).tolist()
df['FOG INDEX'] = np.array(FOG_INDEX).tolist()
df['AVG NUMBER OF WORDS PER SENTENCE'] = np.array(
    AVG_NUMBER_OF_WORDS_PER_SENTENCE).tolist()
df['COMPLEX WORD COUNT'] = np.array(COMPLEX_WORD_COUNT).tolist()
df['WORD COUNT'] = np.array(WORD_COUNT).tolist()
df['SYLLABLE PER WORD'] = np.array(SYLLABLE_PER_WORD).tolist()
df['PERSONAL PRONOUNS'] = np.array(PERSONAL_PRONOUN).tolist()
df['AVG WORD LENGTH'] = np.array(AVG_WORD_LENGTH).tolist()
#save df as excel
writer = pd.ExcelWriter("Output.xlsx", engine='xlsxwriter')
df.to_excel(writer, sheet_name='TextAnalysis', index=False)
writer.save()
