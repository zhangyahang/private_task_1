from pandas.io.json import json_normalize
import pandas as pd
import json
data_str=open('CMID.json',encoding='utf-8').read()
data_list=json.loads(data_str)
data=[[d['originalText'],d['entities'],d['seg_result'],d['label_4class'],d['label_36class']] for d in data_list]
print(len(data))
df_2=pd.DataFrame(data,columns=['originalText','entities','seg_result','label_4class','label_36class'])

df=df_2

#d = {'label_4class':df['label_4class'].value_counts().index, 'count': df['label_4class'].value_counts()}
#print(d)
#df_label_4class= pd.DataFrame(data=d).reset_index(drop=True)
#print(df_label_4class)

for i in range(12254):
    if df['label_4class'][i]==["'病症'"] or df['label_4class'][i]==['病症']:
        df['label_4class'][i]=int(0)
        
for i in range(12254):
    if df['label_4class'][i]==["'药物'"] or df['label_4class'][i]==['药物']:
        df['label_4class'][i]=int(1)
for i in range(12254):
    if df['label_4class'][i]==["'治疗方案'"] or df['label_4class'][i]==['治疗方案']:
        df['label_4class'][i]=int(2)
for i in range(12254):
    if df['label_4class'][i]==["'其他'"] or df['label_4class'][i]==['其他']:
        df['label_4class'][i]=int(3)
        
d = {'label_4class':df['label_4class'].value_counts().index, 'count': df['label_4class'].value_counts()}
df_label_4class= pd.DataFrame(data=d).reset_index(drop=True)
print(df_label_4class)

#定义删除除字母,数字，汉字以外的所有符号的函数
def remove_punctuation(line):
    line = str(line)
    if line.strip()=='':
        return ''
    rule = re.compile(u"[^a-zA-Z0-9\u4E00-\u9FA5]")
    line = rule.sub('',line)
    return line
 
def stopwordslist(filepath):  
    stopwords = [line.strip() for line in open(filepath, 'r', encoding='utf-8').readlines()]  
    return stopwords  
 
#加载停用词
stopwords = stopwordslist("chineseStopWords.txt")





import pandas as pd
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import jieba as jb
import re
#分词，并过滤停用词
df['cut_originalText'] = df['originalText'].apply(lambda x: " ".join([w for w in list(jb.cut(x)) if w not in stopwords]))
print(df.head())

from sklearn.feature_extraction.text import TfidfVectorizer
 
tfidf = TfidfVectorizer(norm='l2', ngram_range=(1, 2))
features = tfidf.fit_transform(df.cut_originalText)
labels = df.label_4class
print(features.shape)
print('-----------------------------')
print(features)

labels=labels.tolist()



from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
X_train, X_test, y_train, y_test = train_test_split(df['cut_originalText'], df['label_4class'], random_state = 0)
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(X_train)
 
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
y_train=y_train.astype('int')
clf = RandomForestClassifier().fit(X_train_tfidf, y_train)
y_test=y_test.astype('int')
X_test_counts = count_vect.transform(X_test)
X_test_tfidf = tfidf_transformer.transform(X_test_counts)
y_pred=clf.predict(X_test_tfidf)
from sklearn.metrics import classification_report

print(accuracy_score(y_test, y_pred))
print(classification_report(y_test,y_pred))
print(confusion_matrix(y_test, y_pred))
cfm = confusion_matrix(y_test, y_pred)
plt.matshow(cfm, cmap=plt.cm.gray)
plt.show()





