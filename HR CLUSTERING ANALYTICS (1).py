#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from datetime import date,timedelta,datetime
import warnings
import matplotlib.pyplot as plt 
import seaborn as sns
from pylab import rcParams
get_ipython().run_line_magic('matplotlib', 'inline')
warnings.filterwarnings("ignore")
from sklearn import preprocessing
import pyodbc
from datetime import datetime
today = datetime.today()
import numpy as np
import time
from dateutil.relativedelta import relativedelta


# In[2]:


start = time.time()
conn = pyodbc.connect('Driver={SQL Server};'
                      'Server=ABP-D;'     
                      'Database=HR;'
                      'UID=user;'
                      'PWD=password')
Demo_script = 'select * from HR.dbo.HR_Demo'
#script_acc = 'select * from CHURN_BASELINE_TRAIN.dbo.ACCOUNTS'#select top 1000 * from CHURN_BASELINE_TRAIN.dbo.DEMOGRAPHICSLAST_SEEN_DATE


dfDemo = pd.read_sql_query(Demo_script, conn, index_col=None,parse_dates=True)

conn.close()
end = time.time()
print(end - start)


# In[3]:


dfDemo.head(2)


# In[4]:


dfDemo.shape


# In[5]:


dfDemo.columns


# In[6]:





# In[8]:


dfDemo.GRADE.unique()


# ##  Select only the professional staff

# In[9]:




# ## Read Atm_uptime

# In[10]:


Atm_uptime= pd.read_excel(open(r'C:\Users\stu\Downloads\DATAPOINTS\BRANCH.xlsx', 'rb'),
              sheet_name='UPTIME SUMMARY') 


# In[11]:


Atm_uptime["AVERAGE BRANCH UPTIME SCORE"]=Atm_uptime["AVERAGE BRANCH UPTIME"].multiply(100)


# In[12]:


Atm_uptime.head(1)


# In[13]:


Atm_uptime.shape


# ## Audited scores

# In[14]:


Audit19= pd.read_excel(open(r'C:\Users\stu\Downloads\Audit_Scores_Dec_20192.xlsx', 'rb'),
              sheet_name='Scores_Branch')
del Audit19['Unnamed: 0']


# In[15]:


Audit19=Audit19[['BRANCH ','Branch Code','AVERAGE CUSTOMER EXPERIENCE ','AVERAGE AMBIENCE','SERVICE AVERAGE','ATMs AVERAGE']]


# In[16]:


Audit19['AVERAGE CUSTOMER EXPERIENCE SCORE']=Audit19['AVERAGE CUSTOMER EXPERIENCE '].div(5).multiply(100)
Audit19['AVERAGE AMBIENCE SCORE']=Audit19['AVERAGE AMBIENCE'].div(5).multiply(100)
Audit19['SERVICE AVERAGE SCORE']=Audit19['SERVICE AVERAGE'].div(5).multiply(100)
Audit19['ATMs AVERAGE SCORE']=Audit19['ATMs AVERAGE'].div(5).multiply(100)


# In[17]:


Audit19=Audit19[['BRANCH ','Branch Code','AVERAGE CUSTOMER EXPERIENCE SCORE','AVERAGE AMBIENCE SCORE','SERVICE AVERAGE SCORE','ATMs AVERAGE SCORE']]


# In[18]:


Audit19.shape


# In[19]:


AuditandAtm =pd.merge(Atm_uptime,Audit19,how ='inner', left_on=['BRANCH CODE'], right_on=['Branch Code'])


# In[20]:


AuditandAtm.head()


# ## One Bank

# In[21]:


#one_bank = pd.read_excel(r'C:\Users\iyaniwuraa\Downloads\ONE BANK 900 CAMPAIGN_ BACK AND MID OFFICE.xlsx')


# In[22]:


OnebankNEW= pd.read_excel(open(r'C:\Users\stu\Downloads\ONEBANK_TENORED_NON_MKT_FACING (82).xlsx', 'rb'),
              sheet_name='Sheet1') 
del OnebankNEW['Unnamed: 1']
del OnebankNEW['Unnamed: 3']


# In[23]:


OnebankNEW.head(1)


# In[24]:


#one_bank_retail= one_bank[one_bank['GROUP NAME']=='Retail Operations']


# In[25]:


#one_bank_retail =one_bank_retail[['STAFF NAME','STAFF ID','STAFF GRADE','HOUSE',"weighted % Ach'mt"]]


# In[26]:


OnebankNEW['STAFF NO']=OnebankNEW['STAFF NO'].replace('\/','', regex=True)


# In[27]:


OnebankNEW.columns


# In[28]:


OnebankNEWRet=OnebankNEW[['STAFF NAME','STAFF NO','STAFF GRADE','HOUSE','%AGE ACHIEVED (DEPOSIT GROWTH)  AVERAGE']]


# In[29]:


OnebankNEWRet['STAFF NO']


# In[30]:


#one_bank_retail["WEIGHTED SCORE"]=one_bank_retail["weighted % Ach'mt"].multiply(100)


# In[31]:


OnebankNEWRet.head(1)


# In[32]:


#one_bank_retail['STAFF ID']=one_bank_retail['STAFF ID'].replace('\/','', regex=True)


# In[33]:


#one_bank_retail['STAFF ID'] = one_bank_retail['STAFF ID'].str.replace('/', '')


# In[34]:


OnebankNEWRet.shape


# In[35]:


AllROG= pd.read_excel(open(r'C:\Users\stu\Downloads\DATAPOINTS\Copy of ALL ROG STAFF DATA.xlsx', 'rb'),
              sheet_name='Source') 


# In[36]:


AllROG.head(1)


# In[37]:


AllROG.columns


# In[38]:


AllROG['CATEGORY OF STAFF (ROLE IN BRANCH)'].unique()


# In[39]:


GRADE_ROG = ['BRANCH SERVICE MANAGER','ASST BRANCH SERVICE MANAGER']
AllROG_pro =AllROG[AllROG['CATEGORY OF STAFF (ROLE IN BRANCH)'].isin(GRADE_ROG)]


# In[40]:


(AllROG_pro['STAFF EMPLOYEE ID'].str.startswith('100')).value_counts()


# In[41]:


AllROG_pro.columns


# In[42]:


AllROG_pro['STAFF EMPLOYEE ID']=AllROG_pro['STAFF EMPLOYEE ID'].replace('\/','', regex=True)


# In[43]:


#AllROG['STAFF EMPLOYEE ID'] = AllROG['STAFF EMPLOYEE ID'].str.replace('/', '')


# In[44]:


AllROG_pro=AllROG_pro[['BRANCH CODE','BRANCH NAME','STAFF EMPLOYEE ID', 'FULL NAME','CATEGORY OF STAFF (ROLE IN BRANCH)']]


# In[45]:


AllROG_pro.head(10)


# In[46]:


AllROG_pro.shape


# In[47]:


#ROG_Confirmation.columns


# In[48]:


#ROG_Confirmation = pd.read_excel(r'C:\Users\iyaniwuraa\Downloads\ROG Confirmation.xlsx')
#ROG_Confirmation.head(1)


# In[49]:


AllROGandonebank =pd.merge(AllROG_pro,OnebankNEWRet,how ='inner', left_on=['STAFF EMPLOYEE ID'], right_on=['STAFF NO'])


# In[50]:


AllROGandonebank.shape


# In[51]:


AllROGandonebank['STAFF GRADE'].unique()


# In[52]:


Final = pd.merge(AuditandAtm,AllROGandonebank,how ='inner', left_on=['Branch Code'], right_on=['BRANCH CODE'])


# In[53]:


Final.columns


# In[54]:


Final  =Final[['STAFF EMPLOYEE ID', 'AVERAGE BRANCH UPTIME SCORE','AVERAGE CUSTOMER EXPERIENCE SCORE','AVERAGE AMBIENCE SCORE', 'SERVICE AVERAGE SCORE', 'ATMs AVERAGE SCORE','%AGE ACHIEVED (DEPOSIT GROWTH)  AVERAGE']]


# In[55]:


Final =Final.set_index('STAFF EMPLOYEE ID')


# In[56]:


Final.shape


# In[57]:


Final = Final.replace([np.inf, -np.inf], np.nan)
Final = Final.fillna(0)


# In[58]:


import numpy as np
import sys
import matplotlib.pyplot as pyt
import pandas as pd
from datetime import date   
date = date.today()
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
sil_score = {}
clust = {}
import warnings
warnings.filterwarnings("ignore")


# In[59]:


label_group = {}
label_group[3] = {1:'Low Performance',2:'Average Performance',3:'High Performance'}
label_group[4] = {1:'Low Performance',2:'Average Performance',3:'Above Average Performance',4:'High Performance'}
label_group[5] = {1:'Very Low Performance',2:'Low Performance',3:'Average Performance',4:'Above Average Performance',5:'High Performance'}


# In[60]:


label_group


# In[61]:


Final.head(10)


# In[62]:


df_cluster = Final.copy()
df_transform = Final.copy()


# In[63]:


Final.columns


# In[64]:


uptime = Final[['AVERAGE BRANCH UPTIME SCORE']]
uptime_complete= df_transform[['AVERAGE BRANCH UPTIME SCORE']]
from sklearn.decomposition import PCA
#pca = PCA(n_components=2)
#counts = pca.fit_transform(counts)
#counts_complete = pca.fit_transform(counts_complete)
sil = {}
for i in range(3,5):
    kmeans = KMeans(n_clusters=i,init='k-means++',max_iter=300,n_init=10,random_state=0,precompute_distances='auto',copy_x=True,n_jobs=-1,algorithm ='auto')
    kmeans.fit(uptime)
    y_clust=kmeans.predict(uptime)
    sil[i] = silhouette_score(uptime, y_clust, metric='euclidean')
k = max(sil, key=sil.get)
kmeans = KMeans(n_clusters=k,init='k-means++',max_iter=300,n_init=10,random_state=0,precompute_distances='auto',copy_x=True,n_jobs=-1,algorithm ='auto') 
y_kmeans = kmeans.fit_predict(uptime)
sil_score['uptime'] = silhouette_score(uptime, y_kmeans, metric='euclidean')
clust['uptime'] = k
df_cluster['uptime'] = y_kmeans
uptime_complete['value'] = y_kmeans


# In[65]:


customerExp = Final[['AVERAGE CUSTOMER EXPERIENCE SCORE']]
customerExp_complete = df_transform[['AVERAGE CUSTOMER EXPERIENCE SCORE']]
sil = {}
for i in range(3,5):
    kmeans = KMeans(n_clusters=i,init='k-means++',max_iter=300,n_init=10,random_state=0,precompute_distances='auto',copy_x=True,n_jobs=-1,algorithm ='auto')
    kmeans.fit(customerExp)
    y_clust=kmeans.predict(customerExp)
    sil[i] = silhouette_score(customerExp, y_clust, metric='euclidean')
k = max(sil, key=sil.get)
kmeans = KMeans(n_clusters=k,init='k-means++',max_iter=300,n_init=10,random_state=0,precompute_distances='auto',copy_x=True,n_jobs=-1,algorithm ='auto') 
y_kmeans = kmeans.fit_predict(customerExp)
sil_score['customerExp'] = silhouette_score(customerExp, y_kmeans, metric='euclidean')
clust['customerExp'] = k
df_cluster['customerExp'] = y_kmeans
customerExp_complete['value'] = y_kmeans


# In[66]:


ambience = Final[['AVERAGE AMBIENCE SCORE']]
ambience_complete = df_transform[['AVERAGE AMBIENCE SCORE']]
sil = {}
for i in range(3,5):
    kmeans = KMeans(n_clusters=i,init='k-means++',max_iter=300,n_init=10,random_state=0,precompute_distances='auto',copy_x=True,n_jobs=-1,algorithm ='auto')
    kmeans.fit(ambience)
    y_clust=kmeans.predict(ambience)
    sil[i] = silhouette_score(ambience, y_clust, metric='euclidean')
k = max(sil, key=sil.get)
kmeans = KMeans(n_clusters=k,init='k-means++',max_iter=300,n_init=10,random_state=0,precompute_distances='auto',copy_x=True,n_jobs=-1,algorithm ='auto') 
y_kmeans = kmeans.fit_predict(ambience)
sil_score['ambience'] = silhouette_score(ambience, y_kmeans, metric='euclidean')
clust['ambience'] = k
df_cluster['ambience'] = y_kmeans
ambience_complete['value'] = y_kmeans


# In[67]:


service = Final[['SERVICE AVERAGE SCORE']]
service_complete = df_transform[['SERVICE AVERAGE SCORE']]
sil = {}
for i in range(3,5):
    kmeans = KMeans(n_clusters=i,init='k-means++',max_iter=300,n_init=10,random_state=0,precompute_distances='auto',copy_x=True,n_jobs=-1,algorithm ='auto')
    kmeans.fit(service)
    y_clust=kmeans.predict(service)
    sil[i] = silhouette_score(service, y_clust, metric='euclidean')
k = max(sil, key=sil.get)
kmeans = KMeans(n_clusters=k,init='k-means++',max_iter=300,n_init=10,random_state=0,precompute_distances='auto',copy_x=True,n_jobs=-1,algorithm ='auto') 
y_kmeans = kmeans.fit_predict(service)
sil_score['service'] = silhouette_score(service, y_kmeans, metric='euclidean')
clust['service'] = k
df_cluster['service'] = y_kmeans
service_complete['value'] = y_kmeans


# In[68]:


atm_score = Final[['ATMs AVERAGE SCORE']]
atm_score_complete = df_transform[['ATMs AVERAGE SCORE']]
sil = {}
for i in range(3,5):
    kmeans = KMeans(n_clusters=i,init='k-means++',max_iter=300,n_init=10,random_state=0,precompute_distances='auto',copy_x=True,n_jobs=-1,algorithm ='auto')
    kmeans.fit(atm_score)
    y_clust=kmeans.predict(atm_score)
    sil[i] = silhouette_score(atm_score, y_clust, metric='euclidean')
k = max(sil, key=sil.get)
kmeans = KMeans(n_clusters=k,init='k-means++',max_iter=300,n_init=10,random_state=0,precompute_distances='auto',copy_x=True,n_jobs=-1,algorithm ='auto') 
y_kmeans = kmeans.fit_predict(atm_score)
sil_score['atm_score'] = silhouette_score(atm_score, y_kmeans, metric='euclidean')
clust['atm_score'] = k
df_cluster['atm_score'] = y_kmeans
atm_score_complete['value'] = y_kmeans


# In[69]:


auc = Final[['%AGE ACHIEVED (DEPOSIT GROWTH)  AVERAGE']]
auc_complete = df_transform[['%AGE ACHIEVED (DEPOSIT GROWTH)  AVERAGE']]
sil = {}
for i in range(3,5):
    kmeans = KMeans(n_clusters=i,init='k-means++',max_iter=300,n_init=10,random_state=0,precompute_distances='auto',copy_x=True,n_jobs=-1,algorithm ='auto')
    kmeans.fit(auc)
    y_clust=kmeans.predict(auc)
    sil[i] = silhouette_score(auc, y_clust, metric='euclidean')
k = max(sil, key=sil.get)
kmeans = KMeans(n_clusters=k,init='k-means++',max_iter=300,n_init=10,random_state=0,precompute_distances='auto',copy_x=True,n_jobs=-1,algorithm ='auto') 
y_kmeans = kmeans.fit_predict(auc)
sil_score['auc'] = silhouette_score(auc, y_kmeans, metric='euclidean')
clust['auc'] = k
df_cluster['auc'] = y_kmeans
auc_complete['value'] = y_kmeans


# In[70]:


auc_complete['value']


# In[71]:


measures = ['uptime', 'customerExp','ambience', 'service', 'atm_score','auc']


# In[72]:


df_main = df_cluster[measures]


# In[73]:


uptime_label = pd.DataFrame(uptime_complete.groupby("value")['AVERAGE BRANCH UPTIME SCORE'].mean())
uptime_label.columns = ['label']
uptime_label = uptime_label.sort_values(by='label', axis=0, ascending=True)
uptime_label['adj_label_value'] = np.arange(len(uptime_label))
uptime_label['adj_label_value'] = uptime_label['adj_label_value'] + 1


# In[74]:


customerExp_label = pd.DataFrame(customerExp_complete.groupby("value")['AVERAGE CUSTOMER EXPERIENCE SCORE'].mean())
customerExp_label.columns = ['label']
customerExp_label = customerExp_label.sort_values(by='label', axis=0, ascending=True)
customerExp_label['adj_label_value'] = np.arange(len(customerExp_label))
customerExp_label['adj_label_value'] = customerExp_label['adj_label_value'] + 1


# In[75]:


ambience_label = pd.DataFrame(ambience_complete.groupby("value")['AVERAGE AMBIENCE SCORE'].mean())
ambience_label.columns = ['label']
ambience_label = ambience_label.sort_values(by='label', axis=0, ascending=True)
ambience_label['adj_label_value'] = np.arange(len(ambience_label))
ambience_label['adj_label_value'] = ambience_label['adj_label_value'] + 1


# In[76]:


service_label = pd.DataFrame(service_complete.groupby("value")['SERVICE AVERAGE SCORE'].mean())
service_label.columns = ['label']
service_label = service_label.sort_values(by='label', axis=0, ascending=True)
service_label['adj_label_value'] = np.arange(len(service_label))
service_label['adj_label_value'] = service_label['adj_label_value'] + 1


# In[77]:


atm_score_label = pd.DataFrame(atm_score_complete.groupby("value")['ATMs AVERAGE SCORE'].mean())
atm_score_label.columns = ['label']
atm_score_label = atm_score_label.sort_values(by='label', axis=0, ascending=True)
atm_score_label['adj_label_value'] = np.arange(len(atm_score_label))
atm_score_label['adj_label_value'] = atm_score_label['adj_label_value'] + 1


# In[78]:


auc_score_label = pd.DataFrame(auc_complete.groupby("value")['%AGE ACHIEVED (DEPOSIT GROWTH)  AVERAGE'].mean())
auc_score_label.columns = ['label']
auc_score_label = auc_score_label.sort_values(by='label', axis=0, ascending=True)
auc_score_label['adj_label_value'] = np.arange(len(auc_score_label))
auc_score_label['adj_label_value'] = auc_score_label['adj_label_value'] + 1


# In[79]:


auc_score_label['adj_label_value']


# In[80]:


df_main['uptime'] = df_main.uptime.map( uptime_label.to_dict(orient='dict')['adj_label_value'] )
df_main['customerExp'] = df_main.customerExp.map( customerExp_label.to_dict(orient='dict')['adj_label_value'])
df_main['ambience'] = df_main.ambience.map( ambience_label.to_dict(orient='dict')['adj_label_value'] )
df_main['service'] = df_main.service.map( service_label.to_dict(orient='dict')['adj_label_value'])
df_main['atm_score'] = df_main.atm_score.map( atm_score_label.to_dict(orient='dict')['adj_label_value'] )
df_main['auc'] = df_main.auc.map(auc_score_label.to_dict(orient='dict')['adj_label_value'])


# In[81]:


scorecard = round(df_main/df_main.max() * 100)


# In[82]:


scorecard['total'] = round(scorecard.sum(axis=1)/(len(scorecard.columns)))


# In[83]:


scorecard


# In[84]:


sil = {}
cluster_total = pd.DataFrame (scorecard.loc[:,'total'])
cluster_total_complete = cluster_total.copy()
#for i in range(3,8):
#    kmeans = KMeans(n_clusters=i,init='k-means++',max_iter=300,n_init=10,random_state=0,precompute_distances='auto',copy_x=True,n_jobs=-1,algorithm ='auto')
#    kmeans.fit(cluster_total)
#    y_clust=kmeans.predict(cluster_total)
#    sil[i] = silhouette_score(cluster_total, y_clust, metric='euclidean')
#k = max(sil, key=sil.get)
k = 5
kmeans = KMeans(n_clusters=k,init='k-means++',max_iter=300,n_init=10,random_state=0,precompute_distances='auto',copy_x=True,n_jobs=-1,algorithm ='auto') 
y_kmeans = kmeans.fit_predict(cluster_total)
sil_score['total'] = silhouette_score(cluster_total, y_kmeans, metric='euclidean')
df_cluster['cluster_total'] = y_kmeans
cluster_total_complete['value'] = y_kmeans
df_main['total'] = y_kmeans
scorecard['cluster'] = y_kmeans
clust['total'] = k


# In[85]:


cluster_total_label = pd.DataFrame(cluster_total_complete.groupby("value").total.mean())
cluster_total_label.columns =['label']
cluster_total_label = cluster_total_label.sort_values(by='label', axis=0, ascending=True)
cluster_total_label['adj_label_value'] = np.arange(len(cluster_total_label))
cluster_total_label['adj_label_value'] = cluster_total_label['adj_label_value'] + 1
df_main['total'] = df_main.total.map( cluster_total_label.to_dict(orient='dict')['adj_label_value'] )
scorecard['cluster'] = cluster_total_complete.value.map( cluster_total_label.to_dict(orient='dict')['adj_label_value'] )


# In[86]:


count = len(scorecard.cluster.unique())
labels = label_group[count] 
scorecard['label'] = scorecard['cluster'] 
scorecard = scorecard.replace({"label": labels})
columns = df_main.columns
l_column = ['uptime_label','customerExp_label','ambience_label','service_label','atm_score_label','auc_score_label','overall_label']
for i,x in zip(columns,l_column):
    count = ''
    count = len(df_main[i].unique())
    labels = label_group[count]
    df_main[x] = df_main[i] 
    df_main = df_main.replace({x: labels})


# In[87]:


df_main[i]


# In[88]:


scorecard.cluster.unique()


# In[89]:


label_group


# In[90]:


len(df_main[i].unique())


# In[91]:


df_main


# In[92]:


sil_score_pd = pd.DataFrame.from_dict(sil_score,orient='index')
sil_score_pd.columns = ['SIL']
clust_pd = pd.DataFrame.from_dict(clust,orient='index')
clust_pd.columns = ['CLUSTER']
sil_clust = pd.concat([sil_score_pd,clust_pd],axis=0)


# In[93]:


sil_clust


# In[94]:


df_main = df_main.drop(columns = columns)
scorecard = scorecard.drop(columns = ['cluster','label'])


# In[95]:


scorecard.total.min()


# In[96]:


scorecard.total.max()


# In[97]:


scorecard.head()


# In[98]:


scorecard.shape


# In[99]:


cluster_total_label


# In[100]:


labels


# In[101]:


scorecard=pd.concat([Final,scorecard,df_main],axis = 1)


# In[102]:


scorecard.head(10)


# In[103]:


scorecard.columns


# In[106]:


scorecard[scorecard['%AGE ACHIEVED (DEPOSIT GROWTH)  AVERAGE']==1308.6651053333333]


# In[ ]:




