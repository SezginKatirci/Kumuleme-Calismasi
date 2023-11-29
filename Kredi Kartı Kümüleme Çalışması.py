#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sbn


# In[2]:


veriler=pd.read_csv("C:\\Users\\Dell\\Desktop\\Credit Card Dataset for Clustering\\CC GENERAL.csv")


# In[3]:


veriler.head()


# In[4]:


veriler.tail()


# In[5]:


veriler.isnull().sum()


# In[6]:


veriler.info()


# In[7]:


veriler.drop(["CUST_ID"],axis=1,inplace=True)
veriler.dropna(inplace=True)
# İd satırını siliyoruz.
# Null veriler toplam verimizin %1'i kadar oluşturduğu için verimizden siliyoruz.


# In[8]:


veriler["index"]=range(0,8636)
veriler.set_index("index",inplace=True)
# Null değerleri sildiğimiz için verimizin index satırında bozulmalar oluştu. İlerleyen aşamada sorun olmaması için 
# yeni index sütunu ekliyoruz.


# In[9]:


veriler


# In[10]:


yeniSutunIsimleri=["Bakiye","Bakiye Güncelleme Sıklığı","Satın Alma","Tek Seferde S_Alma","Taksitli S_Alma","Peşin Nakit",
                  "S_Alma Sıklığı","Tek Seferde S_Alma Sık.","Taksitli S_Alma Sık.","N_Avansın Ne Sık. Ödedi",
                   "Vadeli Nakit Ödeme","S_Alma İşlem Sayısı","K_Kartı Limiti","Ödemeler","Min. Ödeme Tutarı",
                   "Tam Ödeme Yüzdesi","K_Kartı Süresi"]
veriler.columns=yeniSutunIsimleri
#İngilizce kolon isimleri yerine türkçe isimleri ekledim.


# In[11]:


veriler.describe()


# In[12]:


veriler.corr()


# In[13]:


sbn.heatmap(veriler.corr(),cmap=sbn.color_palette("Spectral", as_cmap=True))
#sns.color_palette("flare", as_cmap=True) - cmap="crest"


# In[14]:


sbn.pairplot(veriler)


# # Küme Sayısının Belirlenmesi ve Gruplara Ayrılma İşlemi

# In[15]:


# küme sayısını bulmak için hem k-means algoritması hemde Agglomerative kümüleme algoritması ile elbow metodu denenmiştir.
from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer

kmeans=KMeans()
elbow=KElbowVisualizer(kmeans,k=(2,20))
elbow.fit(veriler)
elbow.show()


# In[16]:


from sklearn.cluster import KMeans,AgglomerativeClustering
from yellowbrick.cluster import KElbowVisualizer

kmeans=AgglomerativeClustering()
elbow=KElbowVisualizer(kmeans,k=(2,20))
elbow.fit(veriler)
elbow.show()


# In[17]:


# her iki algoritma ile küme sayısı 8 olarak belirlenmiştir.
from sklearn.cluster import KMeans, AgglomerativeClustering

ac=AgglomerativeClustering(n_clusters=8)
clusters = ac.fit_predict(veriler)

veriler["Gruplar"]=clusters
veriler.head(10)


# In[18]:


veriler.info()


# # Sınıfların Özeti

# In[19]:


gruplar=veriler.groupby("Gruplar")
grup1=gruplar.get_group(0)
grup2=gruplar.get_group(1)
grup3=gruplar.get_group(2)
grup4=gruplar.get_group(3)
grup5=gruplar.get_group(4)
grup6=gruplar.get_group(5)
grup7=gruplar.get_group(6)
grup8=gruplar.get_group(7)


# In[20]:


sbn.countplot(veriler,x="Gruplar")
print(veriler["Gruplar"].value_counts())
liste=[]
liste2=["Grup 1","Grup 2","Grup 3","Grup 4","Grup 5","Grup 6","Grup 7","Grup 8"]
explode = (0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3)
liste.append(grup1["Bakiye"].count())
liste.append(grup2["Bakiye"].count())
liste.append(grup3["Bakiye"].count())
liste.append(grup4["Bakiye"].count())
liste.append(grup5["Bakiye"].count())
liste.append(grup6["Bakiye"].count())
liste.append(grup7["Bakiye"].count())
liste.append(grup8["Bakiye"].count())
fig, ax = plt.subplots()
fig.suptitle("Toplam Kişi Oranı")
ax.pie(liste, explode=explode, labels=liste2, autopct='%1.1f%%', shadow=True, startangle=30)
plt.show()


# # Sınıfların karşılaştırılması
1. Karşılaştırma için fonksiyonların oluşturulması
# In[21]:


def tabloOlustur(sutun): 
      
    kolonlar=["Grup 1","Grup 2","Grup 3","Grup 4","Grup 5","Grup 6","Grup 7","Grup 8"]
    tablo2=pd.DataFrame(index=["Toplam Üye Sayısı","Toplam","Ortalama","Standart Sapma","Min. Değer","Mak. Değer",
                               "Toplam Üye Sayısı %"],columns=kolonlar)         
    for i in range(0,8):
        g=0
        liste=[]
        while g<len(veriler):
            if veriler["Gruplar"][g]==i:
                  liste.append(veriler[sutun][g])
            g=g+1
        liste=np.array(liste)
        toplam=liste.sum()        
        say=len(liste)
        ort=liste.mean()
        stds=liste.std()
        mind=liste.min()
        mak=liste.max()
        tablo2.iloc[0:1,i:i+1]=say
        tablo2.iloc[1:2,i:i+1]=int(toplam)
        tablo2.iloc[2:3,i:i+1]=int(ort)
        tablo2.iloc[3:4,i:i+1]=int(stds)
        tablo2.iloc[4:5,i:i+1]=int(mind)
        tablo2.iloc[5:6,i:i+1]=int(mak)
        tablo2.iloc[6:7,i:i+1]=say/len(veriler)
    return tablo2


# In[22]:


def GrafikOlustur(sutun_adi,islem,explode = (0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2),startangle=10):
    liste=[]
    liste2=["Grup 1","Grup 2","Grup 3","Grup 4","Grup 5","Grup 6","Grup 7","Grup 8"]
    if islem=="ortalama":
        liste.append(grup1[sutun_adi].mean())
        liste.append(grup2[sutun_adi].mean())
        liste.append(grup3[sutun_adi].mean())
        liste.append(grup4[sutun_adi].mean())
        liste.append(grup5[sutun_adi].mean())
        liste.append(grup6[sutun_adi].mean())
        liste.append(grup7[sutun_adi].mean())
        liste.append(grup8[sutun_adi].mean())
        fig, ax = plt.subplots()
        fig.suptitle(f"{sutun_adi} Ortalama Dağılımı")
        ax.pie(liste, explode=explode, labels=liste2, autopct='%1.1f%%', shadow=True, startangle=startangle)
        plt.show()
    if islem=="topla":
        liste.append(grup1[sutun_adi].sum())
        liste.append(grup2[sutun_adi].sum())
        liste.append(grup3[sutun_adi].sum())
        liste.append(grup4[sutun_adi].sum())
        liste.append(grup5[sutun_adi].sum())
        liste.append(grup6[sutun_adi].sum())
        liste.append(grup7[sutun_adi].sum())
        liste.append(grup8[sutun_adi].sum())
        fig, ax = plt.subplots()
        fig.suptitle(f"Toplam {sutun_adi} Dağılımı")
        ax.pie(liste, explode=explode, labels=liste2, autopct='%1.1f%%', shadow=True, startangle=startangle)
        plt.show()

2. Grafikler ve özet tablolar
# In[23]:


tablo=tabloOlustur("Bakiye")
grafik1=GrafikOlustur("Bakiye","topla")
grafik2=GrafikOlustur("Bakiye","ortalama")
tablo


# In[24]:


tablo=tabloOlustur("Bakiye Güncelleme Sıklığı")
grafik1=GrafikOlustur("Bakiye Güncelleme Sıklığı","topla")
grafik2=GrafikOlustur("Bakiye Güncelleme Sıklığı","ortalama")
tablo


# In[25]:


tablo=tabloOlustur("Satın Alma")
grafik1=GrafikOlustur("Satın Alma","topla")
grafik2=GrafikOlustur("Satın Alma","ortalama")
tablo


# In[26]:


tablo=tabloOlustur("Tek Seferde S_Alma")
grafik1=GrafikOlustur("Tek Seferde S_Alma","topla")
grafik2=GrafikOlustur("Tek Seferde S_Alma","ortalama")
tablo


# In[27]:


tablo=tabloOlustur("Taksitli S_Alma")
grafik1=GrafikOlustur("Taksitli S_Alma","topla")
grafik2=GrafikOlustur("Taksitli S_Alma","ortalama")
tablo


# In[28]:


tablo=tabloOlustur("Peşin Nakit")
grafik1=GrafikOlustur("Peşin Nakit","topla")
grafik2=GrafikOlustur("Peşin Nakit","ortalama")
tablo


# In[29]:


tablo=tabloOlustur("S_Alma Sıklığı")
grafik1=GrafikOlustur("S_Alma Sıklığı","topla")
grafik2=GrafikOlustur("S_Alma Sıklığı","ortalama")
tablo


# In[30]:


tablo=tabloOlustur("Tek Seferde S_Alma Sık.")
grafik1=GrafikOlustur("Tek Seferde S_Alma Sık.","topla")
grafik2=GrafikOlustur("Tek Seferde S_Alma Sık.","ortalama")
tablo


# In[31]:


tablo=tabloOlustur("Taksitli S_Alma Sık.")
grafik1=GrafikOlustur("Taksitli S_Alma Sık.","topla")
grafik2=GrafikOlustur("Taksitli S_Alma Sık.","ortalama")
tablo


# In[32]:


tablo=tabloOlustur("N_Avansın Ne Sık. Ödedi")
grafik1=GrafikOlustur("N_Avansın Ne Sık. Ödedi","topla")
grafik2=GrafikOlustur("N_Avansın Ne Sık. Ödedi","ortalama")
tablo


# In[33]:


tablo=tabloOlustur("Vadeli Nakit Ödeme")
grafik1=GrafikOlustur("Vadeli Nakit Ödeme","topla")
grafik2=GrafikOlustur("Vadeli Nakit Ödeme","ortalama")
tablo


# In[34]:


tablo=tabloOlustur("S_Alma İşlem Sayısı")
grafik1=GrafikOlustur("S_Alma İşlem Sayısı","topla")
grafik2=GrafikOlustur("S_Alma İşlem Sayısı","ortalama")
tablo


# In[35]:


tablo=tabloOlustur("K_Kartı Limiti")
grafik1=GrafikOlustur("K_Kartı Limiti","topla")
grafik2=GrafikOlustur("K_Kartı Limiti","ortalama")
tablo


# In[36]:


tablo=tabloOlustur("Ödemeler")
grafik1=GrafikOlustur("Ödemeler","topla")
grafik2=GrafikOlustur("Ödemeler","ortalama")
tablo


# In[37]:


tablo=tabloOlustur("Min. Ödeme Tutarı")
grafik1=GrafikOlustur("Min. Ödeme Tutarı","topla")
grafik2=GrafikOlustur("Min. Ödeme Tutarı","ortalama")
tablo


# In[38]:


tablo=tabloOlustur("Tam Ödeme Yüzdesi")
grafik1=GrafikOlustur("Tam Ödeme Yüzdesi","topla")
grafik2=GrafikOlustur("Tam Ödeme Yüzdesi","ortalama")
tablo


# In[39]:


tablo=tabloOlustur("K_Kartı Süresi")
grafik1=GrafikOlustur("K_Kartı Süresi","topla")
grafik2=GrafikOlustur("K_Kartı Süresi","ortalama")
tablo


# # Grup özeliklerinin açıklanması
Grup 1 
Satın alma işleminin en düşük yapıldığı, kredi karti harcamalarının minimum tutarının ödediği bakaların kara listelerini
oluşturabilecek grup.Grup 2
1. gruba benzerlik gösteren ama satın alma minimum ödeme tutarlarında daha iyi olarak gözlenmiştir.Grup 3
Taksitli satın almayı az, genelde peşin alma yapan ve nakit avans kullanmaktan çekinmeyin grup.Grup 4
Hem taksitli hem de tek çekim satın alma işleminin yüksek olduğu, ödemelerin çoğunluğunun tamamının yapıldığı gözlenmiştir.Grup 5
Satın alma işlemlerinin en az gözlemlendiği, ödemelerin tamamının yapıldığı gözlenmiştir. Bu gruba kredi kartı kullanmayı sevmeyen grup diyebiliriz.Grup 6
Satın alma işlemlerinin ve ödemelerin en yüksek gözlendiği, kredi kartı limitinin en yüksek olduğu gruptur. Bankanın 1. sınıf müşterilerini oluşturur.Grup 7
Peşin alışverişin en az olduğu gözlenen gruptur.Grup 8
Peşin alışverişin en çok yapıldığı, nakit avansında kullanıldığı ve avansın en çok geri ödendiği izlenmiştir. Ayrıca kredi kartı limitinin yüksek olduğu da izlenmiştir.
# # Sonuç
1. ve 2. grupların, 3. ve 8 grupların birbirine benzer özellikler gösterdiği izlenmiştir. Bu gruplar 
birleştirebilir ve toplam grup sayısı 6 e düşürebilir.