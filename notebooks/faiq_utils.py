import re
from typing import List, Tuple

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from bs4 import BeautifulSoup
from wordcloud import WordCloud, STOPWORDS
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer

def stem_text(
        raw_text: str,
        stopwords_list: List[str] = [],
    ) -> str:
    """
    This is a function to convert raw text to a string of meaningful words.

    ### Arguments
    - `raw_text`: The input text to pre-process.
    - `stopwords_list`: A list of stopwords to be added in the stopwords list.

    ### Returns
    A pre-processed string.
    """
  
    # Remove HTML tags
    review_text = BeautifulSoup(raw_text, "html.parser").get_text()
    
    # Remove non-letters
    letters_only = re.sub("[^a-zA-Z]", " ", review_text)
    
    # Convert words to lower case and split each word up
    words = letters_only.lower().split()
    
    # Searching through a set is faster than searching through a list 
    # Hence, we will convert stopwords to a set
    stops = set(stopwords.words('english'))
    
    # Adding on stopwords that were appearing frequently in both positive and negative reviews 
    stops.update(stopwords_list)
    
    # Remove stopwords
    meaningful_words = [w for w in words if w not in stops]
        
    # Instantiate PorterStemmer
    p_stemmer = PorterStemmer()
    
    # Stem words
    meaningful_words = [p_stemmer.stem(w) for w in meaningful_words]        
   
    # Join words back into one string, with a space in between each word
    return(" ".join(meaningful_words))

def barplot_cvec(
        df: pd.DataFrame,
        titles: List[str],
        color: str,
        xlimit: Tuple[int, int],
        category: str = None,
    ):
    """
    Plots top 20 uni-grams, bi-grams & tri-grams for various categories.
    
    ### Arguments
    - `df`: The input dataframe
    - `category`: The transaction category.
    - `title`: A list of titles 
    - `color`: Color palette to use for charts.
    - `xlimit`: Horizontal axis limits
    """
    df = df.copy() # best practice to ensure we don't alter the original dataframe
    
    if category != None:
        words_series = df[df['category']==category]['description_stem']
    else:
        words_series = df['description_stem']
    
    fig, ax = plt.subplots(1, 3, figsize=(25,12))
    
    ngram = [(1,1),(2,2),(3,3)] # The ngrams that we would like to plot
    
    for i in range(3):
            
        # Use CountVectorizer to tokenize the text, 
        cvec = CountVectorizer(stop_words='english', ngram_range=ngram[i])

        # Save the tokens in a dataframe
        cvec_df = pd.DataFrame(cvec.fit_transform(words_series).todense(), columns=cvec.get_feature_names_out())
        sum_words = cvec_df.sum(axis=0) # Sum up the no. of occurences for each word
        top_words = sum_words.sort_values(ascending=False).head(20)
        top_words.sort_values(ascending=True).plot(kind='barh', color=color, ax=ax[i])

        # Adjust plot aesthetics
        ax[i].set_title(titles[i], size=20, weight='bold')
        ax[i].set_xlabel('Count', size=20)
        ax[i].set_xlim(xlimit) # Setting a limit so that the barplots are comparable
        ax[i].tick_params(axis='both', which='major', labelsize=20)
        ax[i].tick_params(axis='both', which='minor', labelsize=20)

    plt.tight_layout()

def plot_wordcloud(
        category: str,
        title: str,
        df: pd.DataFrame,
        text_col_in_df: str,
        max_words: int = 50,
    ):
    """
    This is a function to plot a wordcloud of the most frequently occurring 
    words based on the transaction category.

    ### Arguments
    - `category`: The transaction category.
    - `title`: The title of the WordCloud chart generated.
    - `df`: The input dataframe. It must have a `content` column.
    - `text_col_in_df`: The text column in the input dataframe.
    - `max_words`: The maximum no. of words in the wordcloud.
    """
    df = df.copy() # best practice to ensure we don't alter the original dataframe
    
    text = df[df['category']==category][text_col_in_df] 

    wordcloud = WordCloud(width=2000, 
                          height=1000, 
                          background_color='white', 
                          max_words=max_words,
                          stopwords=STOPWORDS
                ).generate(' '.join(text)) # Remove stopwords
    
    plt.figure(figsize=(10,8))
    plt.title(title, fontsize=15, weight='bold')
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off') # Removes the axis
    plt.tight_layout()

    df_wc = pd.DataFrame(wordcloud.words_.items(), columns=['Word', 'Frequency'])
    plt.figure(figsize=(12,6))
    tmp = df_wc.head(20)
    sns.barplot(x=tmp['Frequency'], y=tmp['Word'])
    plt.show()

def make_confusion_matrix(
        cf,
        group_names: List[str] = None,
        categories: List[str] = 'auto',
        count: bool =True,
        percent: bool =True,
        cbar: bool =True,
        xyticks: bool =True,
        xyplotlabels: bool =True,
        sum_stats: bool = True,
        figsize: Tuple[int, int] = None,
        cmap: str = 'Blues',
        title: str = None
    ):
    """
    This function will make a pretty plot of an sklearn Confusion Matrix cm using a Seaborn heatmap visualization.
    
    ### Arguments
    - `cf`:            Confusion matrix to be passed in
    - `group_names`:   List of strings that represent the labels row by row to be shown in each square.
    - `categories`:    List of strings containing the categories to be displayed on the x,y axis. Default is 'auto'
    - `count`:         If True, show the raw number in the confusion matrix. Default is True.
    - `normalize`:     If True, show the proportions for each category. Default is True.
    - `cbar`:          If True, show the color bar. The cbar values are based off the values in the confusion matrix. 
    Default is True.
    - `xyticks`:       If True, show x and y ticks. Default is True.
    - `xyplotlabels`:  If True, show 'True Label' and 'Predicted Label' on the figure. Default is True.
    - `sum_stats`:     If True, display summary statistics below the figure. Default is True.
    - `figsize`:       Tuple representing the figure size. Default will be the matplotlib rcParams value.
    - `cmap`:          Colormap of the values displayed from matplotlib.pyplot.cm. Default is 'Blues'
    See http://matplotlib.org/examples/color/colormaps_reference.html                   
    - `title`:         Title for the heatmap. Default is None.
    """

    # CODE TO GENERATE TEXT INSIDE EACH SQUARE
    blanks = ['' for i in range(cf.size)]

    if group_names and len(group_names)==cf.size:
        group_labels = ["{}\n".format(value) for value in group_names]
    else:
        group_labels = blanks

    if count:
        group_counts = ["{0:0.0f}\n".format(value) for value in cf.flatten()]
    else:
        group_counts = blanks

    if percent:
        group_percentages = ["{0:.2%}".format(value) for value in cf.flatten()/np.sum(cf)]
    else:
        group_percentages = blanks

    box_labels = [f"{v1}{v2}{v3}".strip() for v1, v2, v3 in zip(group_labels,group_counts,group_percentages)]
    box_labels = np.asarray(box_labels).reshape(cf.shape[0],cf.shape[1])


    # CODE TO GENERATE SUMMARY STATISTICS & TEXT FOR SUMMARY STATS
    if sum_stats:
        #Accuracy is sum of diagonal divided by total observations
        accuracy  = np.trace(cf) / float(np.sum(cf))

        #if it is a binary confusion matrix, show some more stats
        if len(cf)==2:
            #Metrics for Binary Confusion Matrices
            precision = cf[1,1] / sum(cf[:,1])
            recall    = cf[1,1] / sum(cf[1,:])
            f1_score  = 2*precision*recall / (precision + recall)
            stats_text = "\n\nAccuracy={:0.3f}\nPrecision={:0.3f}\nRecall={:0.3f}\nF1 Score={:0.3f}".format(
                accuracy,precision,recall,f1_score)
        else:
            stats_text = "\n\nAccuracy={:0.3f}".format(accuracy)
    else:
        stats_text = ""


    # SET FIGURE PARAMETERS ACCORDING TO OTHER ARGUMENTS
    if figsize==None:
        #Get default figure size if not set
        figsize = plt.rcParams.get('figure.figsize')

    if xyticks==False:
        #Do not show categories if xyticks is False
        categories=False


    # MAKE THE HEATMAP VISUALIZATION
    plt.figure(figsize=figsize)
    sns.heatmap(cf,annot=box_labels,fmt="",cmap=cmap,cbar=cbar,xticklabels=categories,yticklabels=categories)

    if xyplotlabels:
        plt.ylabel('True label')
        plt.xlabel('Predicted label' + stats_text)
    else:
        plt.xlabel(stats_text)
    
    if title:
        plt.title(title)