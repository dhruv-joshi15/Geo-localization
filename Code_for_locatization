!pip install nltk

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

files=["data/datajson/geoLondonSep2022_1.json",
       "data/datajson/geoLondonSep2022_2.json",
       "data/datajson/geoLondonSep2022_3.json", 
       "data/datajson/geoLondonSep2022_4.json", 
       "data/datajson/geoLondonSep2022_5.json"]
 
def merge_JsonFiles(filename):
    result = list()
    for f1 in filename:
        with open(f1, 'r', encoding='utf-8') as infile:
            result.extend(json.load(infile)) 
    with open('Merged_Json.json', 'w') as output_file:
        json.dump(result, output_file)
 
merge_JsonFiles(files)
 
tweet_df = pd.read_json('Merged_Json.json')
tweets_dfs = pd.read_json('Merged_Json.json')
tweets_dfs

# Distance calculation between two points
def compute_distance(location_1, location_2):
    long_1, lat_1, long_2, lat_2 = map(np.radians, [location_1[0], location_1[1], location_2[0], location_2[1]]) 
    haversine = np.sin((lat_2 - lat_1) / 2) ** 2 + np.cos(lat_1) * np.cos(lat_2) * np.sin((long_2 - long_1) / 2) ** 2  
    return 6371 * 2 * np.arcsin(np.sqrt(haversine))

london_coordinates = [-0.563, 51.2C61318, 0.28036, 51.686031]

# For generating grid details
rows = np.ceil(compute_distance([london_coordinates[0], london_coordinates[1]], [london_coordinates[0], london_coordinates[3]])).astype(int)
print('Number of rows:' , rows)

columns = np.ceil(compute_distance([london_coordinates[0], london_coordinates[1]], [london_coordinates[2], london_coordinates[1]])).astype(int)
print('Number of columns: ',columns)

noofgrids = int(rows * columns)
print('Number of grids: ',noofgrids)

# Plotting the grid 
grids = np.zeros((rows, columns), dtype = int)

for coordinate in tweets_dfs['coordinates']:
    row_index = np.ceil(compute_distance([london_coordinates[0], london_coordinates[1]], [london_coordinates[0], coordinate[1]])).astype(int)
    col_index = np.ceil(compute_distance([london_coordinates[0], london_coordinates[1]], [coordinate[0], london_coordinates[1]])).astype(int)
    grids[row_index, col_index] += 1
    

no_of_tweets = np.ravel(grids)
grid_col = 'Grid'
no_of_tweets_count = 'Tweets count'
no_of_tweets_dfs = pd.DataFrame({grid_col: np.arange(1, noofgrids + 1), no_of_tweets_count: no_of_tweets})  # Create a DataFrame containing the number of tweets per grid.


# Visualize the distribution using a histogram
plt.figure(figsize=(12, 6))
plt.hist(no_of_tweets, bins = np.max(no_of_tweets), log = True, color='black') #, edgecolor='black')
plt.title('Tweets Distribution - 1km x 1km Grid')
plt.xlabel('Number of Tweets')
plt.ylabel('Number of Grid Cells')
plt.ylim(0, 50)
plt.xlim(0, 800)
plt.show()

# Create heatmap of tweet distribution
plt.figure(figsize=(8, 6))

grids_map = np.log(grids, out = np.zeros_like(grids, dtype = float), where = (grids != 0))
sns.heatmap(grids_map, cmap='cividis') 
plt.title('Heatmap of Tweet Distribution in London ')
plt.xlabel('Longitude (in log)')
plt.ylabel('Latitude (in log)')
plt.show()

# Compute statistics Provide statistics of the data (total tweets, how many are on the cells, and how it is distributed etc.)
# and interpret the statistics – what does this mean?

total_no_of_tweets = len(tweets_dfs)
tweets_per_cell = grids.flatten()
per_cell_mean = np.mean(tweets_per_cell)
per_cell_median = np.median(tweets_per_cell)
per_cell_max = np.max(tweets_per_cell)
per_cell_min = np.min(tweets_per_cell)

print("Total number of tweets:", total_no_of_tweets)
print("Maximum number of tweets in a grid cell:", per_cell_max)
print("Minimum number of tweets in a grid cell:", per_cell_min)
print("Average number of tweets per grid cell:", per_cell_mean)
print("Median number of tweets per grid cell:", per_cell_median)


import pandas as pd
from collections import Counter

# Load the tweet data as Pandas DataFrames
background_df = pd.read_json('data/credModelFiles/bgQuality.json', lines=True)
high_quality_df = pd.read_json('data/credModelFiles/highQuality.json', lines=True)
low_quality_df = pd.read_json('data/credModelFiles/lowQuality.json', lines=True)


# Calculate term frequencies and document frequencies
def calculate_term_document_frequencies(tweets_df):
    frequency_terms = Counter()
    frequency_doc = Counter()

    for index, row in tweets_df.iterrows():
        text = ' '.join(row['text'])
        terms = text.split() 
        unique_terms = set(terms)

        # Increment term frequencies for each term in the tweet
        frequency_terms.update(terms)

        # Increment document frequencies for each unique term in the tweet
        frequency_doc.update(unique_terms)

    return frequency_terms, frequency_doc

# Calculate term frequencies and document frequencies for each dataset
HighQ_term_frequency, HighQ_doc_frequency = calculate_term_document_frequencies(high_quality_df)
LowQ_term_frequency, LowQ_doc_frequency = calculate_term_document_frequencies(low_quality_df)
BckgrndQ_term_frequency, BckgrndQ_Doc_frequency = calculate_term_document_frequencies(background_df)

# Calculate likelihood ratios
def calculate_likelihood(tf_model, F_model, tf_bg, F_bg):
    tf_model_sum = sum(tf_model.values())
    F_model_sum = sum(F_model.values())
    tf_bg_sum = sum(tf_bg.values())
    F_bg_sum = sum(F_bg.values())
    return (tf_model_sum / F_model_sum) / (tf_bg_sum / F_bg_sum)

# Define thresholds 
HighQ_threshold = 2.0
LowQ_threshold = 2.0

# Calculate likelihood ratios
RHQ = calculate_likelihood(HighQ_term_frequency, HighQ_doc_frequency, BckgrndQ_term_frequency, BckgrndQ_Doc_frequency)
RLQ = calculate_likelihood(LowQ_term_frequency, LowQ_doc_frequency, BckgrndQ_term_frequency, BckgrndQ_Doc_frequency)

# Define newsworthy scores based on likelihood ratios and thresholds
SHQ = {term: RHQ if RHQ >= HighQ_threshold else 0 for term in HighQ_term_frequency}
SLQ = {term: RLQ if RLQ >= LowQ_threshold else 0 for term in LowQ_term_frequency}

# Calculate newsworthy scores for each tweet
def calculate_newsworthy_score(text, quality_score, scores):
    terms = text.split()
    return sum(scores.get(term, 0) * quality_score for term in terms)

high_quality_df['newsworthy_score'] = high_quality_df.apply(lambda row: calculate_newsworthy_score(row['text'], row['qualityS'], SHQ), axis=1)
low_quality_df['newsworthy_score'] = low_quality_df.apply(lambda row: calculate_newsworthy_score(row['text'], row['qualityS'], SLQ), axis=1)

import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
import nltk
nltk.download('punkt')
nltk.download('stopwords')
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer


# Functions for data preprocessing
def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    porter = PorterStemmer()
    tokens = [porter.stem(token) for token in tokens]
    return tokens

# Function to calculate newsworthiness scores
def calculate_newsworthiness_score(text, scores, threshold):
    tokens = preprocess_text(text)
    score = sum(scores.get(token, 0) for token in tokens)
    return score >= threshold

# Calculate term frequencies (tf) and document frequencies (F) for each dataset
def calculate_term_document_frequencies(tweets_df):
    frequency_term = Counter()
    frequency_doc = Counter()

    for index, row in tweets_df.iterrows():
        tokens = preprocess_text(' '.join(row['text']))
        unique_terms = set(tokens)
        frequency_term.update(tokens)
        frequency_doc.update(unique_terms)

    return frequency_term, frequency_doc

# Calculate term and document frequencies for each dataset
bckgrnd_frequency_term, bckgrnd_frequency_doc = calculate_term_document_frequencies(background_df)
highquality_frequency_term, highquality_frequency_doc = calculate_term_document_frequencies(high_quality_df)
lowquality_frequency_term, lowquality_frequency_doc = calculate_term_document_frequencies(low_quality_df)

# Calculate likelihood ratios
RLQ = {}
for term, freq in lowquality_frequency_term.items():
    bckgrnd_frequency = bckgrnd_frequency_term.get(term, 0) 
    if bckgrnd_frequency != 0:
        RLQ[term] = (freq / lowquality_frequency_doc[term]) / (bckgrnd_frequency / bckgrnd_frequency_doc[term])
    else:
        RLQ[term] = 0 
        
RHQ = {}
for term, freq in highquality_frequency_term.items():
    bckgrnd_frequency = bckgrnd_frequency_term.get(term, 0) 
    if bckgrnd_frequency != 0:
        RHQ[term] = (freq / highquality_frequency_doc[term]) / (bckgrnd_frequency / bckgrnd_frequency_doc[term])
    else:
        RHQ[term] = 0  

# Define thresholds for considering terms as newsworthy
thresholds = [0.5, 1, 2, 3, 4] 
for threshold in thresholds:
    high_quality_df['newsworthiness'] = high_quality_df['text'].apply(lambda text: calculate_newsworthiness_score(text, RHQ, threshold)).astype(int)
    low_quality_df['newsworthiness'] = low_quality_df['text'].apply(lambda text: calculate_newsworthiness_score(text, RLQ, threshold)).astype(int)

import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import Point
import nltk
nltk.download('punkt')
nltk.download('stopwords')
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from collections import Counter

# Load the geo-tagged tweet dataset
tweets_dataset = pd.read_json('Merged_Json.json')

# Function to calculate newsworthiness scores
def calculate_newsworthiness_score(text, scores, threshold):
    tokens = preprocess_text(text)
    score = sum(scores.get(token, 0) for token in tokens)
    return score >= threshold


# Functions for data preprocessing
def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    processed_text = ' '.join(tokens)
    return processed_text

tweets_dataset['processed_text'] = tweets_dataset['text'].apply(preprocess_text)

def calculate_newsworthiness(tweet_text):
    # Assuming you have defined a function to calculate newsworthiness scores
    return calculate_newsworthiness_score(tweet_text, RHQ, threshold)

# Calculate newsworthiness scores for each tweet
tweets_dataset['newsworthiness_score'] = tweets_dataset['processed_text'].apply(calculate_newsworthiness)

# Create a GeoDataFrame from the tweet data
tweets_dataset[['longitude', 'latitude']] = tweets_dataset['coordinates'].apply(lambda x: pd.Series([x[0], x[1]]))
geometry = [Point(xy) for xy in zip(tweets_dataset['longitude'], tweets_dataset['latitude'])]
geo_data = gpd.GeoDataFrame(tweets_dataset, geometry=geometry)

tweets_data = tweets_dataset.copy()

# Tweets with high scores
high_score = 0.75  # Define a threshold for high scores
high_score_tweets = tweets_data[tweets_data['newsworthiness_score'] > high_score]
print("\nNumber of Tweets with High Newsworthiness Scores:", len(high_score_tweets))
print("Example of High Score Tweets:")
print(high_score_tweets[['text', 'newsworthiness_score']].head())

# Tweets with low scores
low_score = 0.25  # Define a threshold for low scores
low_score_tweets = tweets_data[tweets_data['newsworthiness_score'] < low_score]
print("\nNumber of Tweets with Low Newsworthiness Scores:", len(low_score_tweets))
print("Example of Low Score Tweets:")
print(low_score_tweets[['text', 'newsworthiness_score']].head())



# Calculate the mean and median of newsworthiness scores
mean_value_tweets = tweets_data['newsworthiness_score'].mean()
print("Mean Newsworthiness Score:", mean_value_tweets)

median_value_tweets = tweets_data['newsworthiness_score'].median()
print("Median Newsworthiness Score:", median_value_tweets)

# Visualize the distribution of newsworthiness scores
sns.histplot(tweets_data['newsworthiness_score'], kde=True)
plt.title('Distribution of Newsworthiness Scores')
plt.axvline(x=mean_value_tweets, linestyle='-.', label='Mean of tweets', color='red')
plt.axvline(x=median_value_tweets, linestyle='-.', label='Median of tweets',color='Black')
plt.xlabel('Newsworthiness Score')
plt.ylabel('Frequency')
plt.legend()
plt.show()

# remove tweets with low newsworthy scores
threshold = tweets_data['newsworthiness_score'].mean()

# Filter out tweets with low newsworthy scores
unwanted_tweets = tweets_data[tweets_data['newsworthiness_score'] >= threshold]
unwanted_tweets

print(tweets_dataset['newsworthiness_score'].describe())

plt.figure(figsize=(8, 6))
sns.histplot(tweets_dataset['newsworthiness_score'], kde=True)
plt.title("Distribution of Newsworthiness Scores")
plt.xlabel("Newsworthiness Score")
plt.ylabel("Frequency")
plt.show()

# Set a threshold for separating high and low 
threshold = 0.5 

# Filter tweets with high and low scores
high_newsworthy_score = tweets_dataset[tweets_dataset['newsworthiness_score'] >= threshold]
low_newsworthy_score = tweets_dataset[tweets_dataset['newsworthiness_score'] < threshold]

# Justify the threshold choice
print("\nJustifying the threshold:")
print("The threshold is like a dividing line that sorts tweets into two groups – highly newsworthy and less newsworthy. This decision comes from carefully looking at how newsworthiness scores are spread out.")

# Calculate the total number of tweets
no_of_tweets = len(tweets_dataset)

# Determine the distribution of newsworthiness scores
newsworthiness = tweets_dataset['newsworthiness_score'].value_counts()

# Set a threshold 
threshold = 0.5

# Count the number of tweets above and below the threshold
high_newsworthy_tweets = tweets_dataset[tweets_dataset['newsworthiness_score'] >= threshold]
low_newsworthy_tweets = tweets_dataset[tweets_dataset['newsworthiness_score'] < threshold]

# Calculate the percentage of removed tweets
percent_of_removed_tweets = len(low_newsworthy_tweets) / no_of_tweets * 100

# Display the statistics
print("Statistics of the tweet dataset:")
print("Total number of tweets:", no_of_tweets)
print("Distribution of Newsworthiness:")
print(newsworthiness)
print("Number of High Newsworthy Tweets:", len(high_newsworthy_tweets))
print("Number of Low Newsworthy Tweets:", len(low_newsworthy_tweets))
print("Percentage of Removed Tweets:", percent_of_removed_tweets)

import matplotlib.pyplot as plt
import seaborn as sns

# Filter newsworthy tweets
newsworthy_tweets = tweets_dataset[tweets_dataset['newsworthiness_score'] > 0.5]

newsworthy_tweets[['longitude', 'latitude']] = newsworthy_tweets['coordinates'].apply(lambda x: pd.Series([x[0], x[1]]))

london_coordinates = [-0.563, 51.261318, 0.28036, 51.686031]

# For generating grid details
rows = np.ceil(compute_distance([london_coordinates[0], london_coordinates[1]], [london_coordinates[0], london_coordinates[3]])).astype(int)
print('Number of rows:' , rows)

columns = np.ceil(compute_distance([london_coordinates[0], london_coordinates[1]], [london_coordinates[2], london_coordinates[1]])).astype(int)
print('Number of columns: ',columns)

noofgrids = int(rows * columns)
print('Number of grids: ',noofgrids)

# Plotting the grid 
grids = np.zeros((rows, columns), dtype = int)

for coordinate in tweets_dfs['coordinates']:
    row_index = np.ceil(compute_distance([london_coordinates[0], london_coordinates[1]], [london_coordinates[0], coordinate[1]])).astype(int)
    col_index = np.ceil(compute_distance([london_coordinates[0], london_coordinates[1]], [coordinate[0], london_coordinates[1]])).astype(int)
    grids[row_index, col_index] += 1
    
no_of_tweets = np.ravel(grids)
grid_col = 'Grid'
no_of_tweets_count = 'Tweets count'
no_of_tweets_dfs = pd.DataFrame({grid_col: np.arange(1, noofgrids + 1), no_of_tweets_count: no_of_tweets}) 

# Visualize the distribution using a histogram
plt.figure(figsize=(12, 6))
plt.hist(no_of_tweets, bins = np.max(no_of_tweets), log = True, color='black')
plt.title('Tweet Distribution:  1km x 1km Grid Cells')
plt.xlabel('Number of Tweets')
plt.ylabel('Number of Grid Cells')
plt.ylim(0, 50)
plt.xlim(0, 800)
plt.show()

# Create heatmap of tweet distribution
plt.figure(figsize=(10, 8))
grids_map = np.log(grids, out = np.zeros_like(grids, dtype = float), where = (grids != 0)) 
sns.heatmap(grids_map, cbar_kws={'label': 'Number of Tweets'},cmap='cividis')
plt.title('Heatmap of Newsworthy Tweets Distribution in London (1km x 1km Grid)')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.show()

