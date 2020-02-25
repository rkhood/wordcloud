import pandas as pd

import string
import wordcloud

import matplotlib.pyplot as plt

from nltk.corpus import stopwords


def read_data(f='progressive-tweet-sentiment.csv'):
	return pd.read_csv(
		f,
		usecols=(5,14,15,16),
		names=['results', 'topics', 'tweet','id'],
		header=0,
		index_col=3,
		)


def top_words(data, n=50):
	stop = set(stopwords.words('english'))
	return data.tweet.str.lower() \
			 	     .str.replace('[{}]'.format(string.punctuation), '') \
			         .apply(lambda x: ' '.join([
			         	item for item in x.split() if item not in stop])) \
			         .str.split(expand=True) \
			         .stack() \
			         .value_counts()[:n] \
			         .to_dict()


def grab_data(data, topic, result):
	return data.query('topics == @topic and results == @result')


def make_wordcloud(data, topic, result):
	data_subset = grab_data(data, topic, result)
	freq = top_words(data_subset)

	cloud = wordcloud.WordCloud(
        background_color='white',
        width=1200,
        height=1200,
	)
	wc = cloud.generate_from_frequencies(freq)

	plt.imshow(wc, interpolation='bilinear')
	plt.axis('off')
	plt.savefig('imgs/word_cloud_{}_{}.png'.format(
		topic.split()[0], result.split(':')[0]), dpi=300)


if __name__ == '__main__':

	df = read_data()
	for topic in set(df.topics):
		for result in set(df.results):
			print(topic, result.split(':')[0])
			make_wordcloud(df, topic, result)





