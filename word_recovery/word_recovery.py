import numpy as np

class WordRecovery:
	
	def __init__(self, similarity, vocabulary):
		self.similarity = similarity
		self.vocabulary = vocabulary
	
	def __call__(self, word):
		mapsims = map(lambda w: self.similarity(word, w), self.vocabulary)
		npsims = np.array(list(mapsims))
		return self.vocabulary[npsims.argmax()]
