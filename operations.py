import numpy as np


# (x_1, ..., x_n) -> { (x_i, x_j) : 1 <= i < j <= n }
def set_of_orders(word):
	n = len(word)
	return { (word[i], word[j]) for i in range(n - 1) for j in range(i + 1, n) }


# (x_1, ..., x_n) -> { x_i : 1 <= i <= n }
class SetOfLetters:
	
	def __init__(self):
		self.cache = dict()
	
	def __call__(self, word):
		if word not in self.cache:
			self.cache[word] = set(word)
		return self.cache[word]


# (x_1, ..., x_n) -> { (x_i, x_j) : 1 <= i < j <= n }
class SetOfOrders:
	
	def __init__(self):
		self.cache = dict()
	
	def __call__(self, word):
		if word not in self.cache:
			n = len(word)
			self.cache[word] = { (word[i], word[j]) for i in range(n - 1) for j in range(i + 1, n) }
		return self.cache[word]


# (A, B) -> measure(A & B) / measure(A | B)
def andor_similarity(A, B):
	
	emptyA = (len(A) == 0)
	emptyB = (len(B) == 0)
	
	if emptyA or emptyB:
		return float(emptyA and emptyB)
	
	return len(A & B) / len(A | B)


# (A, B) -> measure(A & B) / sqrt(measure(A cross B))
def Ochiai_coefficient(A, B):
	
	lenA, lenB = len(A), len(B)
	
	emptyA = (lenA == 0)
	emptyB = (lenB == 0)
	
	if emptyA or emptyB:
		return np.float64(emptyA and emptyB)
	
	return len(A & B) / np.sqrt(lenA * lenB)


def first(*x):
	return x[0]


def second(*x):
	return x[1]


def last(*x):
	return x[-1]


def mean(*x):
	return np.array(x).mean()


# k -> ((x_1, ..., x_k, ..., x_n) -> x_k)
class Member:
	
	def __init__(self, index):
		self.index = index
	
	def __call__(self, *x):
		return x[self.index]


# (w_1, ..., w_n) -> ((x_1, ..., x_n) -> (w_1 x_1 + ... + w_n x_n) / (w_1 + ... + w_n))
class WeightedMean:
	
	def __init__(self, weights):
		self.weights = np.array(weights)
	
	def __call__(self, *x):
		return (self.weights @ np.array(x)) / self.weights.sum()


# (f, g) -> ((x_1, ..., x_n) -> f(g(x_1), ..., g(x_n)))
class Composition:
	
	def __init__(self, outer, inner):
		self.outer = outer
		self.inner = inner
	
	def __call__(self, *x):
		return self.outer(*map(self.inner, x))


# (f, (g_1, ..., g_n)) -> (x -> f(g_1(x), ..., g_n(x)))
class Ensemble:
	
	def __init__(self, parents, crossover):
		self.parents = parents
		self.crossover = crossover
	
	def __call__(self, *x):
		return self.crossover(*[parent(*x) for parent in self.parents])


# (f, (g_1, ..., g_n)) -> ((x_1, ..., x_n) -> f(g_1(x_1), ..., g_n(x_n)))
class Compound:
	
	def __init__(self, outer, inners):
		self.outer = outer
		self.inners = inners
	
	def __call__(self, *x):
		return self.outer(*[f_i(x_i) for f_i, x_i in zip(self.inners, x)])


# f -> (x -> if f(x) then 1 else 0)
# where
#     f: X -> Booleans
class Indicator:
	
	def __init__(self, bool_fun):
		self.bool_fun = bool_fun
	
	def __call__(self, *x):
		return int(self.bool_fun(*x))


# (f, g) -> (x -> if f(x) then 1 else g(x))
# where
#     f: X -> Booleans
class IndicatorWithAlternative:
	
	def __init__(self, bool_fun, alt_fun):
		self.bool_fun = bool_fun
		self.alt_fun = alt_fun
	
	def __call__(self, *x):
		return 1 if self.bool_fun(*x) else self.alt_fun(*x)
