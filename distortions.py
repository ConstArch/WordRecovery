import random


alphabet = 'абвгдеёжзийклмнопрстуфхцчшщъыьэюя'


def rand_add(word):
	
	li = list(word)
	i = random.randint(0, len(word))
	c = random.choice(alphabet)
	
	if i == len(word):
		li.append(c)
	else:
		li.insert(i, c)
	
	return ''.join(li)


def rand_delete(word):
	
	li = list(word)
	li.pop(random.randint(0, len(word) - 1))
	
	return ''.join(li)


def rand_replace(word):
	
	li = list(word)
	i = random.randint(0, len(word) - 1)
	li[i] = random.choice(alphabet)
	
	return ''.join(li)


def rand_swap(word):
	
	#if len(word) <= 1:
	#	return word
	
	li = list(word)
	i, j = random.sample(range(len(word)), 2)
	li[i], li[j] = li[j], li[i]
	
	return ''.join(li)
