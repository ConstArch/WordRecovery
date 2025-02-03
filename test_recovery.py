import time
import numpy as np

import operations as op

import word_recovery.distortions as dt
import word_recovery.word_recovery as wr


class CheckAndAlert:
	
	def __init__(self, alert):
		self.alert = alert
	
	def __call__(self, x):
		if x[0] == x[2]:
			return 1
		else:
			self.alert(x)
			return 0


class ListMapper:
	
	def __init__(self, fun):
		self.fun = fun
	
	def __call__(self, args):
		return list(map(self.fun, args))


def relative_error(original, recovered):
	total_count = len(original)
	if total_count != len(recovered):
		print('relative_error: different lengths')
		exit()
	true_count = sum(map(lambda x: x[0] == x[1], zip(original, recovered)))
	return (total_count - true_count) / total_count


print('Words reading...')

words = []
with open('input/10000-russian-words-cyrillic-only.txt', 'r', encoding = 'utf-8') as fin:
	words = fin.read().splitlines()

print('Done.\n')


word_count = 1000

print(f'Sampling {word_count} words...')

words = np.random.choice(
	list(filter(lambda w: len(w) > 3, words)),
	size = word_count, replace = False
)

print('Done.\n')
#print(f'len(words) = {len(words)}')


print('Preparing distortion functions and model...')

distortfun_LM_list = [
	ListMapper(dt.rand_add),
	ListMapper(dt.rand_delete),
	ListMapper(dt.rand_replace),
	ListMapper(dt.rand_swap),
]

model = wr.WordRecovery(
	#similarity = op.Compound(outer = op.andor_similarity, inners = [set, op.SetOfLetters()]),
	#similarity = op.Composition(outer = op.andor_similarity, inner = op.set_of_orders),
	#similarity = op.Compound(outer = op.andor_similarity, inners = [op.set_of_orders, op.SetOfOrders()]),
	similarity = op.Ensemble(
		parents = [
			op.Compound(outer = op.andor_similarity, inners = [set, op.SetOfLetters()]),
			op.Compound(outer = op.andor_similarity, inners = [op.set_of_orders, op.SetOfOrders()]),
		],
		crossover = op.mean
	),
	vocabulary = words
)

model_LM = ListMapper(model)

print('Done.\n')


print('Words distortion...')

D_1D = [
	F_i(words)
		for F_i in distortfun_LM_list
]

D_2D = [
	[
		F_i(F_j(words))
			for F_j in distortfun_LM_list
	]
		for F_i in distortfun_LM_list
]

D_3D = [
	[
		[
			F_i(F_j(F_k(words)))
				for F_k in distortfun_LM_list
		]
			for F_j in distortfun_LM_list
	]
		for F_i in distortfun_LM_list
]

print('Done.\n')
#print(f'np.array(D_1D).shape = {np.array(D_1D).shape}')
#print(f'np.array(D_2D).shape = {np.array(D_2D).shape}')
#print(f'np.array(D_3D).shape = {np.array(D_3D).shape}')
#exit()


print('Words 1D recovery...')

time_start = time.time()

R_1D = [model_LM(D_i) for D_i in D_1D]

time_finish = time.time()

total_s = time_finish - time_start
average_ms = total_s / (len(distortfun_LM_list) * word_count) * 1000

print('Done.')
print(f'Average word recovery time: {average_ms:.3f} ms. Total time: {total_s:.3f} s.\n')


print('Words 2D recovery...')

time_start = time.time()

R_2D = [[model_LM(D_ij) for D_ij in D_i] for D_i in D_2D]

time_finish = time.time()

total_s = time_finish - time_start
average_ms = total_s / ((len(distortfun_LM_list) ** 2) * word_count) * 1000

print('Done.')
print(f'Average word recovery time: {average_ms:.3f} ms. Total time: {total_s:.3f} s.\n')


print('Words 3D recovery...')

time_start = time.time()

R_3D = [[[model_LM(D_ijk) for D_ijk in D_ij] for D_ij in D_i] for D_i in D_3D]

time_finish = time.time()

total_s = time_finish - time_start
average_ms = total_s / ((len(distortfun_LM_list) ** 3) * word_count) * 1000

print('Done.')
print(f'Average word recovery time: {average_ms:.3f} ms. Total time: {total_s:.3f} s.\n')
#print(f'np.array(R_1D).shape = {np.array(R_1D).shape}')
#print(f'np.array(R_2D).shape = {np.array(R_2D).shape}')
#print(f'np.array(R_3D).shape = {np.array(R_3D).shape}')


relerrors_1D = [relative_error(words, R_i) for R_i in R_1D]

relerrors_2D = [[relative_error(words, R_ij) for R_ij in R_i] for R_i in R_2D]

relerrors_3D = [[[relative_error(words, R_ijk) for R_ijk in R_ij] for R_ij in R_i] for R_i in R_3D]

with open('output/test_recovery/relerrors_M.txt', 'w', encoding = 'utf-8') as fout:
	
	#print('+{0:32}+{0:32}+{0:32}+'.format('--------------------------------'), file = fout)
	#print('|{:32}|{:32}|{:32}|'.format('ORIGINAL', 'DISTORTED', 'RECOVERED'), file = fout)
	#print('+{0:32}+{0:32}+{0:32}+'.format('--------------------------------'), file = fout)
	
	#checker = CheckAndAlert(alert = lambda x: print(f'|{x[0]:32}|{x[1]:32}|{x[2]:32}|', file = fout))
	
	#true_count = [sum(map(checker, zip(words, dw, rw))) for dw, rw in zip(dt1_words, rec1_words)]
	
	#print('+{0:32}+{0:32}+{0:32}+'.format('--------------------------------'), file = fout)
	
	#count = len(words)
	
	#print('Relative errors:', file = fout)
	#print(f'\tadd: {1.0 - true_count_add  / count : .3f}', file = fout)
	#print(f'\tdel: {1.0 - true_count_del  / count : .3f}', file = fout)
	#print(f'\trpl: {1.0 - true_count_repl / count : .3f}', file = fout)
	#print(f'\tswp: {1.0 - true_count_swap / count : .3f}', file = fout)
	
	with np.printoptions(formatter = {'all': lambda x: f'{x * 100:>4.1f} % '}):
		print(f'Relative errors 1D:\n{np.array(relerrors_1D)}\n', file = fout)
		print(f'Relative errors 2D:\n{np.array(relerrors_2D)}\n', file = fout)
		print(f'Relative errors 3D:\n{np.array(relerrors_3D)}\n', file = fout)
	
	#print(f'Average relative error: {error1d.mean() * 100 : .3f}%', file = fout)
