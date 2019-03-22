import numpy as np
import math
import sys
def convert_to_bin(data):
	result = data / 8
	result = result.astype(int)
	result = result.astype("float64")
	return result

def discrete_classifier(data,labels,test_data,test_labels):
	prob_digit = dict()
	data_digit_count = dict()

	for i in range(10):
		prob_digit[i] = 0
		data_digit_count[i] = 0

	for i in labels:
		prob_digit[i] = prob_digit[i] + 1
		data_digit_count[i] = data_digit_count[i] + 1

	for i in range(10):
		prob_digit[i] = prob_digit[i] / len(labels)
	
	
	liklihood = dict()
	for d in range(10):
		liklihood[d] = dict()
		for i in range(28*28):
			liklihood[d][i] = dict()
			for j in range(32):
				liklihood[d][i][j] = 0

	for i in range(data.shape[0]):
		count = 0
		for j in range(data.shape[1]):
			for k in range(data.shape[2]):
				liklihood[labels[i]][count][data[i][j][k]] = liklihood[labels[i]][count][data[i][j][k]] + 1
				count = count + 1

	for i in range(10):
		for j in range(28*28):
			min = 33
			for k in range(32):
				if liklihood[i][j][k] != 0 and liklihood[i][j][k] < min:
					min = liklihood[i][j][k]
			for k in range(32):
				if liklihood[i][j][k] == 0:
					liklihood[i][j][k] = min/ data_digit_count[i]
				else:
					liklihood[i][j][k] = liklihood[i][j][k] / data_digit_count[i]
	
	error = 0
	for i in range(test_data.shape[0]):
		ans = []
		sum = 0
		for j in range(10):
			posterior = 0
			count = 0
			for idx1 in range(test_data.shape[1]):
				for idx2 in range(test_data.shape[2]):
					#if liklihood[j][count][test_data[i][idx1][idx2]] > 0:
					posterior = posterior + math.log(liklihood[j][count][test_data[i][idx1][idx2]])
					count = count + 1
			if prob_digit[j] > 0:
				posterior = posterior + math.log(prob_digit[j])
		
			sum = sum + posterior
			ans.append(posterior)
		min = ans[0]/sum
		pred = 0
	
		for j in range(10):
			print(j,end='')
			print(": ",end = '')
			temp = ans[j]/sum
			print(temp)
			if temp < min:
				min = temp
				pred = j
		print("Prediction: %d, Ans: %d" %(pred,test_labels[i]))
		if pred != test_labels[i]:
			error = error + 1
		
	for i in range(10):
		count = 0
		print("%d:" %(i))
		for j in range(28):
			for k in range(28):
				prob_white = 0
				prob_black = 0
				for l in range(16):
					prob_white = prob_white + liklihood[i][count][l]
					prob_black = prob_black + liklihood[i][count][31-l]
				count = count + 1
				if prob_black > prob_white:
					print("1 ",end = '')
				else:
					print("0 ",end = '')
			print('')

	print(error/len(test_labels))

def mean(data,data_digit_idx):
	
	list = []
	
	for i in range(10):
		sum = np.zeros((28,28),dtype = 'float64')
		for j in range(len(data_digit_idx[i])):
			sum = sum + data[data_digit_idx[i][j]]
		sum = sum/len(data_digit_idx[i])
		list.append(sum)
	return list

def var(data,data_digit_idx,mean_list):
	
	list1 = []
	for i in range(10):
		sum_sqrt = np.zeros((28,28),dtype = 'float64')
		
		for j in range(len(data_digit_idx[i])):
			sum_sqrt = sum_sqrt + (data[data_digit_idx[i][j]] - mean_list[i]) ** 2
		
		sum_sqrt = sum_sqrt/len(data_digit_idx[i])
		
		list1.append(sum_sqrt)

	return list1
	




def continue_classifier(data,labels,test_data,test_labels):
	prob_digit = dict()
	data_digit_idx = dict()

	for i in range(10):
		prob_digit[i] = 0
		data_digit_idx[i] = []

	for i in labels:
		prob_digit[i] = prob_digit[i] + 1
	
	for i in range(len(labels)):
		data_digit_idx[labels[i]].append(i)

	for i in range(10):
		prob_digit[i] = prob_digit[i] / len(labels)

	mean_list = mean(data,data_digit_idx)
	var_list = var(data,data_digit_idx,mean_list)
	error = 0

	for idx in range(test_data.shape[0]):
		ans = []
		sum = 0
		for i in range(10):
			posterior = 0
			for j in range(28):
				for k in range(28):
						
					a = (-0.5)*math.log(2*math.pi) 
					b = -math.log(var_list[i][j][k]+1e-2)
					c = (test_data[idx][j][k] - mean_list[i][j][k]) 
					d = (var_list[i][j][k]+1e-2) 

					gaussian = a + b + (-0.5)*((c/d)**2)

					posterior = posterior + gaussian
			
			
			posterior = posterior + math.log(prob_digit[i])
			
			ans.append(posterior)
			sum = sum + posterior
	
		min = ans[0]/sum
		pred = 0

		for j in range(10):
			print(j,end='')
			print(": ",end = '')
			temp = ans[j]/sum
			print(temp)
			if temp < min:
				min = temp
				pred = j
		print("Prediction: %d, Ans: %d" %(pred,test_labels[idx]))
		if pred != test_labels[idx]:
			error = error + 1

	data = data*256
	mean_list = mean(data,data_digit_idx)
	var_list = var(data,data_digit_idx,mean_list)
	for i in range(10):
		
		print("%d:" %(i))
		for j in range(28):
			for k in range(28):
				prob_white = 0
				prob_black = 0
				for l in range(128):
					gaussian1 = (2*math.pi*(var_list[i][j][k]+1e-2)**2) ** (-0.5) * math.exp((-0.5) * (l - mean_list[i][j][k]) ** 2 / (var_list[i][j][k]+1e-2) ** 2)
					gaussian2 = (2*math.pi*(var_list[i][j][k]+1e-2)**2) ** (-0.5) * math.exp((-0.5) * (255-l - mean_list[i][j][k]) ** 2 / (var_list[i][j][k]+1e-2) ** 2)
					prob_white = prob_white + gaussian1
					prob_black = prob_black + gaussian2
				
				if prob_black > prob_white:
					print("1 ",end = '')
				else:
					print("0 ",end = '')
			print('')
		
	print(error/len(test_labels))


def main():
	argument = sys.argv[1:]
	if len(argument) != 1:
		print("You should input all the arguments,including path to testfile, number of bases n and lambda")
		sys.exit(1)


	conti_or_dis = int(argument[0])
	data_type = np.dtype("int32").newbyteorder('>')

	
	data = np.fromfile("train-images.idx3-ubyte", dtype = "ubyte")
	magic_number,number_of_images,number_of_rows,number_of_columns = np.frombuffer(data[:4*data_type.itemsize],data_type)
	
	data = data[4*data_type.itemsize:].astype("float64").reshape([number_of_images,number_of_rows,number_of_columns])

	labels = np.fromfile("train-labels.idx1-ubyte",dtype = "ubyte").astype("int")
	labels = labels[2*data_type.itemsize:]
	
	test_data = np.fromfile("t10k-images.idx3-ubyte", dtype = "ubyte")
	magic_number,number_of_images,number_of_rows,number_of_columns = np.frombuffer(test_data[:4*data_type.itemsize],data_type)
	
	test_data = test_data[4*data_type.itemsize:].astype("float64").reshape([number_of_images,number_of_rows,number_of_columns])

	test_labels = np.fromfile("t10k-labels.idx1-ubyte",dtype = "ubyte").astype("int")
	test_labels = test_labels[2*data_type.itemsize:]
	if conti_or_dis == 0:
		bin = convert_to_bin(data)
		test_bin = convert_to_bin(test_data)
		discrete_classifier(bin,labels,test_bin,test_labels)
	else:
		data_con = data/256
		test_data_con = test_data/256
		continue_classifier(data_con,labels,test_data_con,test_labels)

	
	

	

if __name__ == '__main__':
	main()
