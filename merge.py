import numpy as np
import csv

#take two result
file1 = 'resultsvm.csv'
file2 = 'result.csv'

#weights for merging result
coef1 = 0.3
coef2 = 0.7

fileout = 'resultmerge.csv'

#read first file
names1 = []
data1 = []
flag1 = 0;
with open(file1, 'r') as fp1:
	reader1 = csv.reader(fp1)
	for row in reader1:
		#for title in result		
		if (flag1 == 0):
			flag1 = 1
		else:
			names1.append(row[0])
			data1.append(float(row[1]))

#read second file
names2 = []
data2 = []
flag2 = 0;
with open(file2, 'r') as fp2:
	reader2 = csv.reader(fp2)
	for row in reader2:
		#for title in result		
		if (flag2 == 0):
			flag2 = 1
		else:
			names2.append(row[0])
			data2.append(float(row[1]))

#make new result
data = np.array(data1)*coef1 + np.array(data2)*coef2

fout = open(fileout, 'w')
fout.write('driver_trip,prob\n')

for i in range(0, len(indices1)):
	fout.write(indices1[i] + ',' + str(data[i]) + '\n')

fout.close()
