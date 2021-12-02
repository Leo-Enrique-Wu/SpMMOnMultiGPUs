import argparse
import numpy
import io

from scipy.sparse import random
from scipy.sparse import csr_matrix
from scipy.stats import rv_continuous
from numpy.random import default_rng

class CustomDistribution(rv_continuous):
	def _rvs(self,  size=None, random_state=None):
		return random_state.standard_normal(size)

def convert1dArrayToStr(arr:numpy.ndarray, fp:io.TextIOWrapper, itemNumPerLine:int):
	# arrStr = numpy.array2string(a=arr, separator=' ', prefix='', suffix='')
	# arrStr = arrStr.replace("[", "")
	# arrStr = arrStr.replace("]", "")
	
	idx = 0
	arrStr = ""
	for x in numpy.nditer(arr):
		idx = idx + 1
		xStr = numpy.format_float_positional(x, precision=6)
		if idx <= itemNumPerLine:
			arrStr = arrStr + xStr + " "
		else:
			fp.write(arrStr + '\n')
			idx = 1
			arrStr = xStr + " "
	
	if (len(arrStr) > 0):
		fp.write(arrStr + '\n')

def convert2dArrayToStr(arr:numpy.ndarray, fp:io.TextIOWrapper, itemNumPerLine:int):
	
	rowNum = 0
	for row in arr:
		
		rowNum += 1
		rowStr = ""
		
		idx = 0
		for x in numpy.nditer(row):
			idx = idx + 1
			xStr = numpy.format_float_positional(x, precision=6)
			if idx <= itemNumPerLine:
				rowStr = rowStr + xStr + " "
			else:
				fp.write(rowStr + '\n')
				idx = 1
				rowStr = xStr + " "
				
		if (len(rowStr) > 0):
			fp.write(rowStr + '\n')
		
		if (rowNum % 500 == 0):
			print("Finished writing rowNum = %d" % rowNum)
		

def printSparseMatrixToFile(matrix:csr_matrix, fileName:str):
	
	# print(matrix.A)
	
	print("shape of indptr: " + str(matrix.indptr.shape[0]))
	print("shape of indices: " + str(matrix.indices.shape[0]))
	print("shape of data: " + str(matrix.data.shape[0]))
	
	
	outputFile = open(fileName, "wt")
	
	convert1dArrayToStr(S.indptr, outputFile, 10)
	print("Finished writing [indptr]")
	
	nnzNum = str(matrix.indices.shape[0])
	outputFile.write(nnzNum + '\n')
	
	convert1dArrayToStr(S.indices, outputFile, 10)
	print("Finished writing [indices]")
	
	convert1dArrayToStr(S.data, outputFile, 10)
	print("Finished writing [data]")
	
	outputFile.close()
	
	
	
def printDenseMatrixToFile(matrix:csr_matrix, fileName:str):
	
	# print(S.A)
	
	outputFile = open(fileName, "wt")
	convert2dArrayToStr(S.A, outputFile, 5)
	outputFile.close()


if __name__ == "__main__":
	
	# Read Params
	parser = argparse.ArgumentParser(description='Testing Matrix Generator')
	parser.add_argument('-m', dest='mStr', action='store', help="The row number of the generated matrix")
	parser.add_argument('-n', dest='nStr', action='store', help="The column number of the generated matrix")
	parser.add_argument('-isSp', dest='isSpStr', action='store', help="Whether is the generated matrix a sparse matrix? T or F")
	parser.add_argument('-density', dest='densityStr', action='store', help="The density of the sparse matrix")
	parser.add_argument('-o', dest='outputFileName', action='store', help="The file name of the output file")
	args = parser.parse_args()
	
	m = int(args.mStr)
	n = int(args.nStr)
	isSp = True if (args.isSpStr == "T") else False
	density = 1 if (not(isSp)) else float(args.densityStr)
	matrixType = "dense" if (not(isSp)) else ("sparse(density=" + str(density) + ")")
	outputFileName = args.outputFileName
	print("Generating %d x %d %s matrix to %s" % (m, n, matrixType, outputFileName))
	
	rng = default_rng()
	X = CustomDistribution(seed=rng)
	Y = X()  # get a frozen version of the distribution
	
	S = random(m, n, density=density, format='csr', random_state=rng, data_rvs=Y.rvs)
	print("Generated the matrix.")
	
	if (isSp):
		printSparseMatrixToFile(S, outputFileName)
	else:
		printDenseMatrixToFile(S, outputFileName)
	print("Done.")