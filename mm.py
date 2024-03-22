import numpy as np
import ctypes as C
import os

this_dir, this_filename = os.path.split(__file__)
_lib = np.ctypeslib.load_library('libMM', os.path.join(this_dir, "."))    

class CMulticlassMerlin(C.Structure):
	None

class CMerlinModel(C.Structure):
	None

mc_mm_pointer = C.POINTER(CMulticlassMerlin)

mm_pointer = C.POINTER(CMerlinModel)

array_1d_uint = np.ctypeslib.ndpointer(
	dtype=np.uint32,
	ndim=1,
	flags='CONTIGUOUS')

array_1d_int = np.ctypeslib.ndpointer(
	dtype=np.int32,
	ndim=1,
	flags='CONTIGUOUS')


# Multiclass Tsetlin Machine

_lib.CreateMultiClassMerlinMachine.restype = mc_mm_pointer                    
_lib.CreateMultiClassMerlinMachine.argtypes = [C.c_int, C.c_int, C.c_int, C.c_int, C.c_int, C.c_int, C.c_double, C.c_double] 

_lib.mc_mm_fit.restype = None                      
_lib.mc_mm_fit.argtypes = [mc_mm_pointer, array_1d_uint, array_1d_uint, C.c_int, C.c_int] 

_lib.mc_mm_initialize.restype = None                      
_lib.mc_mm_initialize.argtypes = [mc_mm_pointer] 

_lib.mc_mm_predict.restype = None                    
_lib.mc_mm_predict.argtypes = [mc_mm_pointer, array_1d_uint, array_1d_uint, C.c_int] 

# Tsetlin Machine

_lib.CreateMerlinMachine.restype = mm_pointer                    
_lib.CreateMerlinMachine.argtypes = [C.c_int, C.c_int, C.c_int, C.c_int, C.c_int, C.c_double, C.c_double] 

# Tools

_lib.mm_encode.restype = None                      
_lib.mm_encode.argtypes = [array_1d_uint, array_1d_uint, C.c_int, C.c_int, C.c_int, C.c_int, C.c_int, C.c_int] 



class MultiClassMerlinModel():
	def __init__(self, T, s, number_of_state_bits=8, s_range=False):
		self.number_of_state_bits = number_of_state_bits
		self.T = int(T)
		self.s = s
		self.mc_mm = None
		self.imm = None
		if s_range:
			self.s_range = s_range
		else:
			self.s_range = s

	def fit(self, X, Y, epochs=100, incremental=False):
		number_of_examples = X.shape[0]

		self.number_of_classes = int(np.max(Y) + 1)

		if self.mc_mm == None:
			self.number_of_features = X.shape[1]*2

			self.number_of_patches = 1
			self.number_of_ta_chunks = int((self.number_of_features-1)/32 + 1)
			self.mc_mm = _lib.CreateMultiClassMerlinMachine(self.number_of_classes, self.number_of_features, 1, self.number_of_ta_chunks, self.number_of_state_bits, self.T, self.s, self.s_range)
		elif incremental == False:
			_lib.mc_tm_destroy(self.mc_tm)
			self.mc_mm = _lib.CreateMultiClassMerlinMachine(self.number_of_classes, self.number_of_features, 1, self.number_of_ta_chunks, self.number_of_state_bits, self.T, self.s, self.s_range)

		self.encoded_X = np.ascontiguousarray(np.empty(int(number_of_examples * self.number_of_ta_chunks), dtype=np.uint32))

		Xm = np.ascontiguousarray(X.flatten()).astype(np.uint32)
		Ym = np.ascontiguousarray(Y).astype(np.uint32)

		_lib.mm_encode(Xm, self.encoded_X, number_of_examples, self.number_of_features//2, 1, 1, self.number_of_features//2, 1, 1)

		_lib.mc_mm_fit(self.mc_mm, self.encoded_X, Ym, number_of_examples, epochs)

		return

	def predict(self, X):
		number_of_examples = X.shape[0]
		
		self.encoded_X = np.ascontiguousarray(np.empty(int(number_of_examples * self.number_of_patches * self.number_of_ta_chunks), dtype=np.uint32))

		Xm = np.ascontiguousarray(X.flatten()).astype(np.uint32)

		_lib.mm_encode(Xm, self.encoded_X, number_of_examples, self.number_of_features//2, 1, 1, self.number_of_features//2, 1, 1)
	
		Y = np.ascontiguousarray(np.zeros(number_of_examples, dtype=np.uint32))

		_lib.mc_mm_predict(self.mc_mm, self.encoded_X, Y, number_of_examples)

		return Y