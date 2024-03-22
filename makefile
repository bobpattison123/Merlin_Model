libMM: MultiClassMerlinSingle.c MultiClassMerlinSingle.h MerlinModel.c MerlinModel.h Tools.c Tools.h 
	gcc -O3 -g -fopenmp -ffast-math -c -Wall -Werror -fpic MultiClassMerlinSingle.c MerlinModel.c Tools.c
	gcc -shared -g -O3 -fopenmp -ffast-math -o libMM.so MultiClassMerlinSingle.o MerlinModel.o Tools.o

clean:
	rm *.o libMM.so