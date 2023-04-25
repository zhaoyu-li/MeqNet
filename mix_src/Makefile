
CC=/usr/bin/gcc
# GCC_LIB_INCLUDE=/usr/lib/gcc/x86_64-linux-gnu/11/include 
# CC_FLAGS = -I ${GCC_LIB_INCLUDE} -I .

learn.exe: mix_learn.o mixnet_cpu.o
	${CC} -g -o $@ $^ -lm -lstdc++

%.o: %.cpp mixnet.h
	${CC} -g -c $<   -msse4.1

clear:
	rm *.o *.exe