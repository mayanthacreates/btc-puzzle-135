CC      = clang
CFLAGS  = -O3 -march=native -funroll-loops -flto -Wall -Wno-unused-function
INC     = -I/opt/homebrew/include
LIB     = -L/opt/homebrew/lib -lsecp256k1
PTH     = -lpthread

all: kangaroo test_field test_group

kangaroo: kangaroo.c field.h group.h
	$(CC) $(CFLAGS) -o kangaroo kangaroo.c $(PTH)

test_field: test_field.c field.h
	$(CC) $(CFLAGS) -o test_field test_field.c

test_group: test_group.c group.h field.h
	$(CC) $(CFLAGS) $(INC) -o test_group test_group.c $(LIB)

gpu_test: gpu_test.m gpu_field.metal group.h field.h
	$(CC) -O3 -fobjc-arc -framework Metal -framework Foundation -o gpu_test gpu_test.m

check: test_field test_group
	./test_field 5000 | python3 check_field.py
	./test_group 3000

check-gpu: gpu_test
	./gpu_test 8192

clean:
	rm -f kangaroo test_field test_group
