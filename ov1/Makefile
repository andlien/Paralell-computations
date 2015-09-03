.PHONY: clean all

START:=2
STOP:=100
PROCESSES:= 8

all: serial parallel

serial: computeSerial.c
	gcc -o $@ computeSerial.c -std=c99 -lm -O2

run_serial: serial
	./serial $(START) $(STOP)

parallel: computeMPI.c
	mpicc -o $@ computeMPI.c -std=c99 -lm -O2

run_parallel: parallel
	mpirun -np $(PROCESSES) ./parallel $(START) $(STOP)

clean:
	rm -f serial
	rm -f parallel
