.PHONY: clean all

START:=7
STOP:=41
PROCESSES:= 11

all: run_serial run_parallel

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
