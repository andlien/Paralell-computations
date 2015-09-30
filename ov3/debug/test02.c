#include <stdio.h>
#include <stdlib.h>

int main(int argc, char *argv[])
{
	int *mem = malloc(sizeof(int)*96);
	int i;
	for(i = 0; i < 32; i++)
		mem[i] = i;
	for(i = 0; i < 32; i++)
		mem[i+64] = mem[i];
	free(mem);
	return 0;
}
