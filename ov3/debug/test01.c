#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char *argv[])
{
	char *mem;
	mem = (char *) malloc(sizeof(char)*7);
	strcpy(mem, "Hello\n");
	printf("%s", mem);
	free(mem);
	return 0;
}
