#include <CL/opencl.h>
// Mac: OpenCL/opencl.h, Linux: CL/opencl.h
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <tgmath.h>

#include "cl_error_helper.h"
#include "lodepng.h"

#define DATA_SIZE 10

struct Color
{
	float angle;
	float intensity;
};

struct CircleInfo
{
	float x;
	float y;
	float radius;
	struct Color color;
};

struct LineInfo
{
	float x1, y1;
	float x2, y2;
	float thickness;
	struct Color color;
};

char *readText(const char *filename)
{
	FILE * file = fopen(filename, "r");
	fseek(file, 0, SEEK_END);
	size_t length = ftell(file);
	(void) fseek(file, 0L, SEEK_SET);
	char * content = calloc(length+1, sizeof(char));
	int itemsread = fread(content, sizeof(char), length, file);
	if (itemsread != length)
	{
		printf("Error, reeadText(const char *), failed to read file");
		exit(1);
	}
	return content;
}


void parseLine(char *line, struct LineInfo li[], cl_int *lines)
{
	float x1, x2, y1, y2, thickness, angle, intensity;
	int items = sscanf(line, "line %f,%f %f,%f %f %f,%f", &x1, &y1, &x2, &y2, &thickness, &angle, &intensity);
	if (items == 7)
	{
		li[*lines].x1 = x1;
		li[*lines].x2 = x2;
		li[*lines].y1 = y1;
		li[*lines].y2 = y2;
		li[*lines].thickness = thickness;
		li[*lines].color.angle = angle;
		li[*lines].color.intensity = intensity;
		(*lines)++;
	}
}


void parseCircle(char *line, struct CircleInfo ci[], cl_int *circles)
{
	float x, y, radius;
	struct Color c;
	int items = sscanf(line, "circle %f,%f %f %f,%f", &x, &y, &radius, &c.angle, &c.intensity);
	if (items == 5)
	{
		ci[*circles].x = x;
		ci[*circles].y = y;
		ci[*circles].radius = radius;
		ci[*circles].color.angle = c.angle;
		ci[*circles].color.intensity = c.intensity;
		(*circles)++;
	}
}


void printLines(struct LineInfo li[], cl_int lines)
{
	for (int i = 0; i < lines; i++)
	{
		printf("line: from:%f,%f to:%f,%f thick:%f, %f,%f\n",
			   li[i].x1,
			   li[i].y1,
			   li[i].x2,
			   li[i].y2,
			   li[i].thickness,
			   li[i].color.angle,
			   li[i].color.intensity
		);
	}
}


void printCircles(struct CircleInfo ci[], cl_int circles)
{
	for (int i = 0; i < circles; i++)
	{
		printf("circle %f,%f %f %f,%f\n",
			   ci[i].x,
			   ci[i].y,
			   ci[i].radius,
			   ci[i].color.angle,
			   ci[i].color.intensity
		);
	}
}

int printPossibleError(char *tag, int error_code)
{
	printf("Tag %s with cl code %s", tag, getErrorString(error_code));

	if (error_code != CL_SUCCESS) {
		return 1;
	}
	return 0;
}


int main()
{
	// Parse input
	int numberOfInstructions;
	char* *instructions = NULL;
	size_t *instructionLengths;

	struct CircleInfo *circleinfo;
	cl_int circles = 0;
	struct LineInfo *lineinfo;
	cl_int lines = 0;

	char *line = NULL;
	size_t linelen = 0;
	int width=0, height = 0;
	ssize_t read = getline(&line, &linelen, stdin);

	// Read size of canvas
	sscanf(line, "%d,%d", &width, &height);
	read = getline(&line, &linelen, stdin);

	// Read amount of primitives
	sscanf(line, "%d", &numberOfInstructions);

	// Allocate memory for primitives
	instructions = calloc(sizeof(char*), numberOfInstructions);
	instructionLengths = calloc( sizeof(size_t), numberOfInstructions);
	circleinfo = calloc( sizeof(struct CircleInfo), numberOfInstructions);
	lineinfo = calloc( sizeof(struct LineInfo), numberOfInstructions);

	// Read in each primitive
	for (int i = 0; i < numberOfInstructions; i++)
	{
		ssize_t read = getline(&instructions[i], &instructionLengths[i], stdin);
		/*Read in the line or circle here*/
	}

	unsigned char *image;

	/*** START EXAMPLE PROGRAM ***/

	cl_context context;
	cl_context_properties properties[3];
	cl_kernel kernel;
	cl_command_queue command_queue;
	cl_program program;
	cl_int err;
	cl_uint num_of_platforms=0;
	cl_platform_id platform_id;
	cl_device_id device_id;
	cl_uint num_of_devices=0;
	cl_mem input, output;
	size_t global;

	float inputData[DATA_SIZE]={1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
	float results[DATA_SIZE]={0};

	int i, cl_code;

	// retreives a list of platforms available
	cl_code = clGetPlatformIDs(1, &platform_id, &num_of_platforms);
	if (printPossibleError("clGetPlatformIDs", cl_code))
		return 1;

	// try to get a supported GPU device
	cl_code = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU, 1, &device_id, &num_of_devices);
	if (printPossibleError("clGetDeviceIDs", cl_code))
		return 1;

	// context properties list - must be terminated with 0
	properties[0]= CL_CONTEXT_PLATFORM;
	properties[1]= (cl_context_properties) platform_id;
	properties[2]= 0;

	// create a context with the GPU device
	context = clCreateContext(properties, 1, &device_id, NULL, NULL, &err);
	if (printPossibleError("clCreateContext", err))
		return 1;

	// create command queue using the context and device
	command_queue = clCreateCommandQueue(context, device_id, 0, &err);
	if (printPossibleError("clCreateCommandQueue", err))
		return 1;

	// create a program from the kernel source code
	char *source = readText("kernel.cl");
	program = clCreateProgramWithSource(context, 1, (const char **) &source, NULL, &err);
	if (printPossibleError("clCreateProgramWithSource", err))
		return 1;

	// compile the program
	cl_code = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
	if (printPossibleError("clBuildProgram", cl_code))
		return 1;

	// specify which kernel from the program to execute
	kernel = clCreateKernel(program, "hello", &err);
	if (printPossibleError("clCreateKernel", err))
		return 1;

	// create buffers for the input and ouput
	input = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) *DATA_SIZE, NULL, &err);
	if (printPossibleError("clCreateBuffer:CL_MEM_READ_ONLY", err))
		return 1;
	output = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float) *DATA_SIZE, NULL, &err);
	if (printPossibleError("clCreateBuffer:CL_MEM_WRITE_ONLY", err))
		return 1;

	// load data into the input buffer
	cl_code = clEnqueueWriteBuffer(command_queue, input, CL_TRUE, 0, sizeof(float) *DATA_SIZE, inputData, 0, NULL, NULL);
	if (printPossibleError("clEnqueueWriteBuffer", cl_code))
		return 1;

	// set the argument list for the kernel command
	cl_code = clSetKernelArg(kernel, 0, sizeof(cl_mem), &input);
	if (printPossibleError("clSetKernelArg:0", cl_code))
		return 1;
	cl_code = clSetKernelArg(kernel, 1, sizeof(cl_mem), &output);
	if (printPossibleError("clSetKernelArg:1", cl_code))
		return 1;

	global = DATA_SIZE;

	// enqueue the kernel command for execution
	cl_code = clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL, &global, NULL, 0, NULL, NULL);
	if (printPossibleError("clEnqueueNDRangeKernel", cl_code))
		return 1;
	cl_code = clFinish(command_queue);
	if (printPossibleError("clFinish", cl_code))
		return 1;

	// copy the results from out of the output buffer
	cl_code = clEnqueueReadBuffer(command_queue, output, CL_TRUE, 0, sizeof(float) * DATA_SIZE, results, 0, NULL, NULL);
	if (printPossibleError("clEnqueueReadBuffer", cl_code))
		return 1;

	// print the results
	printf("output: ");
	for(i = 0; i < DATA_SIZE; i++)
	{
		printf("%f ",results[i]);
	}

	// cleanup - release OpenCL resources
	cl_code = clReleaseMemObject(input);
	if (printPossibleError("clReleaseMemObject", cl_code))
		return 1;
	cl_code = clReleaseMemObject(output);
	if (printPossibleError("clReleaseMemObject", cl_code))
		return 1;
	cl_code = clReleaseProgram(program);
	if (printPossibleError("clReleaseProgram", cl_code))
		return 1;
	cl_code = clReleaseKernel(kernel);
	if (printPossibleError("clReleaseKernel", cl_code))
		return 1;
	cl_code = clReleaseCommandQueue(command_queue);
	if (printPossibleError("clReleaseCommandQueue", cl_code))
		return 1;
	cl_code = clReleaseContext(context);
	if (printPossibleError("clReleaseContext", cl_code))
		return 1;

	/*** END EXAMPLE PROGRAM ***/

	size_t memfile_length = 0;
	unsigned char * memfile = NULL;
	lodepng_encode24(
		&memfile,
		&memfile_length,
		image,
		width,
		height);

	// KEEP THIS LINE. Or make damn sure you replace it with something equivalent.
	// This "prints" your png to stdout, permitting I/O redirection
	fwrite(memfile, sizeof(unsigned char), memfile_length, stdout);

	return 0;
}
