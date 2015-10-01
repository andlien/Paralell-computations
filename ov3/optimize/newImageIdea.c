#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>

#include "ppm.h"

typedef struct {
     double red,green,blue;
} AccuratePixel;

typedef struct {
     int x, y;
     AccuratePixel *data;
} AccurateImage;

AccurateImage *convertImageToNewFormat(PPMImage *image)
{
	int i, size = image->x * image->y;

	AccurateImage *imageAccurate;
	imageAccurate = (AccurateImage *) malloc(sizeof(AccurateImage));
	imageAccurate->data = (AccuratePixel*) malloc(size * sizeof(AccuratePixel));

	for (i = 0; i < size; i++)
	{
		imageAccurate->data[i].red   = (double) image->data[i].red;
		imageAccurate->data[i].green = (double) image->data[i].green;
		imageAccurate->data[i].blue  = (double) image->data[i].blue;
	}
	imageAccurate->x = image->x;
	imageAccurate->y = image->y;

	return imageAccurate;
}

AccurateImage *performNewIdea(PPMImage *image, int size)
{
	AccurateImage *tempOut, *tempIn, *imageIn = convertImageToNewFormat(image), *imageOut = convertImageToNewFormat(image);

	int i, x, y, centerX, centerY, currentX, currentY, countIncluded, offsetOfThePixel, imageInX = imageIn->x, imageInY = imageIn->y, imageOutX = imageOut->x;
	double redSum, greenSum, blueSum;

	// performNewIdeaIteration
	for (i = 0; i < 5; i++)
	{
		tempOut = (i % 2 == 0 ? imageOut : imageIn);
		tempIn = (i % 2 == 0 ? imageIn : imageOut);
		int count = 0;

		for (centerX = 0; centerX < imageInX; centerX++)
		{
			for (centerY = 0; centerY < imageInY; centerY++)
			{
				countIncluded = 0;
				redSum = greenSum = blueSum = 0;
				for (x = -size; x <= size; x++)
				{
					for (y = -size; y <= size; y++)
					{
						currentX = centerX + x;
						currentY = centerY + y;

						if (currentX < 0 || currentX >= imageInX || currentY < 0 || currentY >= imageInY)
						{
							continue;
						}

						offsetOfThePixel = (imageInX * currentY + currentX);
						redSum += tempIn->data[offsetOfThePixel].red;
						greenSum += tempIn->data[offsetOfThePixel].green;
						blueSum += tempIn->data[offsetOfThePixel].blue;
						count++;
						countIncluded++;
					}

				}

				offsetOfThePixel = (imageOutX * centerY + centerX);
				tempOut->data[offsetOfThePixel].red = redSum / countIncluded;
				tempOut->data[offsetOfThePixel].green = greenSum / countIncluded;
				tempOut->data[offsetOfThePixel].blue = blueSum / countIncluded;
			}
		}
		printf("Size: %d, Count: %d\n", size, count);
	}
	return imageOut;
}


double performColorSave(double imageInLargeData, double imageInSmallData)
{
	double value = (imageInLargeData - imageInSmallData);
	if (value > 255)
		return 255;
	else if (value < -1.0)
	{
		value = 257.0 + value;
		return value > 255 ? 255 : floor(value);
	}
	else if (value > -1.0 && value < 0.0)
		return 0;
	else
		return floor(value);
}

PPMImage * performNewIdeaFinalization(AccurateImage *imageInSmall, AccurateImage *imageInLarge)
{
	PPMImage *imageOut;
	imageOut = (PPMImage *)malloc(sizeof(PPMImage));
	imageOut->data = (PPMPixel*)malloc(imageInSmall->x * imageInSmall->y * sizeof(PPMPixel));
	imageOut->x = imageInSmall->x;
	imageOut->y = imageInSmall->y;

	int i, size = imageInSmall->x * imageInSmall->y;

	for(i = 0; i < size; i++)
	{
		imageOut->data[i].red = performColorSave(imageInLarge->data[i].red, imageInSmall->data[i].red);
		imageOut->data[i].green = performColorSave(imageInLarge->data[i].green, imageInSmall->data[i].green);
		imageOut->data[i].blue = performColorSave(imageInLarge->data[i].blue, imageInSmall->data[i].blue);
	}

	return imageOut;
}


int main(int argc, char** argv)
{
	PPMImage *image;
	image = argc > 1 ? readPPM("flower.ppm") : readStreamPPM(stdin);

	AccurateImage *imageAccurate_tiny = performNewIdea(image, 2);
	AccurateImage *imageAccurate_small = performNewIdea(image, 3);
	AccurateImage *imageAccurate_medium = performNewIdea(image, 5);
	AccurateImage *imageAccurate_large = performNewIdea(image, 8);

	PPMImage *final_tiny = performNewIdeaFinalization(imageAccurate_tiny,  imageAccurate_small);
	PPMImage *final_small = performNewIdeaFinalization(imageAccurate_small,  imageAccurate_medium);
	PPMImage *final_medium = performNewIdeaFinalization(imageAccurate_medium,  imageAccurate_large);

	if(argc > 1)
	{
		writePPM("flower_tiny.ppm", final_tiny);
		writePPM("flower_small.ppm", final_small);
		writePPM("flower_medium.ppm", final_medium);
	}
	else
	{
		writeStreamPPM(stdout, final_tiny);
		writeStreamPPM(stdout, final_small);
		writeStreamPPM(stdout, final_medium);
	}

	return 0;
}

