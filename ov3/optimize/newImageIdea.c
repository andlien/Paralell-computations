#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>

#include "ppm.h"

// Image from:
// http://7-themes.com/6971875-funny-flowers-pictures.html

typedef struct {
     double red,green,blue;
} AccuratePixel;

typedef struct {
     int x, y;
     AccuratePixel *data;
} AccurateImage;

// Convert ppm to high precision format.
AccurateImage *convertImageToNewFormat(PPMImage *image) {
	// Make a copy
	AccurateImage *imageAccurate;
	imageAccurate = (AccurateImage *)malloc(sizeof(AccurateImage));
	imageAccurate->data = (AccuratePixel*)malloc(image->x * image->y * sizeof(AccuratePixel));
	for(int i = 0; i < image->x * image->y; i++) {
		imageAccurate->data[i].red   = (double) image->data[i].red;
		imageAccurate->data[i].green = (double) image->data[i].green;
		imageAccurate->data[i].blue  = (double) image->data[i].blue;
	}
	imageAccurate->x = image->x;
	imageAccurate->y = image->y;

	return imageAccurate;
}

// Perform the new idea:
void performNewIdeaIteration(AccurateImage *imageOut, AccurateImage *imageIn, int size) {

	int currentX, currentY, countIncluded, offsetOfThePixel, numberOfValuesInEachRow = imageIn->x;
	double redSum, greenSum, blueSum;

	// Iterate over each pixel
	for(int senterX = 0; senterX < imageIn->x; senterX++) {

		for(int senterY = 0; senterY < imageIn->y; senterY++) {

			// For each pixel we compute the magic number
			countIncluded = 0;
			redSum = greenSum = blueSum = 0;
			for(int x = -size; x <= size; x++) {

				for(int y = -size; y <= size; y++) {
					currentX = senterX + x;
					currentY = senterY + y;

					// Check if we are outside the bounds
					if(currentX < 0 || currentX >= imageIn->x || currentY < 0 || currentY >= imageIn->y)
						continue;

					// Now we can begin
					offsetOfThePixel = (numberOfValuesInEachRow * currentY + currentX);

					redSum += imageIn->data[offsetOfThePixel].red;
					greenSum += imageIn->data[offsetOfThePixel].green;
					blueSum += imageIn->data[offsetOfThePixel].blue;

					// Keep track of how many values we have included
					countIncluded++;
				}

			}

			// Update the output image
			int numberOfValuesInEachRow = imageOut->x; // R, G and B
			int offsetOfThePixel = (numberOfValuesInEachRow * senterY + senterX);

			imageOut->data[offsetOfThePixel].red = redSum / countIncluded;
			imageOut->data[offsetOfThePixel].green = greenSum / countIncluded;
			imageOut->data[offsetOfThePixel].blue = blueSum / countIncluded;
		}

	}

}


void performColorSave(double imageInLargeData, double imageInSmallData, double data)
{
	double value = (imageInLargeData - imageInSmallData);
	if(value > 255)
		data = 255;
	else if (value < -1.0) {
		value = 257.0+value;
		if(value > 255)
			data = 255;
		else
			data = floor(value);
	} else if (value > -1.0 && value < 0.0) {
		data = 0;
	} else {
		data = floor(value);
	}
}

// Perform the final step, and return it as ppm.
PPMImage * performNewIdeaFinalization(AccurateImage *imageInSmall, AccurateImage *imageInLarge) {
	PPMImage *imageOut;
	imageOut = (PPMImage *)malloc(sizeof(PPMImage));
	imageOut->data = (PPMPixel*)malloc(imageInSmall->x * imageInSmall->y * sizeof(PPMPixel));

	imageOut->x = imageInSmall->x;
	imageOut->y = imageInSmall->y;

	for(int i = 0; i < imageInSmall->x * imageInSmall->y; i++) {
		double value = (imageInLarge->data[i].red - imageInSmall->data[i].red);
		if(value > 255)
			imageOut->data[i].red = 255;
		else if (value < -1.0) {
			value = 257.0+value;
			if(value > 255)
				imageOut->data[i].red = 255;
			else
				imageOut->data[i].red = floor(value);
		} else if (value > -1.0 && value < 0.0) {
			imageOut->data[i].red = 0;
		} else {
			imageOut->data[i].red = floor(value);
		}

		value = (imageInLarge->data[i].green - imageInSmall->data[i].green);
		if(value > 255)
			imageOut->data[i].green = 255;
		else if (value < -1.0) {
			value = 257.0+value;
			if(value > 255)
				imageOut->data[i].green = 255;
			else
				imageOut->data[i].green = floor(value);
		} else if (value > -1.0 && value < 0.0) {
			imageOut->data[i].green = 0;
		} else {
			imageOut->data[i].green = floor(value);
		}

		value = (imageInLarge->data[i].blue - imageInSmall->data[i].blue);
		if(value > 255)
			imageOut->data[i].blue = 255;
		else if (value < -1.0) {
			value = 257.0+value;
			if(value > 255)
				imageOut->data[i].blue = 255;
			else
				imageOut->data[i].blue = floor(value);
		} else if (value > -1.0 && value < 0.0) {
			imageOut->data[i].blue = 0;
		} else {
			imageOut->data[i].blue = floor(value);
		}
	}


	return imageOut;
}


int main(int argc, char** argv) {

	PPMImage *image;
	if(argc > 1) {
		image = readPPM("flower.ppm");
	} else {
		image = readStreamPPM(stdin);
	}

	AccurateImage *imageAccurate1_tiny = convertImageToNewFormat(image);
	AccurateImage *imageAccurate2_tiny = convertImageToNewFormat(image);

	// Process the tiny case:
	int size_tiny = 2;
	performNewIdeaIteration(imageAccurate2_tiny, imageAccurate1_tiny, size_tiny);
	performNewIdeaIteration(imageAccurate1_tiny, imageAccurate2_tiny, size_tiny);
	performNewIdeaIteration(imageAccurate2_tiny, imageAccurate1_tiny, size_tiny);
	performNewIdeaIteration(imageAccurate1_tiny, imageAccurate2_tiny, size_tiny);
	performNewIdeaIteration(imageAccurate2_tiny, imageAccurate1_tiny, size_tiny);


	AccurateImage *imageAccurate1_small = convertImageToNewFormat(image);
	AccurateImage *imageAccurate2_small = convertImageToNewFormat(image);

	// Process the small case:
	int size_small = 3;
	performNewIdeaIteration(imageAccurate2_small, imageAccurate1_small, size_small);
	performNewIdeaIteration(imageAccurate1_small, imageAccurate2_small, size_small);
	performNewIdeaIteration(imageAccurate2_small, imageAccurate1_small, size_small);
	performNewIdeaIteration(imageAccurate1_small, imageAccurate2_small, size_small);
	performNewIdeaIteration(imageAccurate2_small, imageAccurate1_small, size_small);

	AccurateImage *imageAccurate1_medium = convertImageToNewFormat(image);
	AccurateImage *imageAccurate2_medium = convertImageToNewFormat(image);

	// Process the medium case:
	int size_medium = 5;
	performNewIdeaIteration(imageAccurate2_medium, imageAccurate1_medium, size_medium);
	performNewIdeaIteration(imageAccurate1_medium, imageAccurate2_medium, size_medium);
	performNewIdeaIteration(imageAccurate2_medium, imageAccurate1_medium, size_medium);
	performNewIdeaIteration(imageAccurate1_medium, imageAccurate2_medium, size_medium);
	performNewIdeaIteration(imageAccurate2_medium, imageAccurate1_medium, size_medium);

	AccurateImage *imageAccurate1_large = convertImageToNewFormat(image);
	AccurateImage *imageAccurate2_large = convertImageToNewFormat(image);

	// Do each color channel
	int size_large = 8;
	performNewIdeaIteration(imageAccurate2_large, imageAccurate1_large, size_large);
	performNewIdeaIteration(imageAccurate1_large, imageAccurate2_large, size_large);
	performNewIdeaIteration(imageAccurate2_large, imageAccurate1_large, size_large);
	performNewIdeaIteration(imageAccurate1_large, imageAccurate2_large, size_large);
	performNewIdeaIteration(imageAccurate2_large, imageAccurate1_large, size_large);

	// Save the images.
	PPMImage *final_tiny = performNewIdeaFinalization(imageAccurate2_tiny,  imageAccurate2_small);
	PPMImage *final_small = performNewIdeaFinalization(imageAccurate2_small,  imageAccurate2_medium);
	PPMImage *final_medium = performNewIdeaFinalization(imageAccurate2_medium,  imageAccurate2_large);

	if(argc > 1) {
		writePPM("flower_tiny.ppm", final_tiny);
		writePPM("flower_small.ppm", final_small);
		writePPM("flower_medium.ppm", final_medium);
	} else {
		writeStreamPPM(stdout, final_tiny);
		writeStreamPPM(stdout, final_small);
		writeStreamPPM(stdout, final_medium);
	}

	return 0;
}

