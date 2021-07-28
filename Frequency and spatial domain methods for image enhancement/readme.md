# Frequency and spatial domain methods for image enhancement
## problem
Photos are often capture on camera as appeared distorted or in low quality. In this circumstance, a photo of car with a hard to read license plate
. Identify a license plate number from a photo of a suspect's car. The photo is a copy of a newspaper print affected by a regular pattern of noise, and the license plate number is hardly visible.
You solution strategy is application of the Discrete Fourier Transform and figuring out an appropriate filter in the frequency domain for periodic noise suppression. You will then perform post-processing techniques to further improve image contrast.¶


![car](https://user-images.githubusercontent.com/77212888/127365062-c3426285-0a3b-4801-8102-e018240bc437.png)
## process

1. Calculate the Discrete Fourier transform of a car image, shift the origin of the image domain to the centre, and calculate the magnitude of the Fourier transform
2. find the coordinates of local maxima of this function using "peak_local_max" function
3. clean the original image in the Fourier domain, the "bright sparks"
4. Perform the inverse Fourier transform
5. Image post-processing
