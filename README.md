# dye-based-flow-structure-identification
7/21/2023 Portland State University

custom python code for vortex identification and wake frequency analysis

https://github.com/ofercak/dye-based-flow-structure-videntification

1. DV_0_K-Means.py is a simple K-Means quantization algorithm to illustrate the method. 

2. DV_1_Extract_Color_Images.py is a specific implementation of the K-Means algorithm to extract colors from images. There are two stages: The first is to quantize the image colors and note them [RGB values] for future use. Second use the [RGB values] to create a mask and extract only the needed colors from your images. Two are used in the example but there is no limit to the total number, just computation time.

3. DV_2_Vortex_Tracking.py is a specific usage code for tracking vortices (also know as blobs in computer vision) through a set of video images. This is achieved through a set of filtering commands that remove noise and diffuse sections of dye through morphological transformations.

4. DV_2_Vortex_Tracking_DNS.py is a specific usage code for tracking vortices in a different set of video images. 

**Note that 3 & 4 are essentially the same code but fine tuned for a specific application. Any attemts to use this code for simmilar purposes will require testing and fine tuning of parameters. Also, note that object, vortice, or blob tracking is highly dependent on the underlying conditions of the images, or in this case flow. I dye is tracked it has to contain coherent structures otherwise there is nothing for the algorithm to zero in on. Statistical filtering of the resulting positions is also recomended, and a sorting algorithm may be required if many particles, blobs, or vortices are to be identified. 
