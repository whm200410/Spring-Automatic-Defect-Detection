# Spring-Automatic-Defect-Detection
this algorithm is used to identify all the defects of a tray which contains maximum 30 individual spring.
the image is captured using xray machine, there may have scratchs inside spring or the spring may miss corners.

this algorithm uses opencv as image processing library to process it.
it did the following step

1. blue the image to reduce noise
2. seperate each spring in a tray by find all the cols and rows of each individual
3. for each spring.
 3.1) find center point by up and bottom spring bar
 3.2) find norner position of spring
 3.3) make a thredshold holding all the potencial valid pixels.
 3.4) make another thredshold to exclude the black hole in furthur sobel image
 3.5) make MORPH_GRADIENT image to exclude the border in furthuer sobel image
 3.6) generate a lut image based on potencial gray range to maximize the brightness
 3.7) generate the soble image to detect edge
 3.8) exclude border from sobel image
 3.9) exclude hole from sobel image.
 3.10) make a Hysteresis image based on sobel image, applying the low and high thredhold
 3.11) find contour of the Hysteresis image to check defects
 3.12) filter contour which contains large tham certain pixcels size
 3.13) get all the image.
