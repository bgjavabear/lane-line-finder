# **Finding Lane Lines on the Road** 

**Finding Lane Lines on the Road**

The goals / steps of this project are the following:
* Make a pipeline that finds lane lines on the road

### Reflection

My pipeline consists of 5 steps:
1) convert to gray scale and amplify yellow color. 
It is done to distinguish yellow road lines from bright color of road

2) Apply Gaussian Filter to remove noise 

3) find edges using Canny filter

4) Consider edges only in the region of interest

5) find lines using Hough transformation. 
Find average lines for right and left sides of a road. 

### 2. Potential shortcomings with your current pipeline

1) Noises affect directions of lines. Shadows, etc may affect a direction of a line

2) Sometimes a line flickers. 

### 3. Suggest possible improvements to your pipeline

1) Need to take into account colors of lines. Currently I rely only on a brightness gradient.
Should try different channels. For example a saturation channel is the best to identify yellow lines.
Brightness is not enough to make good predictions.


2) Because I identify edges based only on brightness, I had to weaken boundary conditions
as a result i get noise which affects the direction of lines 
