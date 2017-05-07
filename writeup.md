## Advanced Lane Finding Write-Up

---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[distorted_chessboard]: ./output_images/distorted_chessboard.png "Distorted Chessboard"
[undistorted_chessboard]: ./output_images/undistorted_chessboard.png "Undistorted Chessboard"
[original_image]: ./output_images/1_original_image.png "Original Image"
[corrected_camera_distortion]: ./output_images/2_corrected_camera_distortion.png "Corrected Camera Distortion"
[thresholded]: ./output_images/3_thresholded_image.png "Thresholded Image"
[warped]: ./output_images/4_warped_perspective.png "Warped Perspective"
[histogram]: ./output_images/5_histogram.png "Histogram"
[lane_finding]: ./output_images/6_finding_lane_lines.png "Finding Lane Lines"
[final_lane]: ./output_images/7_final_lane.png "Final lane"
[hud]: ./output_images/8_curvature_and_displacement_hud.png "Heads Up Display"
[video]: ./annotated.mp4 "Annotated Video"


## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the cell #4 of the IPython notebook located in ./pipeline.ipynb. The method in question is called _calibrate_camera().

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function. For convenience, the call to cv2.undistort() was wrapped in the method camera_undistort() in the same code cell in the IPython notebook. This way, the user only needs to provide the image to be undistorted, but the camera matrix and other necessary parameters are stored as fields of the LanePipeline class, to which camera_undistort() belongs.

Undistorting the chessboard images can be seen by comparing the images below. The first one is distorted and the second one is undistorted.

![Distorted Chessboard][distorted_chessboard]
![Undistorted Chessboard][undistorted_chessboard]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![Original Image][original_image]

Here's the undistorted version:

![Corrected Camera Distortion][corrected_camera_distortion]

Note that after the correction, the white car on the right is closer than in the undistorted image. This is reminiscent of the phenomenon described by warning signs on side mirrors: [Objects in mirror are closer than they appear](https://en.wikipedia.org/wiki/Objects_in_mirror_are_closer_than_they_appear).


#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image. The code can be found in cell #4 of the IPython notebook, in the following methods (discussed below):

  - abs_sobel_thresh()
  - sobel_mag_thresh()
  - dir_threshold()
  - combined_threshold()

As the names imply, abs_sobel_thresh() computes the thresholding based on the absolute value of the gradient, either in the x or the y direction. The sobel_mag_threshold() thresholds based on the magnitude of the threashold in both the x and y directions (norm of the two). The dir_threshold() method calculates the direction of the gradient, and calculates the threshold based on that. The combined_threshold() method combines these various other methods into the thresholding pipeline. It is this method that will be described below.

The combined_threshold() method first converts the RGB image to both grayscale and to the HLS colorspace. The grayscale image is used to apply the Sobel transforms to find absolute, magnitude and direction thresholds, while the HLS image is used to determine saturation and lightness thresholds. Saturation is a better measure of whether a pixel is or isn't a lane line, since it's dependent on the intensity of the color, rather than on the exact color. The same goes for lightness and hue in the HLS color space. After multiple experiments, I ended up ignoring the gradient magnitude and direction, and ended up using x- and y-direction Sobel thresholds, as well as saturation and lightness thresholds. The thresholds themselves were selected based on how well the thresholding would preprocess the image for polynomial curve fitting and identifying the lane lines later on.

Here's an example of a thresholded image. Note that the continuous lane line on the left (originally in yellow in the RGB image) and the broken white lane line on the right are clearly visible.

![Thresholded Image][thresholded]


#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

To generate the perspective transform, we first need a perspective matrix from the source and destinations points. These points (referenced by variables src and dst) are provided to the LanePipeline constructor in cell #4 of the IPython notebook. We first save src and dst as fields of the LanePipeline object, then get the transform:

    self.M = cv2.getPerspectiveTransform(self.perspective_src, self.perspective_dst)

We also store the inverse transform so as to be able to return to the original perspective:

    self.Minv = cv2.getPerspectiveTransform(self.perspective_dst, self.perspective_src)

The src and dst points were chosen empirically, based on the idea that farther objects shouldn't be smaller, so the road shouldn't narrow, and instead of looking trapezoidal, should consist of parallel lines demarcating both sides of the road. The src and dst points were chosen to be:

    src = np.float32([[555, 481], [734, 481], [299, 655], [1001, 655]])
    dst = np.float32([[226, 96], [1106, 96], [246, 577], [1106, 577]])

When the pipeline is applied (see the pipeline_apply() method), we can warp the perspective using the M matrix above like so:

    warped = cv2.warpPerspective(thresholded, self.M, (img_size[1], img_size[0]))


Here's an example of how the previously mentioned thresholded image would look like - we intend to get a "bird's-eye view".

![Warped Perspective][warped]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

First, the key is to determine the search area in the image where one can find a cluster of pixels belonging to a lane line. Since lane lines are either straight-vertical or curved, it makes sense to find peaks in the histogram when summing the count of pixels that passed thresholding in the previous step. The peaks should then identify the candidate area in which to start searching. The method calculating the histogram is called find_lines_histogram() and can be found in cell #4 of the IPython notebook, as part of the LanePipeling class.

Here's an example of such a histogram, stacked below the warped thresholded image. As we line them up, we can see that the histogram found a good candidate of where the lane points are concentrated.

![Warped Perspective][warped]

![Histogram][histogram]

After obtaining the histogram for where the points are concentrated, we can start searching for the lane points more finely, by using the sliding window technique. We can start with a rectangle with a height of, say, 1/9th of the image, centered around the pixels which had the highest peaks on the left and right hand sides of the image, and then work our way up the image. As we need to the next set of rows in the image (the next step in the sliding window) and have enough points to get a good estimate of the new center (say 50 points), we can re-center the window for the next step so as to follow the curved lane markers. This is done in the `for window in range(nwindows)` loop in the find_lines_histogram() method mentioned above.

Once we found all the points that seem to belong to lane lines, we can fit a second-degree polynomial using NumPy's polyfit() method. The polyfit() call gives us the coefficients for the various degrees of the polynomial. This is done towards the end of the find_lines_histogram() method, right before returning the results. We can also plot the sliding windows, and the pixels that go through the center of the point cloud for both the left and right lanes, which represent the least-squared polynomial fit. Here's an example of how that might look like. The green boundaries represent the boxes that delineate the boundary for the sliding window point search, and the yellow curves represent the polynomial fits for the left and right lanes.

![Lane Finding][lane_finding]



#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

The radius of the curvature was calculated in the curvature() method of the LanePipeline class in cell #4 of the IPython notebook. Note that by then, we will have fit the second-order polynomial to the points we think represent the left and right lane markers. Let's think about the polynomial first

```math
f(y) = Ay^2+By+C
```
Note that it's specified in terms of y, not x, because the lines can be pretty close to vertical, and so it would be possible to get 2 different values as a function of x, so it's safer to model in terms of y here. 

As explained [here](http://www.intmath.com/applications-differentiation/8-radius-curvature.php), the radius of the curvature can be calculated as follows 

```math
R_curve = [(1 + (2Ay+B)^2)^(3/2)] / |2A|
```

As you can see in the code of the cuvature() method, the calculations assigned to the variables left_curverad and right_curverad achieve precisely that. However, note also that to calculate the radius in real-world measurements (meters) as opposed to on-screen measurements (pixels), we needed to recompute the polynomial coefficients based on a representation that's already in meters rather than pixels. That's the reason why we have the polyfit() calls right before the left_curverad and right_curverad calculations - these fits are based on a rescaling of pixels to meters, at a rate of 30/720 meters per pixel in the y dimension (ym_per_pix) and 3.7/700 meters per pixel in the x direction (xm_per_pix). Given these recomputed polynomial coefficients, we can perform the above-mentioned curve radius calculation (R_curve formula above). We do this calculation for both the left and right lane marker's turning radius. It's important because even though the curves are supposed to be parallel, poor thresholding or other heuristics which lead to identifying the points to fit the polynomials for the two sides of the road could generate different results. A large discrepancy could be an indicator to throw away the sample and for instance use an average of points for polynomial calculation from previous frames, or take other approaches towards recovery.


#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

The lanes are drawn on the original undistorted image in the method called draw_lane(), which belongs to the LanePipeline class, found in cell #4 of the IPython notebok. This method takes the warped image with identified lane lines, fills a polygon between the identified lines (to color the full extent of the road, in this case in green), unwarps the warp perspective back into the original perspective of the undistorted image, and combines the two images so as to color the lane, as an overlay on top of the undistorted image. Here I show two examples, the first one with the lane identified, and the second one with an additional "head-up display" showing the curvature of the left lane marker, the right lane marker, and the displacement from the center of the lane.

![Final Lane][final_lane]

![Final Lane with Curvature Info][hud]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  
