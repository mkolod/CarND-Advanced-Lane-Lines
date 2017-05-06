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



Then I did some other stuff and fit my lane lines with a 2nd order polynomial kinda like this:
Histogram 

![Histogram][histogram]

Lane finding

![Lane Finding][lane_finding]



#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in lines # through # in my code in `my_other_file.py`

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines # through # in my code in `yet_another_file.py` in the function `map_lane()`.  Here is an example of my result on a test image:

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
