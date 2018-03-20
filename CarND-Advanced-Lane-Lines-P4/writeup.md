## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

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

[image1]: ./examples/undistort_output.png "Undistorted"
[image2]: ./test_images/test1.jpg "Road Transformed"
[image3]: ./examples/binary_combo_example.jpg "Binary Example"
[image4]: ./examples/warped_straight_lines.jpg "Warp Example"
[image5]: ./examples/color_fit_lines.jpg "Fit Visual"
[image6]: ./examples/example_output.jpg "Output"
[video1]: ./project_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---


### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the first code cell of the IPython notebook located in "./examples/Advanced-Lane_Lines.ipynb" .

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 


Original image | Calibrated and undistorted image
------------ | -------------
![](http://upload-images.jianshu.io/upload_images/6982894-e20308b770b2c05a.jpg?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240) | ![](http://upload-images.jianshu.io/upload_images/6982894-0026570e14ea9000.jpg?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)


### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images. First, I compute the camera calibration using chessboard images. After that, we should now have objpoints and imgpoints needed for camera calibration. So, we try to calibrate, calculate distortion coefficients, and test undistortion on an image! The experiment result is shown below:

Original image | Calibrated and undistorted image
------------ | -------------
![2-test1.jpg](http://upload-images.jianshu.io/upload_images/6982894-bbbc5152f532442f.jpg?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240) | ![2-test1_undist.jpg](http://upload-images.jianshu.io/upload_images/6982894-99c1408d4e5ac269.jpg?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)


#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I have tried several combinations of color and gradient. Finally, I found that the absolute sobel threshold in the orientation x  and S channel threshold in HLS can achieve a promising result. So, I used a combination of `abs_sobel_thresh(img, orient='x', thresh_min=20, thresh_max=100)` and `color_threshold(img, thresh=(170,200))`  to generate a binary image (thresholding steps at second part in `Advanced-Lane_Lines.ipynb`).  Here's an example of my output for this step.  (note: this is not actually from one of the test images)

Original undistorted image | Gradient x image
------------ | -------------
![undst_img.jpg](http://upload-images.jianshu.io/upload_images/6982894-b612f061f4f52c39.jpg?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240) | ![sxbinary.jpg](http://upload-images.jianshu.io/upload_images/6982894-52ad1ec5ade97bbe.jpg?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

Gradient S image | Combine gradient x and S image
------------ | -------------
![s_binary.jpg](http://upload-images.jianshu.io/upload_images/6982894-ef79f46f2228a52f.jpg?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240) | ![combined_binary.jpg](http://upload-images.jianshu.io/upload_images/6982894-c51acc222529d656.jpg?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)


#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `loadline_unwarp()`, which appears in third part in `Advanced-Lane_Lines.ipynb`.  The `loadline_unwarp()` function takes as inputs an image (`img`), as well as the image shape of gray version (`gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)`).  I chose the hardcode the source and destination points in the following manner:

```python
    offset = 10
    img_size = (imshape[1], imshape[0])#(1280,720)

    left_bottom = (0.10*imshape[1],0.95*imshape[0])
    left_top = (0.4*imshape[1], 0.65*imshape[0])
    right_top = (0.6*imshape[1], 0.65*imshape[0])
    right_bottom = (1*imshape[1],0.95*imshape[0])
    
    src = np.float32([ left_top, right_top, right_bottom, left_bottom])
    
    dst = np.float32([[offset, offset], [img_size[0]-offset, offset], 
                                 [img_size[0]-offset, img_size[1]-offset], 
                                 [offset, img_size[1]-offset]])
```
One important thing need to note is that the direction of four points in src and dst should be the same.
This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 512,   468    | 10,    10        | 
| 768,   4680      | 1270,    10      |
| 1280,   684     | 1270,   710      |
| 128,   684      | 10,   710        |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

Masked image | Combine gradient x and S image | Warped image
------------ | ------------- |---------------
![](http://upload-images.jianshu.io/upload_images/6982894-97625beb2c8a207a.jpg?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)|![](http://upload-images.jianshu.io/upload_images/6982894-17d52d0ea9758afa.jpg?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)|![](http://upload-images.jianshu.io/upload_images/6982894-76e4ca554ad33934.jpg?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)



#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial.

Then I did some other stuff and fit my lane lines with a 2nd order polynomial kinda like this:

f(y) = Ay^2 + By + C

In detail, we detect lane pixels and slide window polyfit to find the lane boundary. The sliding window function is defined in fourth part in `Advanced-Lane_Lines.ipynb` as `Sliding_Window_Search(binary_warped)`. Then, we further illustrate the search window area.
The sliding result and illustration result is presented as follows:

Sliding window result | Illustration result
------------ | ------------- 
![](http://upload-images.jianshu.io/upload_images/6982894-2985bcc84bd36167.jpg?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240) | ![](http://upload-images.jianshu.io/upload_images/6982894-6f45362e97ba170f.jpg?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this by defining a function `calc_curv_rad_and_center_dist()` , which can be found in fifth part in `Advanced-Lane_Lines.ipynb` 

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.
By using the coefficient `Minv`, we can transform the lines back to the original image. We first define a function `draw_lane()` to draw the lanes on the image, then we put the curvature and positon data on to the same image. I implemented this step in sixth part in `Advanced-Lane_Lines.ipynb` .  Here is an example of my result on a test image:

Original image with mask | Original image with mask and data
------------ | ------------- 
![OriginalImg_with_mask.jpg](http://upload-images.jianshu.io/upload_images/6982894-9c4d36296c293c00.jpg?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240) | ![OriginalImg_with_data.jpg](http://upload-images.jianshu.io/upload_images/6982894-c236f16849125874.jpg?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)


---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./processed_project_video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  
First, I think the smoother is not so good as I expected. It can be used to avoid the jitting problem, but the feature extraction from an original frame is more important. I hope that my model can be more efficient to detect the lanes.
Second, maybe some deep learning approaches such as CNN can be used to solve this problem.

