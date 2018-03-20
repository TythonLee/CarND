# **Behavioral Cloning** 

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of two types of convolution neural networks. The first has 5x5 filter sizes and depths between 24 and 48 (model.py lines 75-77) . The second has 3x3 filter sizes and depth 64(model.py lines 78-79).

The model includes RELU layers to introduce nonlinearity (model.py lines 75-79), and the data is normalized in the model using a Keras lambda layer (model.py line 69). 

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py line 82). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (model.py lines 51-58, 88-89). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 87).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I only used a center lane driving dataset to train and test our model. The dataset is collected by the simulator and it has only anticlockwise rotation routine. The number of training graph is 6750.

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to extract features from pictures and feed them to a regression model to fit the driving angles.

My first step was to use a convolution neural network model similar to the Lenet. I thought this model might be appropriate because these cells act as local filters over the input space and are well-suited to exploit the strong spatially local correlation present in natural images. This is vital for our task to capture the image features. However, traditional Lenet is not complex enough to model these non-linear transitions of spaces. We propose to add several CNN layers to the Lenet so that the model has more capacity to capture the features.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set (model.py lines 51-58, 88-89). I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model by adding a dropout layer between Dense layers so that the model will be more robust to the whole dataset.

Then I add several Dense layers on the top of the model, and the output of the last layer is the angle that we predict.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track. For example, the turning that just over the bridge and some sudden turns. To improve the driving behavior in these cases, I increased the number of training epochs so that the parameter can fit the dataset well. Experimental results show that the driving accident can be avoided when training epochs increasing.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 67-85) consisted of a convolution neural network with the following layers and layer sizes, which is visualized in the architecture illustration.
![cnn-architecture-624x890-1.png](http://upload-images.jianshu.io/upload_images/6982894-81af4213543284a5.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)




#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded one lap on track one using center lane driving. Here is an example image of center lane driving:

![center_2017_10_30_10_25_23_859.jpg](http://upload-images.jianshu.io/upload_images/6982894-bb5eda3fcaa8d9a7.jpg?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)


I also recorded the vehicle recovering from the left side and right sides of the road back to center but they are not adopted by our model. These images are showing as follows:

![left_2017_10_30_10_25_23_859.jpg](http://upload-images.jianshu.io/upload_images/6982894-af13a8bc0d67941a.jpg?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

![right_2017_10_30_10_25_23_859.jpg](http://upload-images.jianshu.io/upload_images/6982894-3c55d15432ddb6ec.jpg?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)


To augment the data sat, I also flipped images and angles thinking that this would make the model more robust. For example, here is an image that has then been flipped around y-axis:

![test.png](http://upload-images.jianshu.io/upload_images/6982894-3ad7062e8f8e0eb3.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

![test_flip.png](http://upload-images.jianshu.io/upload_images/6982894-924eeba31038a57b.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)


After the collection process, I had 6750 number of data points. I then preprocessed this data by converting them to YUV color space.
```sh
new_img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
```
It is worth noting that this method is the same used in drive.py, except this version uses BGR to YUV and drive.py uses RGB to YUV (due to using cv2 to read the image here, where drive.py images are received in RGB).
```sh
new_img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
```


I finally randomly shuffled the dataset and put 20% of the data into a validation set (model.py lines 51-58). 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 20 as evidenced by the experimental results. I used an adam optimizer so that manually training the learning rate wasn't necessary.
