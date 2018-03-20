
# **Traffic Sign Recognition** 

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/visualization.jpg "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./examples/placeholder.png "Traffic Sign 1"
[image5]: ./examples/placeholder.png "Traffic Sign 2"
[image6]: ./examples/placeholder.png "Traffic Sign 3"
[image7]: ./examples/placeholder.png "Traffic Sign 4"
[image8]: ./examples/placeholder.png "Traffic Sign 5"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/TythonLee/CarND/blob/master/Project2%3A%20Traffic-Sign-Classifier-Project/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the numpy library to calculate summary statistics of the traffic signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. 
First, it is an overview of 43  traffic signs.

![images_statisticsa.png](http://upload-images.jianshu.io/upload_images/6982894-b93adac15f8f4486.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1000)

Then, It is a bar chart showing how the data distribution in terms of each category. Axis x indicates number of each sign. Axis y shows 43 kinds of images. From the statistics chart, we can tell the imbalance of dataset is a big problem.

![images_statistics.png](http://upload-images.jianshu.io/upload_images/6982894-1e8e643dd5e4717a.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1000)


### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step,  I decided to convert the images to grayscale because some features can be captured in this dimension and it cost less than original images with 3 channels.

Here is an example of a traffic sign image before and after grayscaling.

Original| Grayscale
------------ | -------------
![images_original.png](http://upload-images.jianshu.io/upload_images/6982894-0f4340b053046873.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240) | ![images_gray.png](http://upload-images.jianshu.io/upload_images/6982894-9ad40b94788eb111.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)


As a last step, I normalized the image data because the original value is integers between 0 to 255. It is not easy for the deep learning method to model it. 
``` python
def normalization(input_features):
    return (input_features - 128.0)/128.0

def grayscale(rgb):
    r, g, b = rgb[:,:,:,0], rgb[:,:,:,1], rgb[:,:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    bb = list(gray.shape)
    bb.append(1)
    gray = gray.reshape(bb)
    return gray
```
I decided to generate additional data because the dataset is so imbalance that makes model hard to train.
Another reason is to enhance the robust of model.
To add more data to the the data set, I used the following techniques:
``` python
def regenerate_data(X_train, y_train):
    print('Regenerating data...')
    import scipy.ndimage
    X_all = X_train
    y_all = y_train
    # Generate additional data for underrepresented classes
    print('Generating additional data...')
    angles = [-5, 5, -10, 10, -15, 15, -20, 20]

    class_count = np.bincount(y_train)
    for i in range(len(class_count)):
        input_ratio = min(int(np.max(class_count) / class_count[i]) - 1, len(angles) - 1)
        if input_ratio <= 1:
            continue
        new_features = []
        new_labels = []
        mask = np.where(y_train == i)
        for j in range(input_ratio):
            for feature in X_train[mask]:
                new_features.append(scipy.ndimage.rotate(feature, angles[j], reshape=False))
                new_labels.append(i)
        X_all = np.concatenate((X_all, np.asarray(new_features)), axis=0)#np.append(X_train, new_features, axis=0)
        y_all = np.concatenate((y_all, np.asarray(new_labels)), axis=0)#np.append(y_train, new_labels, axis=0)
    print('Regenarating data completed. Number of total samples', len(y_all))
    return X_all, y_all
```
For different kind, we generate different number of new data according to their ratio to maximum of all categories. After this processing, each kind of image will be fair to each other.
Here is an example of an original image and an augmented image:

![original.png](http://upload-images.jianshu.io/upload_images/6982894-0cbb8250d46aeac1.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

![augments.png](http://upload-images.jianshu.io/upload_images/6982894-abb6dd7e58458ec9.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)



The difference between the original data set and the augmented data set is the following.
The augments are rotated images of originals. They have eight-angle versions that are [-5, 5, -10, 10, -15, 15, -20, 20 ] .
After regeneration of images, the data set is coming to balance. The statistics of each category is the showing below:

![after_statistics.png](http://upload-images.jianshu.io/upload_images/6982894-a6caf5943ebe5d75.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Gray image   							| 
| Convolution 5x5    	| 1x1 stride, same padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 				|
| Convolution 5x5	    | 1x1 stride, same padding, outputs 10x10x16      							|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16				|
| Flatten                  | Output = 400
| Fully connected    | Output = 256									|
| Dropout layer        | keep_prob = 0.5
| Fully connected    | Output = 43									|
| Softmax				| Output = 43	      									|

 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used a learning rate = 0.001, a batch size = 256. I apply Adam_optimizer as an optimizer and cross_entropy as loss function. More importantly, I set an auto iteration mechanics to get the optimal epochs. It works with a list, which stores the highest value of accuracy. If the iteration accuracy keeps smaller than the highest value for a patience ( we set it to 10) times, Then, the iteration break and we get the best result.
``` python
from sklearn.utils import shuffle
import time
start_time = time.time()
print(rate)
EPOCHS = 100
BATCH_SIZE = 128*2
results = []
patients = 10
cnt = 0
save_file = './train_model.ckpt'

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    num_examples = len(X_train)
    
    print("Training...")
    print()
    for i in range(EPOCHS):
        X_train, y_train = shuffle(X_train, y_train)
        for offset in range(0, num_examples, BATCH_SIZE):
            end = offset + BATCH_SIZE
            batch_x, batch_y = X_train[offset:end], y_train[offset:end]
            sess.run(training_operation, feed_dict={x: batch_x, y: batch_y})
            
        validation_accuracy = evaluate(X_valid, y_valid)
        print("EPOCH {} ...".format(i+1))
        print("Validation Accuracy = {:.3f}".format(validation_accuracy))
        print()
        
        results.append(validation_accuracy)
        max_value = max(results)
        if validation_accuracy<max_value:
            cnt = cnt+1
        else:
            cnt = 0
        if cnt>patients or i==EPOCHS-1:
            print('Best validation accuracy is:', max_value)
            break
    saver.save(sess, save_file)
    print("Model saved")
print('time cost: ', time.time()-start_time)
```
#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 0.997
* validation set accuracy of 0.936
* test set accuracy of 0.921

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
I tried Lenet as a basic architecture for our task. Because this model is designed to recognise hand-writing numbers. Generally, it is used for image classification. I think it will work on our dataset.

* What were some problems with the initial architecture?
Overfitting problem and nodes in hidden layer are not enough.

* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
Add dropout layer and full connecting layer. To fit the data set well and avoid overfitting.
* Which parameters were tuned? How were they adjusted and why?
batch_size is changed from 128 to 256, because it trains too slow. 
The full connected layer which has 120 outputs, I adjusted it to 256.

* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?
I think the convolution layer is a brilliant design. Because it captures the features from each cluster of pixels. It converts the image into different spaces so the classifier can recognize it efficiently.
A dropout layer can drop connections randomly between two layers, which improves the robustness of model and avoid overfitting.


 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

|30 km/h |Priority road |Stop | Road work |Ahead only |
|:---------------------:|:---------------------------------------------:| -----------:|-----------:|-----------:|
|![1-Speed limit(30kmph).JPG](http://upload-images.jianshu.io/upload_images/6982894-445a0a8890033203.JPG?imageMogr2/auto-orient/strip%7CimageView2/2/w/240) | ![12-Priority road.jpg](http://upload-images.jianshu.io/upload_images/6982894-2251eec7fb6e3ab2.jpg?imageMogr2/auto-orient/strip%7CimageView2/2/w/240) | ![14-Stop.jpg](http://upload-images.jianshu.io/upload_images/6982894-85f513348a5a8491.jpg?imageMogr2/auto-orient/strip%7CimageView2/2/w/240) | ![25-Road work.jpg](http://upload-images.jianshu.io/upload_images/6982894-810deb8ecd864290.jpg?imageMogr2/auto-orient/strip%7CimageView2/2/w/240) | ![35-Ahead only.jpg](http://upload-images.jianshu.io/upload_images/6982894-02ff9c0dd8a16e47.jpg?imageMogr2/auto-orient/strip%7CimageView2/2/w/240)|


The last image might be difficult to classify because the sign occupies a small part of the whole image. It is difficult for the model to identify the sign among so much noise.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 30 km/h      		        | Keep right   									| 
| Priority road    		| Priority road 										|
| Stop Sign      		| Keep right    									| 
| Road work	      		| Road work					 				|
| Ahead only			| Road narrows on the right    							|


The model was able to correctly guess 2 of the 5 traffic signs, which gives an accuracy of 40%. This compares badly to the accuracy on the test set. The reason why it performs worse, I think, is the pictures has noise. The images in training are only traffic signs occupied the whole picture. Our downloaded images have other backgrounds. So, the model does not perform well.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is below:
``` python
labels_pred = tf.nn.softmax(logits)
# feed_dict_new = {features:X_newdata}
top5 = tf.nn.top_k(labels_pred, 5)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver.restore(sess, save_file)
    predictions = sess.run(labels_pred,feed_dict={x:X_newdata })
    top5_pred = sess.run([labels_pred, top5], feed_dict={x:X_newdata })
print(top5_pred[1])
``` 
For the second and fourth images, the model is relatively sure that these are Priority road and Road work sign (probability of 0.9999 and 0.72 ), and the images do contain those signs. The top five soft max probabilities were
Priority  road

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0        			| Priority road  									| 
| .0    				| Speed limit (30km/h) 										|
| .0					| End of all speed and passing limits					|
| .0	      			| Bumpy Road					 				|
| .0				    | Speed limit (50km/h)      							|

Road work 

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .72         			| Road work  									| 
| .27     				| Wild animals crossing										|
| .004					| Double curve											|
| .004	      			| Beware of ice/snow					 				|
| .002				    | Bicycles crossing     							|

For the other three images, the model predicted them into wrong categories.  

Ahead only 

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .88         			| Road narrows on the right  									| 
| .06     				| Pedestrians										|
| .05					| Road work										|
| .004	      			| Dangerous curve to the right				 				|
| .006				    | Double curve     							|

30 km/h 

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0         			| Keep right  									| 
| .0     				| Go straight or right										|
| .0					| No passing											|
| .0	      			| Priority road					 				|
| .0				    | Dangerous curve to the right     							|

Stop  

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .99        			| Go straight or right									| 
| .01    				| Stop										|
| .0					| Speed limit (60km/h)											|
| .0	      			| Road work					 				|
| .0				    | Pedestrians    							|

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?
