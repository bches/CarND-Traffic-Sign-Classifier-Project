# **Traffic Sign Recognition** 

## Writeup

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

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

[image1]: ./class_labels_test.png "Class Labels Histogram, Test Data Set"
[image1a]: ./class_labels_train.png "Class Labels Histogram, Training Data Set"
[image1b]: ./class_labels_valid.png "Class Labels Histogram, Validation Data Set"
[image3]: ./class_labels_train_aug.png "Class Labels Histogram, Training Data Set (Augmented)"
[image4]: ./german_traffic_signs_1.png "Traffic Sign 1"
[image5]: ./german_traffic_signs_2.png "Traffic Sign 2"
[image6]: ./german_traffic_signs_3.png "Traffic Sign 3"
[image7]: ./german_traffic_signs_4.png "Traffic Sign 4"
[image8]: ./german_traffic_signs_5.png "Traffic Sign 5"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/bches/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used python to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34687
* The size of the validation set is 4387
* The size of test set is 12622
* The shape of a traffic sign image is 32x32
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is a histogram of the frequency of occurence of the class labels in the test data set.

![Class Labels Histogram, Test Data Set][image1]

Similarly, the histograms for the training and validation sets are also provided.  

![Class Labels Histogram, Training Data Set][image1a]

![Class Labels Histogram, Validation Data Set][image1b]


The general shape of the training data set seems to match that of the test data set.  However, the validation data set seems to have a different balance of class labels than the training or test data sets.


And here are the individual counts of each class in the training data set (in order from most occurences to fewest occurences):

The red line is the average count of a class label in the test data set.  The green dashed lines are one standard deviation above and below the average.  There are a couple of classes that are clearly underrepresented in the test data set.

Most Common classes:
(2, 2010), (1, 1980), (13, 1920), (12, 1890), (38, 1860), (10, 1800), (4, 1770), (5, 1650), (25, 1350), (9, 1320), (7, 1290), (3, 1260), (8, 1260), (11, 1170), (35, 1080), (18, 1080), (17, 990), (31, 690), (14, 690), (33, 599), (26, 540), (15, 540), (28, 480), (23, 450), (30, 390), (16, 360), (34, 360), (6, 360), (36, 330), (22, 330), (40, 300), (20, 300), (39, 270), (21, 270), (29, 240), (24, 240), (41, 210), (42, 210), (32, 210), (27, 210), (37, 180), (19, 180), (0, 180)



### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I stripped out some images from the training data that were all black.  The problem with these images is that are mislabeled as certain signs and I did not want my neural network training on that.

As a last step, I normalized the image data because this neural network expects 0 mean data between -1 and 1.  I did not get the mean to exactly 0 but got it closer and did get it in between -1 and 1.

I decided to generate additional data because there were a few classes in the training set that were underrepresented as shown above.

To add more data to the the data set, I iterated through each label that was underrepresented and concatenated onto the data set a number of copies of an image corresponding to that label.  The number of copies to add was calculated as the number of copies it would take to bring the most underrepresented labels up to what was prevoiusly the average count for the data set. 

![Class Labels Histogram, Training Data Set (Augmented)][image3]


The difference between the original data set and the augmented data set is the distribution of class labels.


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 				|
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x16 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16 				|
| Fully connected		| input: 400x1, output:n_classes\*20x1			|
| RELU					|												|
| Fully connected		| input: n_classes\*20x1, output:n_classes\*10x1	|
| RELU					|												|
| Fully connected		| input: n_classes\*10x1, output:n_classes\*1x1	|
 
 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used 30 batches of 128 images each.  I used the Adam optimizer and kept the learning rate at 0.001.  The optimizer minimized the cross-entropy between the label of the given image and the softamx of its output from the neural network architecture described in the previous section.


#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 99.9%
* validation set accuracy of 94.1%
* test set accuracy of 93.2%

I chose the LeNet architecture and used it pretty much as is with some minor tweaks to the fully connected layers as well as some pruning and augmentation of the training data.  My choices on the architecture are described in the following paragraph.

I have two convolutional layers followed by three fully connected layers.  The two convolutional layers are both 5x5 convolutions with 2x2 max pooling to reduce the size of the image for successive stages.  The fully connected layers treat the image as flat.  The first fully connected layer actually expands the size of the data from 400x1 to 860x1 (which is parameterized as 20 times the number of classes, 43).  The second fully connected layer drops this down to 430x1 (10 x number of class labels).  The final fully connected layer outputs class labels of the correct size, n_classes x 1.  I found that keeping fully connected layers before reducing it to the final output worked better than trying to make the flattened output of the fully connected layers smaller sooner, with negligible impact on run time for this project.

I tried dropout, but decided not to use it in the final architecture as I was able to achieve 93% test accuracy with the methods described above.
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The first image might be difficult to classify because ...

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop sign   									| 
| U-turn     			| U-turn 										|
| Yield					| Yield											|
| 100 km/h	      		| Bumpy Road					 				|
| Slippery Road			| Slippery Road      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of ...

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .60         			| Stop sign   									| 
| .20     				| U-turn 										|
| .05					| Yield											|
| .04	      			| Bumpy Road					 				|
| .01				    | Slippery Road      							|


For the second image ... 

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


