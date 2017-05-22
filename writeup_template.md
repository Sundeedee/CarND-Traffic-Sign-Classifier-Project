#**Traffic Sign Recognition** 

##Writeup Template

###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

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

[image1]: ./Test_output/dataset_visual.jpg "Visualization"
[image2]: ./Test_output/gray_preprocess.jpg "Grayscaling"
[image3]: ./Test_output/fake_img.jpg "Random Noise"
[image4]: ./Online_testing/10x.png "Traffic Sign 1"
[image5]: ./Online_testing/1x.png "Traffic Sign 2"
[image6]: ./Online_testing/2x.png "Traffic Sign 3"
[image7]: ./Online_testing/3x.png "Traffic Sign 4"
[image8]: ./Online_testing/5x.png "Traffic Sign 5"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set and identify where in your code the summary was done. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

The code for this step is contained in the second code cell of the IPython notebook.  

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of test set is 12630
* The shape of a traffic sign image is 32x32x3
* The number of unique classes/labels in the data set is 43

####2. Include an exploratory visualization of the dataset and identify where the code is in your code file.

The code for this step is contained in the third code cell of the IPython notebook.  

Here is an exploratory visualization of the data set. It is a bar chart showing how the data ...

![alt text][image1]

###Design and Test a Model Architecture

####1. Describe how, and identify where in your code, you preprocessed the image data. What tecniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc.

The code for this step is contained in the fourth code cell of the IPython notebook.

As a first step, I decided to convert the images to grayscale because ...

Here is an example of a traffic sign image before and after grayscaling.

![alt text][image2]

As a last step, I normalized the image data because ...

####2. Describe how, and identify where in your code, you set up training, validation and testing data. How much data was in each set? Explain what techniques were used to split the data into these sets. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, identify where in your code, and provide example images of the additional data)

The code for splitting the data into training and validation sets is contained in the fifth code cell of the IPython notebook.  

To cross validate my model, I randomly split the training data into a training set and validation set. I did this by shuffling the training and test set

My final training set had X number of images. My validation set and test set had Y and Z number of images.

The sixth code cell of the IPython notebook contains the code for augmenting the data set. I decided to generate additional data because some training images for some signs are not enough comparing to other signs. To add more data to the the data set, I used the following techniques because they just simply change the shape or angle of the original image. We can think in that way, the fake datas are collected from different angle, brightness or position from the road. So it would not be overfitted caused by fake datas. 

Here is an example of an original image and an augmented image:

![alt text][image3]

The difference between the original data set and the augmented data set is the following:
1. The fake data is shifted to the right bottom corner a little bit.
2. The fake data is rotated for a few degrees randomly.
3. The fake data get a perspective transform like changing the view point.


####3. Describe, and identify where in your code, what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

The code for my final model is located in the 10th cell of the ipython notebook. 

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 gray image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x10 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x10 				|
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x32	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x32 				    |
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 1x1x800	|
| RELU					|												|
| CONCAT				| Input: 5x5x32,1x1x800 Ouput:1x1600			|
| DROPOUT				| 0.8											|
| Fully connected		| 1600,800   									|
| RELU					|												|
| Fully connected		| 800,43										|
| SOFTMAX				|												|
|						|												|


####4. Describe how, and identify where in your code, you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

The code for training the model is located in the 10th cell of the ipython notebook. 

To train the model, I used an :
EPOCHS = 60
BATCH_SIZE = 250
rate = 0.0005
softmax_cross_entropy_with_logits
AdamOptimizer
####5. Describe the approach taken for finding a solution. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

The code for calculating the accuracy of the model is located in the ninth cell of the Ipython notebook.

My final model results were:
* training set accuracy of 1.000
* validation set accuracy of 0.961
* test set accuracy of 0.931

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?

    I chose to use lenet network first, but it can only reach 91% validation accuracy by tunning the parameters. Because it was easy to implement and I want to see the limit of this network.
* What were some problems with the initial architecture?
    It is weak on detecting more deep feature because of the convolutional layers are too small.
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to over fitting or under fitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.

    I picked a network structure in the article recommended by Udaicty, and I added a third convolutional layer and a parrellel layer from layer2. Then concat them into one full connected layer, also a dropout layer after this, because it has 1600 features.
* Which parameters were tuned? How were they adjusted and why?
    
    I tuned ECHOS, Batch_size, learning rate, and keep_prob. Because it was overfitting at first caused by high ECHOS. And the accuracy was dumping caused by huge learning rate. The keep_prob help making the network avoid overfitting and get more robust.
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

    The third convolutional layer is very important. Because it combined the whole image information into one value and the depth of this layer is 800 and this give it 800 different features to classify the input image. The dropout layer help making the network avoid overfitting and get more robust.
If a well known architecture was chosen:
* What architecture was chosen?
* Why did you believe it would be relevant to the traffic sign application?
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
 

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The first image might be difficult to classify because the '60' are similar to other speed limit, only difference is the first digit

The Second image is similar to general caution

The Third image is 30km/h same as the first one

The fourth image shape is related to other traffic signs

The fifth image arrow is similar to other traffic signs

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. Identify where in your code predictions were made. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

The code for making predictions on my final model is located in the tenth cell of the Ipython notebook.

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 60km/h          		| 60km/h    									| 
| Right-of-way at the next intersection| Right-of-way at the next intersection 										|
| 30km/h				| 30km/h											|
| priority road    		| priority road					 				|
| Keep right			|Keep right          							|


The model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 100%. This compares favorably to the accuracy on the test set of 0.931

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction and identify where in your code softmax probabilities were outputted. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.00        			| 60km/h   									| 

For the second image ... 
| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.00        			| Right-of-way at the next intersection         |

For the third image ...
| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.00        			| 30km/h     									|

For the fourth image ...
| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.00        			| priority road									|

For the fifth image ...
| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.00        			| Keep right   									|