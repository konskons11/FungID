# FungID

[![Downloads](https://img.shields.io/github/downloads/konskons11/FungID/total?style=flat-square)](https://github.com/konskons11/FungID/releases/)

FungID [_(Pouris, J., Konstantinidis, K. et al.)_](https://doi.org/10.3390/pathogens14030242) is a pilot software/application designed to revolutionize the identification of fungal species by leveraging advanced machine learning techniques and chromogenic profiling. This innovative tool utilizes a unique approach, scanning image(s) input by the user for detectable culture plates (i.e. Petri dishes) and analyzing the distinctive color patterns of fungal colonies to accurately classify the examined species. 

Implemented using Python 3.8.5, FungID integrates a [Convolutional Neural Network (CNN)](https://www.researchgate.net/publication/367157330_Understanding_of_Convolutional_Neural_Network_CNN_A_Review) based on the [VGG16](https://github.com/ashushekar/VGG16) architecture, pretrained on the ImageNet dataset, and other libraries such as [tkinter](https://docs.python.org/3/library/tkinter.html) for the Graphical User Interface (GUI), [cv2 (OpenCV)](https://github.com/opencv/opencv) for image processing, [numpy](https://github.com/numpy/numpy) for numerical operations, [h5py](https://github.com/h5py/h5py) for handling HDF5 files, [PIL (Pillow)](https://github.com/python-pillow/Pillow) for image manipulation, and [tensorflow.keras (TensorFlow)](https://github.com/keras-team/keras) for building and training the neural network model. The application features a user-friendly GUI, offering functionalities such as parameter adjustments for efficient culture plate detection, real-time monitoring of training progress, and direct visualization of classification results, thus making it accessible to both researchers and practitioners, regardless of their technical expertise. 

Further development of FungID is expected to be particularly valuable in clinical settings, where prompt and accurate fungal identification is crucial for effective diagnosis and treatment, ultimately contributing to improved patient outcomes and advancing mycological research. 

The FungID algorithm is outlined in the workflow diagram below.
![FungID algorithm](https://i.imgur.com/FqurEAQ.png "FungID algorithm")

## User manual

FungID is a user-friendly application designed to help you classify fungal species efficiently. This manual will guide you through the various features and functionalities of the GUI to ensure you get the most out of the application.

### Installation

FungID operates on Python 3.8.5 and requires the following dependencies:
* h5py==3.7.0
* matplotlib==3.7.0
* numpy==1.22.0
* opencv_python==4.9.0.80
* Pillow==9.4.0
* tensorflow==2.11.1
* Tkinter

_Note: The FungID application will automatically detect and install any missing dependencies._

To launch the FungID application, download the latest FungID Python script release from GitHub, navigate to the directory where the script was downloaded, and execute the script using the following command:

```sh
python FungID.py
```

### Main Interface
Upon launching the application, the main GUI is displayed, which is divided into 2 sections:

TRAIN MODE: For training a model with the images of interest.

TEST MODE: For testing a pre-trained model on the desired images for classification. We highly recommend downloading and using the [latest pre-trained model](https://github.com/konskons11/FungID/blob/main/Models/Ref_model.h5) from our team.

![FungID main screen](https://i.imgur.com/DYFzYqH.png "FungID main screen")

_NOTE: The supported image formats are *.jpg, *.jpeg, *.png, *.bmp, *.tiff, while the supported model format is *.h5_

### TRAIN MODE

![FungID training mode main screen](https://i.imgur.com/qsElKQ9.png "FungID training mode main screen")

A) Mode selection button: Allows users to enable/switch to "TRAIN MODE", while disabling "TEST MODE". 

B) Image(s) directory text field: Displays the path to the input image dataset. 

C) Train Model button: Click to open a file dialog to select the directory of the input image dataset. The input dataset, consisting of images of known fungi species, should be organized in a directory structure, with each subdirectory representing a different species. Once the input image dataset is selected, the model training process is inititated.

D) Progress Bar: Indicates the completion percentage of the training process real-time.

E) Model Performance plot area: Displays training and validation metrics (Accuracy, Loss) after completion of the training process.

F) Model Performance plot buttons: Handles the generated plot like below (buttons explained from left to right)
1. Resets graph to original view
2. Moves backwards to previous view
3. Moves forwards to next view
4. Moves graph accordingly (Left button pans, Right button zooms x/y fixes axis, CTRL fixes aspect)
5. Zooms to the specified rectangular area (x/y fixes axis)
6. Configures subplots
7. Saves the figure as PDF file

G) Output Console: Displays log messages of the training process.

### TEST MODE

![FungID testing mode main screen](https://i.imgur.com/Y6JaAEF.png "FungID testing mode main screen")

H) Mode selection button: Allows users to enable/switch to "TEST MODE", while disabling "TRAIN MODE". 

I) Loaded model directory text field: Displays the path to the input pre-trained model.

J) Load Model button: Click to load a pre-trained model.

K) Parameter Adjustments text field: Displays the current settings of the software's adjustable parameters.

L) Adjust classifier button: Click to open a file dialog for adjusting the software's main settings/parameters.

M) Classified image(s) directory text field: Displays the image(s) directory to be classified.

N) Classify image(s) button: Click to perform image(s) classification.

O) Save Report button: Click to save the generated report in the output log console as text file.

P) Save Image(s) button: Click to save the processed image(s) after completion of classification.

Q) Output Console: Displays log messages of the training process.


### Parameter Adjustments

The user may adjust the following parameters for optimal classification results:
1. dp (Inverse Ratio of the Accumulator Resolution) is the inverse ratio of the accumulator resolution to the image resolution (e.g. if dp=1.0, the accumulator has the same resolution as the original image, but if dp=2.0, the accumulator has half the resolution of the original image).
2. minDist (Minimum Distance Between Circles) specifies the minimum distance between the centers of detected circles (i.e the threshold of detecting multiple nearby circles as separate entities)
3. param1 (Canny Edge Detector Threshold) is the higher threshold passed to the Canny edge detector (the lower one is twice smaller), which is used for edge detection in the image.
4. param2 (Accumulator Threshold for Circle Centers) is the threshold for center detection in the accumulator (The smaller it is, the more false circles may be detected. The larger it is, the more accurate the circle detection will be)
5. minRadius (Minimum Circle Radius) defines the minimum radius of the circles to be detected.
6. maxRadius (Maximum Circle Radius) defines the maximum radius of the circles to be detected.

The dafault values for these parameters are:
* dp=1.0
* minDist=100
* param1=200
* param2=100
* minRadius=0
* maxRadius=0

An indicative example of parameter adjustment is presented below:
![FungID parameter adjustment example](https://i.imgur.com/hexEQrZ.jpg "FungID parameter adjustment example")

_NOTE: The detected circular regions are outlined in bright green circumferences with their center points highlighted red, while their corresponding names are annotated with blue font on the right cide of the detected circular regions._

### Software key features

**TRAIN MODE**

_Custom Model Training:_ Input the desired image(s) and utilize built-in features such as data augmentation, early stopping, and model checkpointing to train and ensure the best-performing version of your model is saved.

_Data Augmentation:_ ImageDataGenerator is utilized to enhance the training dataset with transformations such as rotation, scaling, and flipping, improving model robustness.

_Real-Time Monitoring and Performance Metrics:_ The GUI displays real-time training progress and plots model performance metrics, including accuracy, loss, validation accuracy, and validation loss, with an option to be exported as graphs by the user.

**TEST MODE**

_Parameter Adjustments for Fungal Species Detection:_ The user is able to detect multiple culture plates within a single input image and also tweak various software parameters in order to optimize fungal species detection.

_Image Processing:_ Preprocesses images using OpenCV techniques, including Gaussian blurring and Hough Circle Transform for detecting the circular regions of culture plates with the examined fungal species.

_Comprehensive Results:_ Visualize detected circular regions of culture plates and display classified results with predicted species and confidence levels. (Options to save classification reports and processed images are also provided)
