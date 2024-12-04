# FungID

[![Downloads](https://img.shields.io/github/downloads/konskons11/FungID/total?style=flat-square)](https://github.com/konskons11/FungID/releases/)

FungID is a pilot software/application designed to revolutionize the identification of fungal species by leveraging advanced machine learning techniques and chromogenic profiling. This innovative tool utilizes a unique approach, scanning image(s) input by the user for detectable culture plates (i.e. Petri dishes) and analyzing the distinctive color patterns of fungal colonies to accurately classify the examined species. 

Implemented using Python 3.8.5, FungID integrates a Convolutional Neural Network (CNN) based on the VGG16 architecture, pretrained on the ImageNet dataset, and other libraries such as tkinter for the Graphical User Interface (GUI), cv2 (OpenCV) for image processing, numpy for numerical operations, h5py for handling HDF5 files, PIL (Pillow) for image manipulation, and tensorflow.keras (TensorFlow) for building and training the neural network model. The application features a user-friendly GUI, offering functionalities such as parameter adjustments for efficient culture plate detection, real-time monitoring of training progress, and direct visualization of classification results, thus making it accessible to both researchers and practitioners, regardless of their technical expertise. 

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
Upon launching the application, the main interface is displayed, which is divided into 2 main sections:

![FungID main screen](https://i.imgur.com/DYFzYqH.png "FungID main screen")

TRAIN MODE: For training a model with the images of interest.

TEST MODE: For testing a pre-trained model on the desired images for classification.

_NOTE: The supported image formats are *.jpg, *.jpeg, *.png, *.bmp, *.tiff, while the supported model format is *.h5_

**TRAIN MODE**

![FungID training mode main screen](https://i.imgur.com/qsElKQ9.png "FungID training mode main screen")

Load Training Data: Click the button to select and load the folder containing your training images.

Start Training: Begin the training process. The GUI will display a progress bar to show the training progress in real-time.

Early Stopping & Checkpointing: These features are automatically enabled to prevent overfitting and save the best performing model.

**TEST MODE**

![FungID testing mode main screen](https://i.imgur.com/Y6JaAEF.png "FungID testing mode main screen")

Load Model: Select a pre-trained model file to load into the application.

Load Images: Select and load the image(s) you want to classify.

Classify Images: Click the button to start the classification process. The results will be displayed in the GUI, showing the predicted species and associated probabilities.

**Parameter Adjustments**

Circle Detection: Adjust parameters for detecting circular regions in images, such as minRadius, maxRadius, param1, and param2.

Image Augmentation: Configure settings for augmenting training images, including rotation range, width shift range, height shift range, and horizontal flip.


Results Display

Log Console: View detailed logs and updates on the training or testing process.

Visualization: Visualize detected circular regions in images, with green circumferences and red center points highlighted alongside their name in blue text as in the format Cn (where n is the number of the detected circular region). Classified results will be displayed with the predicted species and confidence levels.

Save Reports: Options to save classification reports and processed images.

### Software key features

**TRAIN MODE**

_Custom Model Training:_ Input the desired image(s) and utilize built-in features such as data augmentation, early stopping, and model checkpointing to train and ensure the best-performing version of your model is saved.

_Data Augmentation:_ ImageDataGenerator is utilized to enhance the training dataset with transformations such as rotation, scaling, and flipping, improving model robustness.

_Real-Time Monitoring and Performance Metrics:_ The GUI displays real-time training progress and plots model performance metrics, including accuracy, loss, validation accuracy, and validation loss, with an option to be exported as graphs by the user.

**TEST MODE**

_Parameter Adjustments for Fungal Species Detection:_ The user is able to detect multiple culture plates within a single input image and also tweak various software parameters in order to optimize fungal species detection.

_Image Processing:_ Preprocesses images using OpenCV techniques, including Gaussian blurring and Hough Circle Transform for detecting the circular regions of culture plates with the examined fungal species.

_Comprehensive Results:_ Visualize detected circular regions of culture plates and display classified results with predicted species and confidence levels. (Options to save classification reports and processed images are also provided)

Troubleshooting
No Detected Circles: If the application fails to detect circles in images, adjust the circle detection parameters (e.g., minRadius, maxRadius, param1, param2) and retry.

Slow Performance: Ensure your system meets the recommended hardware requirements. Training and classification can be computationally intensive.

Loading Issues: Double-check the paths to your image and model files to ensure they are correctly specified.
 
