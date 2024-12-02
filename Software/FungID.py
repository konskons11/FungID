import os
import subprocess
import sys

# Function to check if a package is installed
def check_and_install(package):
    try:
        __import__(package)
    except ImportError:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        except subprocess.CalledProcessError as e:
            print(f"Failed to install {package}: {e}")

# Check and install necessary packages
required_packages = [
    "opencv-python==4.9.0.80", "numpy==1.22.0", "h5py==3.7.0", "Pillow==9.4.0", "tensorflow==2.11.1", "matplotlib==3.7.0"
]

for package in required_packages:
    check_and_install(package)

# Now import the packages
import tkinter as tk
from tkinter import filedialog, Label, Button, Text, messagebox, ttk
import cv2
import numpy as np
import h5py
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, Callback
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from threading import Thread, Event

# Constants
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# Global variables
model = None
class_labels = []
stop_event = Event()  # Global event to stop training
training_thread = None
testing_thread = None
processed_images = [] # To store processed images for saving
default_hough_params = {'dp': 1.0, 'minDist': 100, 'param1': 200, 'param2': 100, 'minRadius': 0, 'maxRadius': 0 } # Default classifier settings
hough_params = default_hough_params.copy() # Dictionary to store HoughCircles parameters

class ProgressBarCallback(Callback):
    def __init__(self, progress_bar, progress_label, num_epochs):
        super().__init__()
        self.progress_bar = progress_bar
        self.progress_label = progress_label
        self.num_epochs = num_epochs

    def on_epoch_end(self, epoch, logs=None):
        progress = (epoch + 1) / self.num_epochs * 100
        self.progress_bar['value'] = progress
        self.progress_label.config(text=f'{progress:.2f}%')
        window.update_idletasks()  # Update the GUI

def train_model():
    global model, class_labels, stop_event, training_thread, training_cycles

    stop_event.clear()  # Clear the stop event before starting training

    if train_button['text'] == 'Stop Training':
        stop_event.set()  # Set the event to Stop Training
        train_button.config(text="Train Model")
        progress_bar['value'] = 0  # Reset progress bar
        progress_label.config(text='0%')  # Reset progress label
        test_mode_button.config(state=tk.NORMAL) # Enable the Test Mode button after training completes
        return

    data_dir = filedialog.askdirectory(title="Select Training Data Directory")
    if not data_dir:
        return

    train_dir_text.config(state=tk.NORMAL)
    train_dir_text.delete(1.0, tk.END)
    train_dir_text.insert(tk.END, data_dir)
    train_dir_text.config(state=tk.DISABLED)

    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
    class_labels = []
    total_image_counter = 0
    training_cycles = 25
    
    for subdir in sorted(os.listdir(data_dir)):
        subdir_path = os.path.join(data_dir, subdir)
        if os.path.isdir(subdir_path) and not subdir.startswith('.'):
            # Check if the subdirectory contains valid image files
            image_counter = sum(1 for file in os.listdir(subdir_path) if file.lower().endswith(valid_extensions))
            if image_counter > 0:
                total_image_counter += image_counter
                class_labels.append(subdir)
    
    if not class_labels:
        messagebox.showerror("Error", "No subdirectories containing valid image files found in the selected directory !")
        return

    num_classes = len(class_labels)

    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(*IMG_SIZE, 3))

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)

    for layer in base_model.layers:
        layer.trainable = False

    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

    # BEST WITH THESE VALUES
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2,
        rotation_range=90,
        width_shift_range=0.1,
        height_shift_range=0.1,
        #brightness_range=[0.9, 1.1],
        shear_range=0.1,
        zoom_range=0.2,
        channel_shift_range=0.1,
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode='nearest'
    )

    train_generator = train_datagen.flow_from_directory(
        data_dir, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='categorical', subset='training',
        classes=class_labels
    )
    validation_generator = train_datagen.flow_from_directory(
        data_dir, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='categorical', subset='validation',
        classes=class_labels
    )

    has_validation_data = validation_generator.samples > 0

    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    model_checkpoint = ModelCheckpoint('best_model.h5', save_best_only=True, monitor='val_loss')
    progress_callback = ProgressBarCallback(progress_bar, progress_label, num_epochs=training_cycles)

    train_button.config(text="Stop Training")
    test_mode_button.config(state=tk.DISABLED) # Disable the Test Mode button when training starts

    def run_training():
        global model, class_labels, stop_event

        progress_bar['value'] = 0
        progress_label.config(text='0%')
        callbacks = [progress_callback]

        if has_validation_data:
            callbacks.extend([early_stopping, model_checkpoint])

        # Initialize lists to accumulate history
        all_acc = []
        all_val_acc = []
        all_loss = []
        all_val_loss = []

        output_text.config(state=tk.NORMAL)  # Enable writing
        output_text.delete(1.0, tk.END)  # Clear previous output
        output_text.insert(tk.END, f"Input directory:\t{data_dir}\n")
        output_text.insert(tk.END, f"Total number of images:\t{total_image_counter}\n")
        output_text.insert(tk.END, f"Total number of classes:\t{num_classes}\n")
        output_text.insert(tk.END, f"Classes:\t{', '.join(class_labels)}\n\n")
        output_text.insert(tk.END, f"Starting {training_cycles} cycles of training, please wait...\n")
        output_text.config(state=tk.DISABLED)  # Make read-only again

        for epoch in range(training_cycles):
            if stop_event.is_set():
                stop_event.clear()  # Clear stop event
                progress_bar['value'] = 0  # Reset progress bar
                progress_label.config(text='0%')  # Reset progress label
                output_text.config(state=tk.NORMAL)  # Enable writing
                output_text.delete(1.0, tk.END)  # Clear previous output
                output_text.insert(tk.END, "Stopped training\n")
                output_text.config(state=tk.DISABLED)  # Make read-only again
                break  # Break out of loop if stopped

            history = model.fit(
                train_generator,
                epochs=1,  # Train for 1 epoch at a time
                validation_data=validation_generator if has_validation_data else None,
                callbacks=callbacks,
                verbose=1
            )

            # Accumulate history
            all_acc.extend(history.history['accuracy'])
            if 'val_accuracy' in history.history:
                all_val_acc.extend(history.history['val_accuracy'])
            
            all_loss.extend(history.history['loss'])
            if 'val_loss' in history.history:
                all_val_loss.extend(history.history['val_loss'])

            progress = (epoch + 1) * 100 / training_cycles  # Calculate progress in percentage
            progress_bar['value'] = progress
            progress_label.config(text=f'{progress:.2f}%')

            output_text.config(state=tk.NORMAL)
            output_text.insert(tk.END, f"Training cycle completed {epoch + 1}/{training_cycles}\n")
            output_text.see(tk.END)
            output_text.config(state=tk.DISABLED)
            window.update_idletasks()  # Update the GUI

            # Check if the training reached 100% progress
            if progress_bar['value'] == 100:
                # Prompt user to select location to save the trained model
                model_save_path = filedialog.asksaveasfilename(
                    defaultextension=".h5", filetypes=[("H5 files", "*.h5")], title="Save Trained Model"
                )
                if model_save_path:
                    model.save(model_save_path)
                    with h5py.File(model_save_path, 'a') as f:
                        f.attrs['class_labels'] = class_labels
                
                progress_bar.stop()
                train_button.config(text="Train Model")
                accumulated_history = {
                    'accuracy': all_acc,
                    'val_accuracy': all_val_acc,
                    'loss': all_loss,
                    'val_loss': all_val_loss
                }
                window.after(0, lambda: plot_training_history(accumulated_history))
                
                test_mode_button.config(state=tk.NORMAL) # Enable the Test Mode button after training completes
                output_text.config(state=tk.NORMAL)
                output_text.insert(tk.END, "Training completed\n")
                output_text.see(tk.END)
                output_text.config(state=tk.DISABLED)
                window.update_idletasks()  # Update the GUI

    training_thread = Thread(target=run_training)
    training_thread.start()

def plot_training_history(history):
    epochs = range(1, len(history['accuracy']) + 1)

    fig, ax1 = plt.subplots(figsize=(6, 3))

    # Plotting training accuracy
    ax1.set_xlabel('Training cycle', fontsize=10)
    ax1.set_ylabel('Metrics', fontsize=10)
    if history['accuracy']:
        ax1.plot(epochs, history['accuracy'], 'b', label='Training Accuracy')
    if 'val_accuracy' in history and history['val_accuracy']:
        ax1.plot(epochs, history['val_accuracy'], 'b--', label='Validation Accuracy')

    # Plotting training loss
    if history['loss']:
        ax1.plot(epochs, history['loss'], 'r', label='Training Loss')
    if 'val_loss' in history and history['val_loss']:
        ax1.plot(epochs, history['val_loss'], 'r--', label='Validation Loss')

    # Title, axes and legend tweaks
    ax1.set_title('Training and Validation Metrics', fontsize=12)
    ax1.tick_params(axis='both', labelsize=8)

    ax1.xaxis.set_major_locator(plt.MultipleLocator(1))  # Adding major ticks with step 1.0
    ax1.xaxis.set_major_formatter(ticker.FormatStrFormatter('%d')) # Adding numeric labels to major ticks

    ax1.yaxis.set_major_locator(plt.MultipleLocator(0.1))     # Adding major ticks to y-axis with step 0.1
    ax1.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))    # Adding numeric labels to major ticks

    fig.tight_layout()
    fig.legend(loc='center right', bbox_to_anchor=(1, 0.5), bbox_transform=ax1.transAxes)

    # Clear previous plot if exists
    for widget in plot_frame.winfo_children():
        widget.destroy()

    canvas = FigureCanvasTkAgg(fig, master=plot_frame)
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
    toolbar = NavigationToolbar2Tk(canvas, plot_frame)
    toolbar.update()

    toolbar.canvas.get_default_filetype = lambda: 'pdf'    
    canvas.draw()


def load_model_for_testing():
    global model, class_labels, model_load_path

    # Prompt user to select the trained model file
    model_load_path = filedialog.askopenfilename(filetypes=[("H5 files", "*.h5")], title="Select Trained Model File")
    if not model_load_path:
        return

    load_dir_text.config(state=tk.NORMAL)
    load_dir_text.delete(1.0, tk.END)
    load_dir_text.insert(tk.END, model_load_path)
    load_dir_text.config(state=tk.DISABLED)

    model = load_model(model_load_path)

    # Load class labels from model file
    try:
        with h5py.File(model_load_path, 'r') as f:
            class_labels = list(f.attrs['class_labels'])
    except KeyError:
        print("Attribute 'class_labels' not found in the HDF5 file.")
        # Handle the case where the attribute is not found
        class_labels = []


def classifier():
    global processed_images, testing_thread

    stop_event.clear()  # Clear the stop event before starting classification

    if classify_image_button['text'] == 'Stop Testing':
        stop_event.set()  # Set the event to Stop Classification
        classify_image_button.config(text="Classify Image(s)")
        train_mode_button.config(state=tk.NORMAL)  # Enable the Train Mode button when testing stops
        return

    # if model text field empty i.e. model not loaded
    if not load_dir_text.get("1.0", "end-1c"):
        messagebox.showerror("Error", "No model file loaded !")
        return

    file_path = filedialog.askopenfilenames(filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff")], title="Classify Image(s) for Classification")
    if not file_path:
        return
        
    classify_image_button.config(text="Stop Testing")
    train_mode_button.config(state=tk.DISABLED)  # Disable the Train Mode button when testing
    
    # Clear previous output and display classification result
    output_text.config(state=tk.NORMAL)  # Enable writing
    output_text.delete(1.0, tk.END)
    output_message = f"Loaded model file:\n{model_load_path}\n\n"
    output_message += f"Classifier settings:\ndp={hough_params['dp']}, mindist={hough_params['minDist']}, param1={hough_params['param1']}, param2={hough_params['param2']}, minRadius={hough_params['minRadius']}, maxRadius={hough_params['maxRadius']}\n\n"
    output_text.insert(tk.END, output_message)
    output_text.config(state=tk.DISABLED)  # Make read-only again

    def run_classification():
        global stop_event

        predictions = []
        processed_images.clear()  # Clear any previous processed images

        for path in file_path:
            if stop_event.is_set():
                stop_event.clear()  # Clear stop event & break loop
                processed_images.clear()  # Clear any previous processed images
                output_text.config(state=tk.NORMAL)
                output_text.delete(1.0, tk.END)  # Clear previous output
                output_text.insert(tk.END, "Stopped image classification\n")
                output_text.config(state=tk.DISABLED)
                classify_image_button.config(text="Classify Image(s)")
                train_mode_button.config(state=tk.NORMAL)  # Enable the Train Mode button when testing stops
                save_report_button.config(state=tk.DISABLED)
                save_image_button.config(state=tk.DISABLED)
                break
            else :
                classify_dir_text.config(state=tk.NORMAL)
                classify_dir_text.delete(1.0, tk.END)
                classify_dir_text.insert(tk.END, path)
                classify_dir_text.config(state=tk.DISABLED)

                output_text.config(state=tk.NORMAL)  # Enable writing
                output_message = f"Input image:\n{path}\n\n"
                output_text.insert(tk.END, output_message)

                img = cv2.imread(path)
                resize_scale = 0.1
                image_resized = cv2.resize(img, (0, 0), fx=resize_scale, fy=resize_scale, interpolation=cv2.INTER_AREA)

                ### RGB
                # Apply Gaussian blur
                blurred = cv2.GaussianBlur(image_resized, (3, 3), 1.5)

                # Process each color channel separately
                channels = cv2.split(blurred)
                gradients = []
                for channel in channels:
                    grad_x = cv2.Sobel(channel, cv2.CV_64F, 1, 0, ksize=3)  # Adjust kernel size if needed
                    grad_y = cv2.Sobel(channel, cv2.CV_64F, 0, 1, ksize=3)
                    magnitude = cv2.magnitude(grad_x, grad_y)
                    gradients.append(magnitude)

                # Combine the gradients from all channels
                combined_magnitude = np.sqrt(np.sum([grad ** 2 for grad in gradients], axis=0))
                normalized_magnitude = cv2.normalize(combined_magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                ###

                # Perform Circular Hough Transform / Tweak parameters for better circle detection
                circles = cv2.HoughCircles(
                    normalized_magnitude,
                    cv2.HOUGH_GRADIENT,
                    dp=hough_params['dp'],
                    minDist=hough_params['minDist'],
                    param1=hough_params['param1'],
                    param2=hough_params['param2'],
                    minRadius=hough_params['minRadius'],
                    maxRadius=hough_params['maxRadius']
                )

                if circles is not None:
                    circles = np.uint16(np.around(circles))
                    for idx, circle in enumerate(circles[0, :], start=1):
                        # Adjust the circle parameters back to the original image scale
                        center = (int(circle[0] / resize_scale), int(circle[1] / resize_scale))
                        radius = int(circle[2] / resize_scale)

                        # Crop the region inside the circle
                        x, y, r = center[0], center[1], radius
                        cropped_img = img[max(y-r, 0):min(y+r, img.shape[0]), max(x-r, 0):min(x+r, img.shape[1])]
                        cropped_img_rgb = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB)
                        cropped_img_pil = Image.fromarray(cropped_img_rgb)

                        img_array = img_to_array(cropped_img_pil.resize(IMG_SIZE)) / 255.0
                        img_array = np.expand_dims(img_array, axis=0)

                        # Prediction using the custom fungi model
                        predictions = model.predict(img_array)

                        # Get the indices that would sort the probabilities in descending order
                        sorted_indices = np.argsort(predictions[0])[::-1]

                        # Output all possibilities of species predictions in descending percentage order
                        prediction_output = ""
                        for index in sorted_indices:
                            species = class_labels[index]
                            probability = predictions[0][index] * 100
                            prediction_output += f"{species}\t{probability:.2f}%\n"

                        # Clear previous output and display classification result
                        output_text.config(state=tk.NORMAL)  # Enable writing
                        output_message = f"Detected Circle C{idx}\nPredicted Species:\n{prediction_output}\n\n"
                        output_text.insert(tk.END, output_message)

                        # Draw the outer circle
                        cv2.circle(img, center, radius, (0, 255, 0), 20)
                        # Draw the center of the circle
                        cv2.circle(img, center, 2, (0, 0, 255), 30)
                        # Add a label
                        label = f'C{idx}'
                        cv2.putText(img, label, (center[0] + radius + 10, center[1]), cv2.FONT_HERSHEY_DUPLEX, 5, (255, 0, 0), 10)

                        edited_img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        edited_img_pil = Image.fromarray(edited_img_rgb)

                        processed_images.append((edited_img_pil, path))  # Store processed image and file path
                else:
                    classify_image_button.config(text="Classify Image(s)")
                    train_mode_button.config(state=tk.NORMAL)  # Enable the Train Mode button when testing stops
                    output_text.config(state=tk.NORMAL)  # Enable writing
                    output_message = f"No circles detected in the image !\n\n"
                    output_text.insert(tk.END, output_message)
                    output_text.config(state=tk.DISABLED)  # Make read-only again
                    return

                output_text.config(state=tk.NORMAL)  # Enable writing
                output_message = f"Circle detection completed successfully !\n\n"
                output_text.insert(tk.END, output_message)
                output_text.config(state=tk.DISABLED)  # Make read-only again

        classify_image_button.config(text="Classify Image(s)")
        train_mode_button.config(state=tk.NORMAL)  # Enable the Train Mode button when testing completes

        output_text.config(state=tk.DISABLED)  # Make read-only again
        save_report_button.config(state=tk.NORMAL)  # Enable save report button
        save_image_button.config(state=tk.NORMAL)  # Enable save report button

    classification_thread = Thread(target=run_classification)
    classification_thread.start()

# Function to open the parameter settings menu
def settings_menu():
    param_window = tk.Toplevel(window)
    param_window.title("Set HoughCircles Parameters")

    # Create and place labels and entry widgets for each parameter
    params = [
        ('dp', 'dp (Inverse ratio of the accumulator resolution):', 1.0),
        ('minDist', 'minDist (Minimum distance between circle centers):', 100),
        ('param1', 'param1 (Higher threshold for the Canny edge detector):', 100),
        ('param2', 'param2 (Accumulator threshold for the circle centers):', 100),
        ('minRadius', 'minRadius (Minimum circle radius):', 0),
        ('maxRadius', 'maxRadius (Maximum circle radius):', 0)
    ]
    
    entries = {}
    
    for i, (param, text, default) in enumerate(params):
        label = tk.Label(param_window, text=text)
        label.grid(row=i, column=0, padx=10, pady=5, sticky=tk.W)
        
        entry = tk.Entry(param_window)
        entry.grid(row=i, column=1, padx=10, pady=5)
        entry.insert(0, str(hough_params[param]))
        entries[param] = entry

    def set_parameters():
        try:
            hough_params['dp'] = float(entries['dp'].get())
            hough_params['minDist'] = int(entries['minDist'].get())
            hough_params['param1'] = int(entries['param1'].get())
            hough_params['param2'] = int(entries['param2'].get())
            hough_params['minRadius'] = int(entries['minRadius'].get())
            hough_params['maxRadius'] = int(entries['maxRadius'].get())
            param_window.destroy()
            classify_setting_text.config(state=tk.NORMAL)
            classify_setting_text.delete(1.0, tk.END)
            classify_setting_text.insert(tk.END, f"dp={hough_params['dp']}, mindist={hough_params['minDist']}, param1={hough_params['param1']}, param2={hough_params['param2']}, minRadius={hough_params['minRadius']}, maxRadius={hough_params['maxRadius']}")
            classify_setting_text.config(state=tk.DISABLED)
        except ValueError:
            messagebox.showerror("Invalid input", "Please enter valid values for all parameters.")

    def reset_params():
        for param, default in default_hough_params.items():
            entries[param].delete(0, tk.END)
            entries[param].insert(0, str(default))

    # Button to set the parameters
    set_button = tk.Button(param_window, text="Set Parameters", command=set_parameters)
    set_button.grid(row=len(params), column=0, columnspan=2, pady=10)

    # Button to reset the parameters
    reset_button = tk.Button(param_window, text="Reset Parameters", command=reset_params)
    reset_button.grid(row=len(params) + 1, column=0, columnspan=2, pady=10)


# Function to save the output report
def save_report():
    report_file_path = filedialog.asksaveasfilename(defaultextension=".txt", filetypes=[("Text files", "*.txt")], title="Save Output Report")
    if report_file_path:
        with open(report_file_path, 'w') as report_file:
            report_file.write(output_text.get(1.0, tk.END))

# Function to save the processed image(s)
def save_processed_images():
    if not processed_images:
        messagebox.showerror("Error", "No processed images to save !")
        return

    save_dir = filedialog.askdirectory(title="Select Directory to Save Images")
    if not save_dir:
        return

    for img, original_path in processed_images:
        new_filename = "NEW-" + os.path.basename(original_path)
        save_path = os.path.join(save_dir, new_filename)
        img.save(save_path)

def switch_to_train_mode():
    global stop_event, training_thread

    if train_button['text'] == 'Stop Training':
        stop_event.set()  # Set the event to Stop Training
        train_button.config(text="Train Model")
        return

    if training_thread and training_thread.is_alive():
        training_thread.join()  # Wait for the thread to finish

    train_button.config(text="Train Model")
    train_button.config(state=tk.NORMAL)
    train_dir_text.config(state=tk.NORMAL)
    load_model_button.config(state=tk.DISABLED)
    load_dir_text.config(state=tk.DISABLED)
    classify_setting_text.config(state=tk.NORMAL)
    classify_setting_text.delete(1.0, tk.END)
    classify_setting_text.config(state=tk.DISABLED)
    classify_setting_button.config(state=tk.DISABLED)
    classify_image_button.config(state=tk.DISABLED)
    classify_dir_text.config(state=tk.DISABLED)
    output_text.config(state=tk.NORMAL)
    output_text.delete(1.0, tk.END)
    output_text.config(state=tk.DISABLED)
    save_report_button.config(state=tk.DISABLED)
    save_image_button.config(state=tk.DISABLED)

def switch_to_test_mode():
    train_button.config(state=tk.DISABLED)
    train_dir_text.config(state=tk.DISABLED)
    load_model_button.config(state=tk.NORMAL)
    load_dir_text.config(state=tk.NORMAL)
    classify_setting_text.config(state=tk.NORMAL)
    classify_setting_text.delete(1.0, tk.END)
    classify_setting_text.insert(tk.END, f"dp={hough_params['dp']}, mindist={hough_params['minDist']}, param1={hough_params['param1']}, param2={hough_params['param2']}, minRadius={hough_params['minRadius']}, maxRadius={hough_params['maxRadius']}")
    classify_setting_text.config(state=tk.DISABLED)
    classify_setting_button.config(state=tk.NORMAL)
    classify_image_button.config(state=tk.NORMAL)
    classify_dir_text.config(state=tk.NORMAL)
    output_text.config(state=tk.NORMAL)
    output_text.delete(1.0, tk.END)
    output_text.config(state=tk.DISABLED)
    save_report_button.config(state=tk.DISABLED)
    save_image_button.config(state=tk.DISABLED)

def center_window(window, width, height):
    screen_width = window.winfo_screenwidth()
    screen_height = window.winfo_screenheight()
    x = (screen_width // 2) - (width // 2)
    y = (screen_height // 2) - (height // 2)
    window.geometry(f"{width}x{height}+{x}+{y}")

# GUI
window = tk.Tk()
window.title("Fungi Species Classifier")
window_width = 800
window_height = 800
center_window(window, window_width, window_height)

# Mode selection buttons with larger font
train_mode_button = Button(window, text="TRAIN MODE", font=("Helvetica", 16), anchor='w', command=switch_to_train_mode)
train_mode_button.pack(fill='x', pady=10, padx=10)

train_frame = tk.Frame(window)
train_frame.pack(fill='x', pady=5, padx=10)
train_dir_text = Text(train_frame, height=1, state=tk.DISABLED)
train_dir_text.pack(side=tk.LEFT, fill='x', expand=True)
train_button = Button(train_frame, text="Train Model", command=train_model, state=tk.DISABLED)
train_button.pack(side=tk.LEFT, padx=5)

# Progress bar and label
progress_frame = tk.Frame(window)
progress_frame.pack(fill='x', pady=10, padx=10)
progress_bar = ttk.Progressbar(progress_frame, mode='determinate', maximum=100)
progress_bar.pack(side=tk.LEFT, fill='x', expand=True)
progress_label = Label(progress_frame, text='0%')
progress_label.pack(side=tk.LEFT, padx=10)

# Frame for plotting
plot_title = tk.Label(window, text="MODEL PERFORMANCE", font=('Helvetica', 12, 'bold'))
plot_title.pack(side=tk.TOP, anchor='w', pady=(5, 0))

plot_frame = tk.Frame(window)
plot_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

test_mode_button = Button(window, text="TEST MODE", font=("Helvetica", 16), anchor='w', command=switch_to_test_mode)
test_mode_button.pack(fill='x', pady=10, padx=10)

test_frame = tk.Frame(window)
test_frame.pack(fill='x', pady=5, padx=10)

# Create a frame for the setting buttons
load_dir_text = Text(test_frame, height=1, state=tk.DISABLED)
load_dir_text.pack(side=tk.LEFT, fill='x', expand=True)
load_model_button = Button(test_frame, text="Load Model", command=load_model_for_testing, state=tk.DISABLED)
load_model_button.pack(side=tk.LEFT, padx=5)

classify_setting_frame = tk.Frame(window)
classify_setting_frame.pack(fill='x', pady=5, padx=10)
classify_setting_text = Text(classify_setting_frame, height=1, state=tk.DISABLED)
classify_setting_text.pack(side=tk.LEFT, fill='x', expand=True)
classify_setting_button = Button(classify_setting_frame, text="Adjust classifier", command=settings_menu, state=tk.DISABLED)
classify_setting_button.pack(side=tk.LEFT, padx=5)

classify_frame = tk.Frame(window)
classify_frame.pack(fill='x', pady=5, padx=10)
classify_dir_text = Text(classify_frame, height=1, state=tk.DISABLED)
classify_dir_text.pack(side=tk.LEFT, fill='x', expand=True)
classify_image_button = Button(classify_frame, text="Classify Image(s)", command=classifier, state=tk.DISABLED)
classify_image_button.pack(side=tk.LEFT)

# Create a frame for the buttons
button_frame = tk.Frame(window)
button_frame.pack(fill='x', pady=5, padx=10)

# Add a button to save the output report of the image classification
save_report_button = Button(button_frame, text="Save Report", command=save_report, state=tk.DISABLED)
save_report_button.grid(row=0, column=0, padx=5, pady=5, sticky='ew')

# Add a button to save the processed images
save_image_button = Button(button_frame, text="Save Image(s)", command=save_processed_images, state=tk.DISABLED)
save_image_button.grid(row=0, column=1, padx=5, pady=5, sticky='ew')

# Configure the grid to make buttons evenly spaced
button_frame.grid_columnconfigure(0, weight=1)
button_frame.grid_columnconfigure(1, weight=1)

output_label = tk.Label(window, text="OUTPUT CONSOLE", font=('Helvetica', 12, 'bold'))
output_label.pack(side=tk.TOP, anchor='w', pady=(5, 0))
output_text = tk.Text(window, height=5, width=80)
output_text.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
output_text.config(state=tk.DISABLED)  # Make the output console read-only

window.mainloop()
