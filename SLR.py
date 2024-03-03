# Import kivy dependencies
import kivy
kivy.require('2.0.0')
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout

# Import kivy UX components
from kivy.uix.image import Image
from kivy.uix.button import Button
from kivy.uix.label import Label

# Import other kivy stuff
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.logger import Logger
from kivy.lang import Builder

from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam

# Import other dependencies
import cv2
# from layers import L1Dist
import mediapipe as mp
import os
import numpy as np
import pyttsx3



mp_holistic = mp.solutions.holistic  # Holistic model
mp_drawing = mp.solutions.drawing_utils  # Drawing utilities


def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # COLOR CONVERSION BGR 2 RGB
    image.flags.writeable = False  # Image is no longer writeable
    results = model.process(image)  # Make prediction
    image.flags.writeable = True  # Image is now writeable
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # COLOR COVERSION RGB 2 BGR
    return image, results


def draw_landmarks(image, results):
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS,
                              mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1),
                              mp_drawing.DrawingSpec(color=(80, 256, 121), thickness=1, circle_radius=1)
                              )  # Draw face connections
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)  # Draw pose connections
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks,
                              mp_holistic.HAND_CONNECTIONS)  # Draw left hand connections
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks,
                              mp_holistic.HAND_CONNECTIONS)  # Draw right hand connections


def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in
                     results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33 * 4)
    face = np.array([[res.x, res.y, res.z] for res in
                     results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468 * 3)
    lh = np.array([[res.x, res.y, res.z] for res in
                   results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21 * 3)
    rh = np.array([[res.x, res.y, res.z] for res in
                   results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(
        21 * 3)
    return np.concatenate([pose, face, lh, rh])


# DATA_PATH = os.path.join('SignAction_Data')

# Actions that we try to detect
# array=['Beautiful', 'Brinjal', 'Brush', 'Building', 'Cake', 'Car', 'Cauliflower', 'Chilli', 'Clapping', 'Climbing', 'Coconut', 'Combing', 'Counting', 'Cucumber', 'Dizziness', 'Dressing', 'Drinking', 'Driving', 'Fever', 'Good Afternoon!', 'Good Morning!', 'Good Night!', 'Green Peas', 'Hanging', 'Happy', 'I', 'Ironing', 'Kissing', 'Knocking', 'Lady Finger', 'Licking', 'Like', 'Marching', 'Milk', 'My Family', 'My Name is', 'No', 'Onion', 'Peeling', 'Potato', 'Raddish', 'Sorry', 'Thank You!', 'Time', 'Tomato', 'Water', 'Welcome!', 'Yes', 'You']
# array = os.listdir(DATA_PATH)
array=['Brush', 'Combing',  'Dressing',  'I', 'Water']
actions = np.array(array)
label_map = {label: num for num, label in enumerate(actions)}


# Build app and layout


class CamApp(App):

    def build(self):
        # Main layout components
        self.web_cam = Image(size_hint=(1, .8))
        self.button = Button(text="Predict", on_press=self.verify ,size_hint=(0.1,0.1),pos_hint={'x': 0.45, 'y':0.5})

        self.prediction_label = Label(text="Not Predicted Yet", size_hint=(1, .1))

        # Add items to layout
        layout = BoxLayout(orientation='vertical')
        layout.add_widget(self.web_cam)
        layout.add_widget(self.button)
        layout.add_widget(self.prediction_label)

        # Load tensorflow/keras model
        self.model =model= tf.keras.models.load_model('action.h5')

        # Setup video capture device
        cap = self.capture = cv2.VideoCapture(0)
        Clock.schedule_interval(self.update, 1.0 / 33.0)

        return layout

    # Run continuously to get webcam feed
    def update(self, *args):



        # global frame
        ret, frame = self.capture.read()
        # cv2.imwrite(SAVE_PATH, frame)



        buf = cv2.flip(frame, 0).tostring()
        img_texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
        img_texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
        self.web_cam.texture = img_texture



    # Verification function to verify person
    def verify(self, *args):
        # Specify thresholds
        sequence = []
        sentence = []
        predictions = []
        threshold = 0.6
        self.model =model= tf.keras.models.load_model('action.h5')





        cap=0

        with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
            while (cap<30):
                # SAVE_PATH = os.path.join('SLR_Android', 'cap.jpg')


                # Read feed
                ret, frame = self.capture.read()
                # Clock.schedule_interval(self.update, 1.0 / 33.0)

                cv2.imwrite("D:/BE_Android_trail/Images1/{}.jpg".format(cap), frame)



                # Make detections
                image, results = mediapipe_detection(frame, holistic)
                print(results)

                # Draw landmarks
                draw_landmarks(image, results)

                # 2. Prediction logic
                keypoints = extract_keypoints(results)
                sequence.append(keypoints)
                print(len(sequence))

                cap=cap+1

                if len(sequence) == 30:
                    res = model.predict(np.expand_dims(sequence, axis=0))[0]

                    print(actions[np.argmax(res)])
                    sentence.append(actions[np.argmax(res)])
                    print('action is ', sentence)
                    predictions.append(np.argmax(res))
                self.prediction_label.text = str(sentence)
                engine=pyttsx3.init()
                engine.say(str(sentence))
                engine.runAndWait()




        return sentence


if __name__ == '__main__':
    CamApp().run()


    # pyinstaller --name YourAppName main.py
    '''This code appears to be a Kivy application that uses the Mediapipe library and a pre-trained TensorFlow/Keras model to perform real-time action recognition using a webcam feed. Let's break down the code:

Importing Dependencies: The code begins by importing the required dependencies, including Kivy, OpenCV (cv2), TensorFlow, Mediapipe, and other necessary components.

Function Definitions: Two functions are defined in the code:

mediapipe_detection(image, model): This function takes an image and a Mediapipe model as input and performs holistic detection using the model. It returns the processed image and the detection results.

draw_landmarks(image, results): This function takes an image and the detection results from Mediapipe and draws the detected landmarks (face, body, hands) on the image using the mp_drawing utility.

App Class Definition: The CamApp class is defined, which extends the App class from Kivy. This class represents the main application and contains the necessary methods and properties.

build(): This method is called when the application is launched. It sets up the layout of the app, including a webcam image display, a button for prediction, and a label for showing the predicted action.

update(): This method is called repeatedly to update the webcam feed. It captures frames from the webcam, converts them to the appropriate format, and updates the displayed image.

verify(): This method is called when the prediction button is pressed. It performs the action recognition process using the Mediapipe model and the loaded TensorFlow/Keras model. It captures frames from the webcam, performs Mediapipe detection, extracts keypoints, feeds the keypoints to the loaded model, makes predictions, and updates the prediction label accordingly.

Main Execution: The code includes an if statement to check if it is being executed as the main program. If so, it creates an instance of the CamApp class and runs the Kivy application.

The purpose of this code is to create a Kivy application that allows real-time action recognition using a webcam. It uses the Mediapipe library for holistic detection and the TensorFlow/Keras model for making predictions.'''