# Drowsiness Detection System
This project provides a real-time solution for identifying signs of drowsiness in drivers using non-invasive methods. By calculating the Eye Aspect Ratio (EAR) from facial landmarks, the system can assess a person's condition based on the closing of their eyes—requiring only a camera facing the person, with no need for specialized sensors or intrusive devices.

The system uses computer vision techniques to monitor eye activity and detect drowsiness. It can be implemented in real-time, such as in vehicles, to alert drivers before drowsiness leads to accidents.


### Face Landmarks

![image](https://github.com/user-attachments/assets/f00ebb15-c8dd-4916-ad59-a1c3988fa97f)


Facial landmarks are specific points (e.g., eyes, nose, mouth, chin) detected on a face in images or videos. The goal is to locate these landmarks in real-time and use them for various applications, such as facial expression analysis, face recognition, and drowsiness detection.

In this project, I use dlib's facial landmark detector to detect 68 facial landmarks, which allows us to calculate the Eye Aspect Ratio (EAR) and assess the drowsiness level of the person.

### EAR Formula

![image](https://github.com/user-attachments/assets/b4ab0de0-d2a1-440c-86d2-830be47e4380)


The Eye Aspect Ratio (EAR) formula calculates whether the eyes are open or closed. If the eyes remain closed for a prolonged period, the system detects drowsiness.

### Results

The system successfully detects when a person's eyes are closed under proper lighting conditions.

![Result 1](https://raw.githubusercontent.com/user/repository/main/assets/result1.png)
![Result 2](https://raw.githubusercontent.com/user/repository/main/assets/result2.png)

