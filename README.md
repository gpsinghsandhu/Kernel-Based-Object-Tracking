Kernel-Based-Object-Tracking
============================
REQUIREMENTS:
  + OpenCV 2.2 or above (originally developed on OpenCV 2.3.1)
  + This a Visual Studio 2010 project with OpenCV 2.3.1 linked. To use on other platforms use only main.cpp with "conio" and "getch" removed.


This project is the C++ implementation of kernel based object tracking as discussed by Visvanathan Ramesh, Dorin Comaniciu &amp; Peter Meer in their paper "Kernel-Based Object Tracking".

In this project the objects are represented by their color histograms weighted by isotropic kernel. Targets (or objects) are compared in subsequent frames to calculate Bhattacharya distance which is then used to move the tracked using mean shift.

Ceck out the result video - http://youtu.be/Ng8H-mjs62Y

