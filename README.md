# Object Dimensioning Using Two Cameras

The objective of this work is to develop a script to dimension the diameter and height of the object in millimeters using  :

- Images captured by two cameras.
- Intrinsic parameters of the two cameras.
- Parameters of the two Aruco markers.

## Prerequisites

- **Python:** The script has been developed and tested with Python 3.8.2.

- **Libraries:** The following libraries are required:
  - numpy 1.24.4
  - opencv 4.9.0
  - sys
  - json

## Usage

The script is ready to be executed from the command line. You can pass the paths to your folders containing the files as arguments when running the script.

Example: `python Main.py /path/to/folder`

## Expected Output Data

The script produces the following outputs:

Image:

[image1]: assets/1.png
[image2]: assets/2.png
[image3]: assets/3.png
[image4]: assets/4.png
[image5]: assets/5.png




## Detection of Aruco marker with ID 5 in the image captured by camera 1. 

![alt text][image1]

### Detection of circle described by the object in the image captured by camera 1 

![alt text][image2]

### Detection of Aruco marker with ID 0 in the image captured by camera 1 

![alt text][image3]


### Detection of Aruco marker with ID 0 in the image captured by camera 2 

![alt text][image4]

### Detection of the highest point in the object to be dimensioned in the image captured by camera 2 

![alt text][image5]



Console Output:

- Translation of the table in the coordinate system of camera 1.
- Rotation of the table in the coordinate system of camera 1.
- Diameter of the object in mm.
- Translation of Aruco 0 in the coordinate system of camera 1.
- Rotation of Aruco 0 in the coordinate system of camera 1.
- Translation of Aruco 0 in the coordinate system of camera 2.
- Rotation of Aruco 0 in the coordinate system of camera 2.
- Transformation matrix of camera 2 in the coordinate system of camera 1.
- Transformation matrix of camera 1 in the coordinate system of camera 2.
- Coordinates of the highest point in the coordinate system of camera 2.
- Coordinates of the highest point in the coordinate system of camera 1.
- Height of the object in mm.

- A JSON file containing all the results,
  To access the elements of this file, you need to use the `json.load()` function to convert the content of the JSON file into a Python dictionary. 
