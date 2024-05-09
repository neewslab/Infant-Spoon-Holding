# Infant Spoon Holding


Description: In this project, the aim is to study and analyze how infants hold objects such as spoon, pencil etc. We are interested in the following classification problem:

Category 1 (1-5 years of age) (3 class problem):
(a) Power grip (Does not require the details on fingers involved for the grip)
-- Overhand radial
-- Overhand ulnar
(b) Precision grip (Requires the details on fingers involved for the grip) :
-- Fingertip

Category 2 (School age kids: 5th grade and above) (4 class problem)
Precision grip classes (Based on number of fingers involved):
(a) 4 fingers
-- Extension class
-- Mid range class
(b) 3 fingers
-- Extension class
-- Mid range class


The file power_precision.py has the script for the first classification problem (category 1). 
To run the script:

1. Import all the required machine learning and input-output libraries
2. Define how many times the data needs to be shuffled for k-fold validation in parameter K
3. Load the dataset from the local directory as a csv file and store in variable dataset_raw
    -> The file fingertip_vs_overhand_ulnar_labelled.csv is the labelled dataset (3 subjects) for classes Overhand ulnar and Fingertip
    -> The file fingertip_vs_overhand_radial_labelled.csv is the labelled dataset (3 subjects) for classes Overhand radial and Fingertip
    -> The files have processed data with the following features (each colum represents a feature):
       ---- Mean Acceleration (3-axes)
       ---- Mean rotation of axes (3-axes)
       ---- SD of Acceleration (3-axes)
       ---- SD of rotation of axes (3-axes)

5. Shuffle the dataset for making it suitable for k-fold validation
6. Apply min-max normalization
7. Define the NN model with all its parameters, including loss function, optimizer, evaluation metric
8. Compile and run the model
9. Find the confusion matrix and accuracy using variables true_p and false_p


The file midrange_extension.py has the script for the second classification problem (category 2). 
To run the script:

1. Import all the required machine learning and input-output libraries
2. Define how many times the data needs to be shuffled for k-fold validation in parameter K
3. Load the dataset from the local directory as a csv file and store in variable dataset
5. Shuffle the dataset for making it suitable for k-fold validation
6. Apply min-max normalization
7. Define the NN model with all its parameters, including loss function, optimizer, evaluation metric
8. Compile and run the model
9. Find the confusion matrix and accuracy using variables true_p and false_p
