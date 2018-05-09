"# Heart" 
"# HeartDiseaseDetection" 

Dependencies: Python 3.6.5, Tensorflow 1.8.0rc0, pydicom 1.0.2, scipy 1.0.1, opencv-python 3.4.0.12, scikit-learn 0.19.1, numpy 1.4.2, scipy 1.0.1

Import the project directory into your IDE. Install all the dependencies. Make sure you import the whole project and not just the 
directory with this ReadMe.md. The models in the outer direcotry will be required to run the code. 

Check the results file: volume_diastole.csv
Check the results file: volume_systole.csv

Move both files to the directory with all the models.

Next:
Run predict_systole.py
Run predict_diastole.py

The entire dataset is not included with this directory. If you need the whole dataset look for the link to the whole dataset in the report. The results mentioned below won't match unless the whole dataset it used. Download the whole dataset from the link mentioned and replace the testing_data folder with the downloaded testing_data folder. 

Make sure the results volume_diastole.csv, and volume_systole.csv match with the files moved to the outer directory.

If the two match run the r2_score.py

The output should be:
0.03
-6.0

The first number is the diastole model r2_score
The second number is the systole model r2_score



