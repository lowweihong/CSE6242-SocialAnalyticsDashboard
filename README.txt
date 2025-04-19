
# Social Analytics Dashboard For Bitcoin Related Discussion

---------------------
CONTENTS OF THIS FILE
---------------------

* DESCRIPTION
* INSTALLATION
* EXECUTION

-----------
DESCRIPTION
-----------
This package is the project by team 37 from CSE 6242 @ Georgia Tech Spring 2025. 
The project is about "Social Analytics Dashboard For Bitcoin Related Discussion". 
The package contains 1 python file `topic.py`, 1 `Final.html` file, and 3 csv files (combined_df_new.csv, out_40k.csv, topic_detail_new.csv)
The 3 csv files can be downloaded here (access required with Georgia Tech email): https://gtvault-my.sharepoint.com/:f:/g/personal/wlow7_gatech_edu/EmGIhEWNIW5JutheZH5su48BHsg_ckNkBhO3Bly8rJZtNg?e=8CA1ks

------------
INSTALLATION
------------
There is no specific installation to run these files with the generated csv file. 
To run the dashboard, 
1. please make sure you have the 3 csv files downloaded and place at the same directory in directory CODE.
2. navigate to `CODE/`
3. run `python -m http.server`
4. Use your browser to browse to `http://localhost:8000/Final.html`

However, to generate CSV file on a new dataset, run `python topic.py` (note: need to be ran in a device with GPU). 
Make sure you have installed all the requirements in `requirements.txt` with `pip install -r requirements.txt`.

---------
EXECUTION
---------
The dashboard will be displayed on the first page. Once all the 4 graphs loaded,
you may use the filter function, and hover to each of the datapoints to have a closer look.
