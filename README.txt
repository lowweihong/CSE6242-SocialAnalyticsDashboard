
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
This package is the project submission by Team 37 for CSE 6242 at Georgia Tech, Spring 2025. The project, titled "Social Analytics Dashboard for Bitcoin-Related Discussion", focuses on analyzing and visualizing public discourse surrounding Bitcoin on X platforms. By leveraging topic modeling and interactive visualization techniques, this dashboard aims to provide insights into trends, sentiment, and key discussion themes over time.

The package includes the following files:
1. topic.py: A Python script for topic modeling and CSV generation.
2. Final.html: A pre-built interactive dashboard for exploring the processed data.
4. Three CSV files containing the dataset used by the dashboard:
    - combined_df_new.csv
    - out_40k.csv
    - topic_detail_new.csv
Due to data access policies, the CSV files are hosted on a Georgia Tech-accessible link:https://gtvault-my.sharepoint.com/:f:/g/personal/wlow7_gatech_edu/EmGIhEWNIW5JutheZH5su48BHsg_ckNkBhO3Bly8rJZtNg?e=8CA1ks

------------
INSTALLATION
------------
No installation is required to run the dashboard on the pre-generated data. Simply follow the steps below:

1. Download the three CSV files from the provided link.
2. Place the CSV files in the same folder as Final.html, within the CODE/ directory.
3. Open a terminal and navigate to the CODE/ directory:
    `1cd CODE1`
4. Start a local HTTP server:
    `python -m http.server`
5. Open your web browser and go to http://localhost:8000/Final.html to launch the dashboard.

To generate the CSV files from a new dataset:
1. Ensure you have a compatible environment with GPU support.
2. Install the required Python packages:
    `pip install -r requirements.txt`
3. Run the topic modeling script:
    `python topic.py --input_path <you filepath>`

---------
EXECUTION
---------
Once the dashboard is launched in your browser, the main visualization page will be displayed. Wait for all four graphs to load completely. From there, you can:

1. Use filters to narrow down topics or time ranges.
2. Hover over data points for detailed insights.
3. Explore trends in Bitcoin-related discussions with an intuitive and interactive interface.

This dashboard is designed to support real-time exploration and analysis, making it a powerful tool for social analytics in the cryptocurrency space.