# Raw Survey Data Automated Cleaning and Processing  (python, duckdb)

## This repository contains the pipeline for cleaning, standardizing, and enhancing survey data, including geographic information.

## Challenge
- Transform Raw Data: convert raw survey responses into a **clean, structured, and enriched dataset.**
- Prepare for Analysis: process to ensure the data is ready for in-depth analysis and reporting.
- Enhance Location Data: improve the accuracy of geographical information by adding precise coordinates.
- Build Reliable Final Dataset: outcome is a solid, trustworthy dataset for generating insights.

## Task: Data Pipeline Process
The processing pipeline generates a comprehensive set of structured data outputs, including:

- **Main Survey Data:** A primary, cleaned dataset containing core demographic and survey responses.
- **Specialized Tables:** Several detailed tables extracted from multi-select questions, such as:

	- Methods of success in career
	- What used (Skills like SQL, Python, R, etc.)
	- Specific roles and responsibilities
	- Digital learning platforms
	- Data engineering tools for ingestion, transformation, orchestration, etc.

- **Geocoded CSV:** A new CSV file with standardized and geocoded location information.  [python code for map](https://github.com/SandyGCabanes/2024-Survey-Report-on-the-State-of-the-Community-DEP/blob/main/map3_geopy_folium.ipynb)
- **Interactive HTML Map:** An HTML map visualizing the geocoded locations, providing a clear geographic overview.


## Results: Raw Survey Data Cleaning Automation

- This automation process saves time by cutting data cleaning time in half or less vs. manual cleaning in excel, which could take weeks.
- The cleaned and validated data can easily be imported into any business intelligence software like Tableau, Power BI, Looker Studio even Excel charts. [Tableau report here.](https://public.tableau.com/app/profile/sandy.g.cabanes/viz/survey0309/Home)
- The actual pickle files, csv files and duckdb file are withheld for privacy reasons.  A anonymized synthetic data are available here.
	- [Anonymized Real-World Data Using Bayesian Networks to Protect Privacy - Python](https://github.com/SandyGCabanes/Anonymized-Survey-Data-Modeling-with-Bayesian-Networks-in-Python)
	- [Privacy Protection Using Bayesian Networks on Real World Survey Data to Produce Anonymized Dataset - R](https://github.com/SandyGCabanes/Survey-Data-Privacy-Protection-Using-R-and-Bayesian-Networks)
  
## Background: Automation of Data Pipeline
> This project addresses the need for automating data cleaning and data validation of annual survey data in python. After the initial survey in 2024, we are now launching this as an annual survey.  Raw data contains inconsistencies, missing values, and variations in free-text entries. This Python workflow automates the rigorous cleaning, standardization, and transformation of this data. A key enhancement includes a multi-geocoder validation process to accurately identify and map respondent locations, which will provide a snapshot of respondent locations. This uses a modular design, ensuring consistency across various data manipulation steps.




> SGC. 
> Data-powered decisions.
