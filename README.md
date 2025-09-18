# Data Processing Workflow for Survey Data (python, duckdb)

## This repository contains the pipeline for cleaning, standardizing, and enhancing survey data, including geographic information.

## Objective
- Transform Raw Data: convert raw survey responses into a **clean, structured, and enriched dataset.**
- Prepare for Analysis: process ensures the data is ready for in-depth analysis and reporting.
- Enhance Location Data: improve the accuracy of geographical information by adding precise coordinates.
- Build Reliable Foundation: outcome is a solid, trustworthy dataset for generating insights.

## Tracking Survey Automation

- This automation process will facilitate year to year comparison in the future.

- The cleaned and validated data can easily be imported into any business intelligence software like Tableau, Power BI, Looker Studio even Excel charts. [Tableau report here.](https://public.tableau.com/app/profile/sandy.g.cabanes/viz/survey0309/Home)

## Background: Automation of Data Pipeline
> This project addresses the need for automating data cleaning and data validation of annual survey data in python. After the initial survey in 2024, we are now launching this as an annual survey.  Raw data contains inconsistencies, missing values, and variations in free-text entries. This Python workflow automates the rigorous cleaning, standardization, and transformation of this data. A key enhancement includes a multi-geocoder validation process to accurately identify and map respondent locations, which will provide a snapshot of respondent locations. This uses a modular design, ensuring consistency across various data manipulation steps.

## Data Pipeline Process
The processing pipeline generates a comprehensive set of structured data outputs, including:

- **Main Survey Data:** A primary, cleaned dataset containing core demographic and survey responses.
- **Specialized Tables:** Several detailed tables extracted from multi-select questions, such as:

	- Methods of success in career
	- What used (Skills like SQL, Python, R, etc.)
	- Specific roles and responsibilities
	- Digital learning platforms
	- Data engineering tools for ingestion, transformation, orchestration, etc.

- **Geocoded CSV:** A new CSV file with standardized and geocoded location information.
- **Interactive HTML Map:** An HTML map visualizing the geocoded locations, providing a clear geographic overview.

> SGC. 
> Beyond surveys. Data-Powered decisions.
