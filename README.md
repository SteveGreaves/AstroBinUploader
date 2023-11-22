# README.md for GitHub Repository

## Features

This Python script is designed for astronomers and astrophotographers to process and analyze FITS (Flexible Image Transport System) file headers. Key features include:

- **Extraction of FITS Headers**: Automatically extracts headers from FITS files in specified directories.
- **Session Summarization**: Summarizes observation sessions by processing LIGHT, FLAT, BIAS, and DARK frames.
- **Calibration Data Handling**: Creates a DataFrame for calibration data like DARK, BIAS, FLAT, and FLATDARKS.
- **LIGHT Data Processing**: Aggregates 'LIGHT' type data for detailed analysis.
- **Sky Quality Analysis**: Retrieves Bortle scale classification and SQM values based on the location's coordinates.
- **Auxiliary Parameter Calculation**: Calculates additional parameters like HFR, IMSCALE, and FWHM for each observation.
- **AstroBin Compatibility**: Formats aggregated data for easy uploading to AstroBin.

## Prerequisites

Before using this script, ensure you have the following installed:

- Python 3.x
- Pandas library
- Astropy library
- Requests library

## Installation and Usage

To use this script, follow these steps:

1. Clone or download the repository from GitHub.
2. Ensure you have all the required libraries installed. If not, use `pip install pandas astropy requests`.
3. Place the script in a directory with your FITS files or specify the path to the directories as command-line arguments.
4. Run the script using Python: `python script_name.py [directory_path1] [directory_path2] ...`.

## External Files

The script requires the following external CSV files for its operation:

- **filter.csv**: Contains mappings of filter names to their codes.
- **sites.csv**: Stores latitude, longitude, Bortle scale, and SQM values for known locations.
- **secret.csv**: Holds API keys and endpoints for external services (e.g., light pollution data).

Ensure these files are present in the same directory as the script or update their paths in the script accordingly.

## Additional Information

This script is intended for educational purposes in the field of astronomy. It is part of an open-source project and contributions or suggestions for improvements are welcome.

## Filter.csv
Astrobin requires the user provide a 4 digit code that represents the filter being used. The code can be obtained from the url in the equipment section for the filter being used.
For instance for an Astronomik H-alpha CCD 6nm 2 inch filter the equipment explorer url is:

https://app.astrobin.com/equipment/explorer/filter/4663/astronomik-h-alpha-ccd-6nm-2 

The code in this case is 4663.

The filter.csv file contains the filter name and the corresponding code. The filter.csv contents for my filters is given below:

|Filter|Code|
|------|----|
|Ha    |4663|
|SII   |4844|
|OIII  |4752|
|Red   |4649|
|Green |4643|
|Blue  |4637|
|Lum   |2906|
|CLS   |4061|

It's a pain to collect these but once you have them you are set. Ensure the filter.csv file is in teh same directory as the python scipt and the folder containing your astoimages.

# AstroBin's Acquisition CSV File Format

## AstroBin Acquisition Detail Entry

AstroBin allows users to enter acquisition details associated with uploaded images via a CSV file. This document outlines the format for the `acquisition.csv` file, which can be automatically generated from image file data to populate acquisition details on AstroBin.

## AstroBin Long Exposure Acquisition Fields



| **Field** | Description | Validation |
| --------- | ----------- | ---------- |
| **date** | The date when the acquisition took place | YYYY-MM-DD format |
| **filter** | Filter used | Numeric ID of a valid filter (found in the URL of the filter's page in the equipment database) |
| **number*** | Number of frames | Whole number |
| **duration*** | Duration of each frame in seconds | Number, Max decimals: 4, Min value: 0.0001, Max value: 999999.9999 |
| **iso** | ISO setting on the camera | Whole number |
| **binning** | Binning of pixels | One of [1, 2, 3, 4] |
| **gain** | Gain setting on the camera | Number, Max decimals: 2 |
| **sensorCooling** | The temperature of the chip in Celsius degrees, e.g., -20 | Whole number, Min value: -274, Max value: 100 |
| **fnumber** | If a camera lens was used, specify the f-number used for this acquisition session | Number, Max decimals: 2, Min value: 0 |
| **darks** | The number of dark frames | Whole number, Min value: 0 |
| **flats** | The number of flat frames | Whole number, Min value: 0 |
| **flatDarks** | The number of flat dark frames | Whole number, Min value: 0 |
| **bias** | The number of bias/offset frames | Whole number, Min value: 0 |
| **bortle** | Bortle dark-sky scale | Whole number, Min value: 1, Max value: 9 |
| **meanSqm** | Mean SQM mag/arcsec^2 as measured by a Sky Quality Meter | Number, Max decimals: 2, Min value: 0 |
| **meanFwhm** | Mean Full Width at Half Maximum in arc seconds, a measure of seeing | Number, Max decimals: 2, Min value: 0 |
| **temperature** | Ambient temperature in Celsius degrees | Number, Max decimals: 2, Min value: -88, Max value: 58 |


# Example of an `acquisition.csv` File for Part of an Observing Session

| **date** | **filter** | **number** | **duration** | **binning** | **gain** | **sensorCooling** | **darks** | **flats** | **flatDarks** | **bias** | **meanSqm** | **meanFWHm** | **temperature** |
| -------- | ---------- | ---------- | ------------ | ----------- | -------- | ----------------- | --------- | --------- | ------------- | -------- | ----------- | ------------- | -------------- |
| 2023-08-15 | 4663 | 9 | 600.0 | 1 | 100 | -10 | 50 | 150 | 0 | 100 | 20.4 | 3.6 | 10.7 |
| 2023-08-17 | 4663 | 19 | 600.0 | 1 | 100 | -10 | 50 | 150 | 0 | 100 | 20.4 | 3.6 | 13.3 |
| 2023-08-28 | 4663 | 2 | 600.0 | 1 | 100 | -10 | 50 | 150 | 0 | 100 | 20.4 | 3.6 | 11.8 |
| 2023-08-28 | 4844 | 7 | 600.0 | 1 | 100 | -10 | 50 | 150 | 0 | 100 | 20.4 | 3.6 | 10.4 |
| 2023-09-02 | 4663 | 16 | 600.0 | 1 | 100 | -10 | 50 | 150 | 0 | 100 | 20.4 | 3.6 | 13.6 |
| 2023-09-03 | 4663 | 14 | 600.0 | 1 | 100 | -10 | 50 | 150 | 0 | 100 | 20.4 | 3.6 | 11.3 |
| 2023-09-03 | 4844 | 16 | 600.0 | 1 | 100 | -10 | 50 | 150 | 0 | 100 | 20.4 | 3.6 | 15.9 |
| 2023-09-04 | 4844 | 27 | 600.0 | 1 | 100 | -10 | 50 | 150 | 0 | 100 | 20.4 | 3.6 | 13.7 |
| 2023-09-15 | 4752 | 33 | 600.0 | 1 | 100 | -10 | 50 | 150 | 0 | 100 | 20.4 | 3.6 | 12.7 |
| 2023-09-16 | 4752 | 25 | 600.0 | 1 | 100 | -10 | 50 | 150 | 0 | 100 | 20.4 | 3.6 | 11.8 |
| 2023-09-18 | 4844 | 4 | 600.0 | 1 | 100 | -10 | 50 | 150 | 0 | 100 | 20.4 | 3.6 | 11.1 |

The fields in the CSV can be obtained or derived from the FITS header files of the captured images. However, the filter value requires translation from the filter name (e.g., Ha or OIII) as reported in the FITS header to the numeric ID used by AstroBin. To facilitate this translation, a `filter.csv` file is created to define the mapping between filter names and AstroBin filter codes. Below is an example showing a `filter.csv` file that describes a typical filter set:




| **Filter** | **Code** |
| ---------- | -------- |
| Ha | 4663 |
| SII | 4844 |
| OIII | 4752 |
| Red | 4649 |
| Green | 4643 |
| Blue | 4637 |
| Lum | 2906 |
| CLS | 4061 |



### Finding the Numeric ID for Filters

As indicated in the validation column for the filter field, the numeric ID of a valid filter can be found by examining the URL of the filter's page in the AstroBin equipment database. For example, consider a Ha filter: a 2-inch H-alpha CCD 6nm filter from Astronomik. By using AstroBin's equipment explorer to navigate to this filter's page, the URL might look something like:

`https://app.astrobin.com/equipment/explorer/filter/4663/astronomik-h-alpha-ccd-6nm-2`

From this URL, the code for this particular filter is identified as 4663.

### Creating the filters.csv File

The `filters.csv` file can be created using a text editor and should be located in the directory from which the Python script is executed. The script relies on this file to ensure the correct codes are used for your filters.


## Contributing to AstroImage Processing Script

To contribute to this project, follow these steps:

1. Fork this repository.
2. Create a branch: `git checkout -b <branch_name>`.
3. Make your changes and commit them: `git commit -m '<commit_message>'`.
4. Push to the original branch: `git push origin <project_name>/<location>`.
5. Create the pull request.

Alternatively, see the GitHub documentation on [creating a pull request](https://docs.github.com/en/github/collaborating-with-issues-and-pull-requests/creating-a-pull-request).

## Contact

If you want to contact me, you can reach me at `<sgreaves139@gmail>`.

## License

This project uses the following license: [GNU General Public License v3.0](https://github.com/SteveGreaves/AstroBinUploader/blob/main/LICENSE).

