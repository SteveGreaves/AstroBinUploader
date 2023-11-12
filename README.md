# Astroimage Processing Script

I really dislike adding session capture information to my AstroBin uploads. When you have many sessions it is quite tedious collecting, analysing and the entering the data.
For some of my longer imaging session I took to adding the data to the description section, which is not the best way. To resolve this issue I created a simple python script.

This script processes FITS (Flexible Image Transport System) files for astro-imaging. It extracts header data from FITS files, organizes the data into a structured format, and exports it to a CSV file. The script creates summary of acquisition session information that is suitable for using with AstroBin.

## Features

- Extracts headers from all FITS files in a specified directory and its subdirectories.
- Categorizes images into different types (e.g., LIGHT, DARK, BIAS, FLAT, FLATDARK).
- Generates  a dataFrame that contains information from calibration frames based on specific image types and gains.
- Creates a DataFrame with detailed Lights information, replacing filter names with codes.
- Uses this information to create a CSV file that meets AstroBin's data aqusition format. 
- Exports the processed data into a CSV file. The contents are sutible for pasting into to Astrobin's aqusition data input dialogue when using the import from CSV option.

-The python script is OS agnostic, so you will be able to run this on Windows, Linux or MACOS as long as you ensure the installation pre-requsites are met.

## Prerequisites

Before you begin, ensure you have met the following requirements:

- Python 3.x installed
- `pandas` library installed: Run `pip install pandas`
- `astropy` library installed: Run `pip install astropy`
- Git (optional, for version control)

## Installation and Usage

To use the AstroImage Processing Script, follow these steps:

1. Clone the repository or download the script to your local machine.
2. Place your FITS files in a specific directory.
3. Ensure you have a filter file (CSV format) that contains filter names and their corresponding codes.
4. Run the script via a command line or an IDE like VS Code.
5. Follow the prompts to input the necessary data (directory path, filter file name, etc.).
6. The script will process the FITS files and export the data to a CSV file.

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
## Contributing to AstroImage Processing Script

To contribute to this project, follow these steps:

1. Fork this repository.
2. Create a branch: `git checkout -b <branch_name>`.
3. Make your changes and commit them: `git commit -m '<commit_message>'`.
4. Push to the original branch: `git push origin <project_name>/<location>`.
5. Create the pull request.

Alternatively, see the GitHub documentation on [creating a pull request](https://docs.github.com/en/github/collaborating-with-issues-and-pull-requests/creating-a-pull-request).

## Contributors

Thanks to the following people who have contributed to this project:

- [@steveGreaves](https://github.com/SteveGreaves/)

## Contact

If you want to contact me, you can reach me at `<sgreaves@gmail>@example.com`.

## License

This project uses the following license: [GNU General Public License v3.0](https://github.com/SteveGreaves/AstroBinUploader/blob/main/LICENSE).

