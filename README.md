# Astroimage Processing Script

This script processes FITS (Flexible Image Transport System) files for astro-imaging. It extracts header data from FITS files, organizes the data into a structured format, and exports it to a CSV file. This tool creates summary of acquisition session information that is suitable for using with AstroBin..

## Features

- Extracts headers from all FITS files in a specified directory and its subdirectories.
- Categorizes images into different types (e.g., LIGHT, DARK, BIAS, FLAT, FLATDARK).
- Generates a calibration DataFrame based on specific image types and gains.
- Creates a lights DataFrame with detailed information, replacing filter names with codes.
- Exports the processed data into a CSV file sutible for pasting into to Astrobin's aqusition data input dialogue when using the import from CSV option.

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

