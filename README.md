# AstroBin Upload Utility v2.1.2
Scripts to process FITS/XISF headers and create Astrobin data acquisition file and summary text.

Usage:
`python3 AstroBinUpload.py [directory_paths] [--config config_file]`

## **Contents**

- - [Features](#features)   
- [Prerequisites](#prerequisites)    
    - [Installation of the AstroBinUpload.py script](#installation-of-the-astrobinuploadpy-script)
    - [Initialization and Config.ini generation](#initialization-and-configini-generation)
    - [Using Alternative Configuration Files](#using-alternative-configuration-files)
    - [Config.ini contents and editing](#configini-contents-and-editing)
        - [[defaults]](#defaults)
        - [[filters]](#filters)
        - [[secrets]](#secrets)
        - [[sites]](#sites)
        - [[override]](#override) 
        - [[equipmentoverrides]](#equipmentoverrides)
        - [Editing the initial config.ini](#editing-the-initial-configini)
- [Running the Script](#running-the-script)
    - [Initialization (no arguments)](#initialization-no-arguments-are-passed)
    - [Single directory or symbolic link](#a-single-directory-path-or-symbolic-link-argument-is-passed-to-the-script)
    - [Multiple directory paths or symbolic links](#multiple-directory-paths-or-symbolic-links)
    - [Advanced Debugging and Testing](#advanced-debugging-and-testing)
- [Example calls and outputs](#example-calls-and-outputs)
    - [Example 1: Single site, non-mosaic](#example-1-single-site-non-mosaic-no-masters-data-resides-in-structured-single-directory-symbolic-links-used-for-calibration-data)
    - [Example 2: Single site, 2 panel mosaic](#example-2-single-site-2-panel-mosaic-symbolic-links-to-calibration-data-use-of-masterflats)
    - [Example 3: Dual site, structured directory](#example-3-dual-site-structured-directory-2-panel-mosaic-use-of-mastercals)
    - [Example 4: WBPP two-panel mosaic](#example-4-wbpp-two-panel-mosaic)
- [Troubleshooting & Installation Tips](#troubleshooting--installation-tips)
- [References](#references)   
    - [AstroBin's Acquisition CSV File Format](#astrobins-acquisition-csv-file-format)
        - [AstroBin's Long Exposure Acquisition Fields](#astrobin-long-exposure-acquisition-fields)
    - [Astrobin Filter-Code mappings](#astrobin-filter-code-mappings)
        - [Finding the AstroBin's Numeric ID for Filters](#finding-astrobins-numeric-id-for-filters)
    - [Accessing sky quality data](#accessing-sky-quality-data)
    - [Reverse Geocoding](#reverse-geocoding)
    - [FWHM values](#fwhm-values)
    - [Data Sources](#data-sources )
- [Contributing to AstroBinUpload.py Processing Script](#contributing-to-astrobinuploadpy-processing-script)
- [Contact](#contact)
- [Licence](#licence)

<div style="page-break-after: always;"></div>

## **Features**

When run this Python script creates a detailed observation session summary and an acquisition.csv file suitable for upload using AstroBin's import CSV dialogue. 

Data is obtained by extracting FITS (Flexible Image Transport System) or XISF (Extensible Image Serialization Format) headers from image and calibration files associated with the given astronomical target.

Key features include:

- **The ability to pass multiple directories via the command line**: Multiple directories can be passed to the script via the command line. All images results contained within the directories will be accumulated as part of the target.  

- **Structured and unstructured directories**: Image files, including calibration files, can be collected into a single directory, the root directory. The root directory structure can be flat or contain subdirectories. 

- **Symbolic links to directories**: Symbolic links can be used within the root directory or passed directly via the command line, this is useful when reusing calibration directories. The first directory passed should be the root directory.     

- **MASTER calibration files**: If MASTER calibration files are found, these will be used. If the non-MASTER versions of the MASTER files are also found, the non-MASTER versions will be ignored. 

- **Processing of PixInsight's Weighted Batch Pre-processing (WBPP) output**: When the target is a WBPP directory the script will use the calibrated LIGHT frames as well as any MASTER calibration files found in the directory. MASTERLIGHT or processed image files are ignored. 

- **Multiple panel mosaic imaging sessions**: Mosaic imaging sessions are detected from the OBJECT entry in the FITS headers. LIGHT frames are processed on a per-panel basis, whilst calibration data is processed per target. For this to work correctly the image names (OBJECT in FITS header) must have the format  

        "target name Panel x"   

    where x is the panel number. N.I.N.A does this automatically but in Sequence Generator Pro the user will have to edit the directory name in Target Settings before starting the sequence.

- **Multiple site support**: Multi-site collaborative target acquisition or remote observatory image capture is supported. Site location data is reverse-geocoded from HEADER location data. Data from multiple sites is reported with summary outputs that correctly identify the site contribution, for instance equipment, LIGHT, and calibration data. All data is, however, aggregated in the AstroBin.csv file for the image target.

- **Support for multiple file formats**: Extracts headers for all FITS/FIT/FTS/XISF files in specified directories. Directories can have a mix of files. 

- **Accepts files generated by N.I.N.A, SGPro, ACP, MaximDL and PixInsight** 

- **Sky Quality Retrieval**: Recovers SQM and Bortle scale classification based on the observation location coordinates. 

- **Auxiliary Parameter Calculation**: Calculates additional parameters like Image Scale (IMSCALE), and Full-width Half Maximum (FWHM) from measured/estimated HFR values for each image. 

- **AstroBin Compatibility**: Formats aggregated data for upload to AstroBin's import CSV file dialogue.

- **Target summary**: Creates a detailed summary text file for a target acquisition session. Caters for single or multi-site data as well as single or mosaic imaging data.

- **Summary files stored in the working directory**: Creates a folder called AstroBinUploadInfo in the current image working directory. Files saved are:

    - ***AstroBinUploader.log***: log file for current session

    - ***acquisition.csv***: session summary in the correct format to copy and paste to AstroBin's import CSV dialogue.

    - ***session_summary.txt***: copy of the detailed session summary that is output to the screen by the script

    - ***Debug files***: Files created when the --debug switch is used


## **Pre-requisites**

Before using this script, ensure you have Python 3.x installed. [Python Installation Instructions](https://python.land/installing-python).   

### **Installation of the AstroBinUpload.py script**

To install this script, follow these steps:

1. Create a directory to hold the script and associated files

2. Clone or download the repository from my [GitHub repository](https://github.com/SteveGreaves/AstroBinUploader)

3. Ensure the following are in the new directory:
    - `AstroBinUpload.py` — entry point
    - `constants.py`, `models.py`, `_version.py`
    - `engine/` — the pipeline package (`loader.py`, `extractor.py`,
      `processor.py`, `exporter.py`, `reports.py`, and `engine/steps/`)
    - `requirements.txt`

4. To install the required python libraries, navigate to the new directory and run the following command:   

    `pip install -r requirements.txt`

### **Initialization and Config.ini generation**
The script is first run with no arguments. This will create a default config.ini file in the local directory. The config.ini file defines parameters the script needs to run correctly. This file can be personalized by the user. A description of the config.ini file parameters is given later. Then you run the script for the first time or if your config.ini file has been deleted, you will see the following:

        'A new config.ini file was created. Please edit this before re-running the script.' 

The script will halt, allowing you to edit the config.ini file using a text editor. You can then the re-run the script. 

If you run the script when the config.ini file exists and pass no arguments, an error will be flagged, and the script will exit.

### **Using Alternative Configuration Files**

Version 2.1.1 and later allow you to specify a custom configuration file using the `--config` (or `-c`) flag. 

    python3 AstroBinUpload.py "/path/to/my/data" --config my_remote_setup.ini

This is ideal for users who manage different setups (e.g., Mono vs Color, Remote vs Local) and wish to switch profiles without renaming files to `config.ini`. If the specified file does not exist, the script will report an error and exit.

### **Config.ini contents and editing**
The config.ini file contains the following sections:
### **[defaults]**

The [defaults] section holds: 
1. Default FITS keywords that should be present in the header files and that are needed in the production of the astrobin.csv output file
2. Default site location and auxiliary information, used in the production of the summary.txt output file

The data in the [defaults] section can be modified by the user. If the script cannot find the information in the header files it will take it from the [defaults] section of the config.ini file

### **[filters]**
The filter section holds the filter name to AstroBin code mappings. The filter names and codes can be modified here.

### **[secrets]** — removed in v2.1.0

**This section is no longer used and can be deleted from your `config.ini`.**

Earlier versions called two external APIs: lightpollutionmap.info for Bortle
and SQM, and Nominatim reverse geocoding for the site address. Neither is
called any more — the utility is fully offline. Site information now comes
from the `[sites]` section (matched by coordinate), falling back to
`[defaults]` (`SITE`, `SITELAT`, `SITELONG`, `BORTLE`, `SQM`) when no site
matches. Nothing reads an API key or an email address, so leaving the old
section in place is harmless but pointless.

### **[sites]**
The `[sites]` section is your own list of known observing sites. The utility
clusters the GPS coordinates found in your headers (drifting readings within
about 110 m are treated as one site) and looks the resulting position up here.
A match supplies the site's name, Bortle and SQM; no match falls back to
`[defaults]`.

**You maintain this section yourself.** Earlier versions appended new sites
automatically after a reverse-geocoding call; since v2.1.0 there are no API
calls, so add a site by copying the block format shown below and filling in
the coordinates your headers actually carry (`--debug` writes them to
`debug_step_00_RawHeaders.csv` if you need to look them up).

### **[override]**

The [override] section provides a translation layer that allows you to map non-standard FITS keywords to the internal keywords used by the script. This is particularly useful for capture software or camera drivers that utilize alternative naming conventions for standard parameters. 

Users can maintain their own list of FITS keywords in the `config.ini` under the `[override]` section. This allows for mapping standard internal variables to custom hardware keys provided by specific drivers or devices. The script also supports a comma-separated list of keys to handle hardware variations (e.g., different versions of the same sensor).

Example:
```ini
[override]
SQM = AOCSKYQ, AOCSKYQU
FOCTEMP = AOCAMBT
```

In this example, the script will first look for `AOCSKYQ`, and then `AOCSKYQU` if the first is not found, to populate the internal `SQM` variable. The script prioritizes these manual overrides over standard defaults. Once a mapping is successful, the source hardware column is pruned to ensure a clean data hand-off to the aggregator.

### **[equipmentoverrides]**

New in v2.1.1. Where `[override]` remaps a *keyword* (read `AOCAMBT` and call
it `FOCTEMP`), `[equipmentoverrides]` replaces a *value* — it forces a literal
string into a column for every frame, whatever the header actually said.

The case it exists for: N.I.N.A. writes `EAF` for a ZWO focuser, and you want
the AstroBin summary to say `ZWO EAF`.

```ini
[equipmentoverrides]
        INSTRUME = None
        TELESCOP = None
        FOCNAME = ZWO EAF
        FWHEEL = None
        ROTNAME = None
```

`None` (the generated default) or an empty value means "leave whatever the
header carried". Anything else is forced into that column for every row,
applied immediately after default injection. It works on any column, not just
the five equipment fields the generated template lists.

It is advisable to back up you config.ini file regularly. 

<div style="page-break-after: always;"></div>

## **Editing the config.ini**
A config.ini file with an explanation of the sections is given below:

```
[defaults]
        IMAGETYP = LIGHT
        EXPOSURE = 0.0
        DATE-OBS = 2023-01-01
        XBINNING = 1
```
These are place-holders and are fall backs. These parameters should be created by the capture software.

```
        GAIN = -1
        EGAIN = -1
```
If you have a CCD camera, leave these values as they are. If you have a CMOS camera these gain values can be set to the typical values of your camera. The script will, however, collect the correct results for both CCD and CMOS cameras from the headers processed. 
```
        INSTRUME = None
        TELESCOP = None
        FOCNAME = None
        FWHEEL = None
        ROTNAME = None
        ROTANTANG = 0
        XPIXSZ = 1
        CCD-TEMP = -10
        FOCALLEN = 540
        FOCRATIO = 5.4
```
This is where default the equipment configuration. Again the script should be able to populate these parameters from the header information. XPIXSZ is the X-pixel size in um and is used to represent the sensor pixel size in the script.
```
        SITE = Papworth Everard
        SITELAT = 52.2484
        SITELONG = -0.1231
        BORTLE = 4
        SQM = 20.5
```
You should modify these parameters to reflect your own site. If any API call fails the script will fall back to these parameters

<div style="page-break-after: always;"></div>

```
        FILTER = No Filter
```
If you use a color camera and don't report your filters automatically you should enter your filter name here as there may be no filter information in the header.
```
        OBJECT = No target
        FOCTEMP = 20
```
These are place-holders and should not be required as they should be populated by the capture software

```
        HFR = 1.6
```
You should set this value to the typical value for your imaging train. If you use N.I.N.A you can add the measured HFR for the image to the file name. The script looks for HFR=X.XX in the image file name and if present uses the value found, if HFR is not in the image file name the script falls back to this value.
```
        SWCREATE = Unknown package
```
This is a place-holder and should not be required, it should be created by the capture software.
```
        USEOBSDATE = TRUE
```
USEOBSDATE if set to True the actual date of the observation session is used when aggregating data for the astrobin acqusition.csv output. If this prameter is set to False then, per session, the date the observation session was started is used. 
```


[filters]
        #Filter     code
        Ha        = 4663
        SII       = 4844
        OIII      = 4752
        Red       = 4649
        Green     = 4643
        Blue      = 4637
        Lum       = 2906

```
Modify the [filters] section to reflect your imaging set up, see [Astrobin Filter-Code mappings](#astrobin-filter-code-mappings) for information on how to populate this table. If you use a color camera and don't report your filters automatically you should enter your filter name and the corresponding code here. Delete filters you don't require.

<div style="page-break-after: always;"></div>


A `[secret]` section appears in configs generated by v2.0.x and earlier. It
held a sky-quality API key and an email address for reverse geocoding. **It is
no longer read** — since v2.1.0 the utility makes no network calls at all —
and can be deleted.

```
[sites]
```
Site information comes from the `[sites]` section, which you maintain by hand
(earlier versions appended to it automatically after a geocoding call). A site
block looks like this — the coordinates are matched against the clustered
positions found in your headers:

```
[sites]
        [["Norton Close, Papworth Everard, South Cambridgeshire, Cambridgeshire, Cambridgeshire and Peterborough, England, CB23 3XT, United Kingdom"]]
                latitude = 52.2484
                longitude = -0.1231
                bortle = 4
                sqm = 20.52
```
When the script processes [SITELAT] and [SITELONG] header entries it looks here first to see if a site has been seen before. If it has the script uses the site information found, if not it calls the external APIs to retrieve the information. If the external API call fails the script falls back to site parameters found in the [defaults] section of the configs.ini file. New sites can also be manually added in the [sites] section following the format shown above.

```
[override]

        # Internal Key = Alternative FITS Keyword
        SITE = SITENAME
        EXPOSURE = EXPTIME
        INSTRUME = CAMERA_MODEL
```
The [override] section provides a translation layer that allows you to map non-standard FITS keywords to the internal keywords used by the script. This is particularly useful for capture software or camera drivers that utilize alternative naming conventions for standard parameters.



## **Running the Script**

The script is called from the command line. There are three calling methods:

### **Initialization (no arguments)**:  

    Linux/MACOS example: python3 AstroBinUpload.py 
    Windows example:     python  AstroBinUpload.py 

When called with no argument the script will create a new default config.ini file in the local directory and then exit. The user can then edit the config.ini file, using a text editor, before running the script again. If an existing config.ini file is lost and the script run, a new default config.ini file will be created, and the code will exit. Once you have personalized your config.ini file make a backup.

### **A single directory path or symbolic link**

 Note: only Linux calling examples are used going forward.

    python3 AstroBinUpload.py "dir 1" 

The script expects to find all data contained in the directory passed to it. Symbolic links can be used as the argument passed to script and can also be present in the directory. The directory leaf or child directory name must be the target name if the output files are to be named correctly. From the processing perspective the only condition required to ensure data is associated with a given target is that all data and links must reside in the one directory.

### **Multiple directory paths or symbolic links** 

    python3 AstroBinUpload.py "dir 1" "dir 2" .... 

All directory arguments are assumed to belong to one target. Again the first directory leaf, or child directory name should contain the target name for the output files to be named correctly.

### **Advanced Debugging and Testing**

Version 2.1.1 and later provide a robust diagnostic system designed for high-precision troubleshooting and workflow verification.

#### **1. Generating Debug Data**
To inspect the internal state of your metadata as it flows through the pipeline, run the utility with the `--debug` flag:

    python3 AstroBinUpload.py "/path/to/my/data" --debug

When enabled, the utility generates a sequence of CSV files in the `AstroBinUploadInfo` directory:
- **`debug_step_00_RawHeaders.csv`**: The exact metadata extracted from your files BEFORE any processing.
- **`debug_step_01_NormalizeHeadersStep.csv`**: Data after hardware overrides and sanitization.
- **`debug_step_04_CalibrationMatcherStep.csv`**: Data after calibration frames have been assigned.
- **`debug_step_06_AggregationStep.csv`**: The final grouped statistics.

#### **2. Using the Diagnostic Test Mode (`--test`)**
The `--test` flag allows you to re-run the entire pipeline using a CSV file instead of scanning your hard drive. This is ideal for verifying configuration changes or reproducing bugs.

**Crucial Note**: Because the test mode injects data at the very beginning of the pipeline, **you must only use files containing raw metadata**. 

**Supported Files for `--test`:**
1.  **`debug_step_00_RawHeaders.csv`**: Use this for standard testing. It is generated every time you run a successful scan with `--debug`.
2.  **`emergency_raw_dump.csv`**: Use this for crash recovery. It is generated automatically if the utility encounters a fatal error during a scan.

**Example Usage:**
    
    python3 AstroBinUpload.py "/path/to/my/data" --test "/path/to/debug_step_00_RawHeaders.csv"

#### **3. Error Handling and Logging**
If the utility encounters a fatal error, it automatically performs an "Emergency Dump":
- A full Python traceback is recorded in `AstroBinUploader.log`, identifying the exact line of failure.
- Whatever metadata was successfully scanned is saved to **`emergency_raw_dump.csv`**. 
- This dump can be fed directly back into the utility using the `--test` flag once the issue is resolved.

The log file also includes a **Horizontal Header Echo**, which prints the full raw metadata dictionary for every file processed (visible when logging level is set to DEBUG).


## **Diagnostic Mode**

The `--test` flag allows developers and users to troubleshoot issues using a `.csv` file (typically `basic_headers.csv` generated via the `--debug` run) without needing access to the raw FITS data. This ensures consistent logic verification across different environments.

Example usage:
`python3 AstroBinUpload.py "/path/to/data" --test "basic_headers.csv"`

**Note:** The CSV file must reside within the first directory provided in the command line arguments.
<div style="page-break-after: always;"></div>

# **Example calls and outputs**

## **Example 1: Single site, non-mosaic** 

![Alt text](images/image-1.png)

### Example 1: Directory structure

    python3 AstroBinUpload.py "/mnt/preselected/Sadr Region"


### Example 1: Script calling syntax

The output files being named:     
- Sadr_Region_session_summary.txt
- Sadr_Region_aquisition.csv  


<div style="page-break-after: always;"></div>


![Alt text](images/image-2.png)

### Example 1: Summary output

<div style="page-break-after: always;"></div>

![Alt text](images/image-3.png)

### Example 1: AstroBin.csv output


## Example 2: Single site, 2 panel mosaic

![Alt text](images/image-4.png)

### Example 2: Directory structure

    python3 AstroBinUpload.py '/mnt/preselected/NGC 1499 Mosaic'
    
    or using a symbolic link:
    
    python3 AstroBinUpload.py '/home/steve/Desktop/AstroData/Link to NGC 1499 Mosaic'

### Example 2: Script calling syntax

<div style="page-break-after: always;"></div>

![Alt text](images/image-5.png)

### Example 2: Summary Output

<div style="page-break-after: always;"></div>

![Alt text](images/image-6.png)

### Example 2: AstroBin.csv output

## Example 3: Dual site, structured directory

Note: although data is reported on a per-site basis, data is aggregated from all sites to create the AstroBin.csv output. Symbolic links can also be used. A non-structured directory can also be used as long as all files reside under the main target directory. Mosaics can be generated per site, however, summary files can be quite large.

![Alt text](images/image-7.png)

### Example 3: Directory structure

![Alt text](images/image-8.png)

### Example 3: Script calling syntax

    python3 AstroBinUpload.py "/mnt/preselected/AstroBinTest/M51"

<div style="page-break-after: always;"></div>

### Example 3: Summary Output

### Site 1

![Alt text](images/image-9.png)

<div style="page-break-after: always;"></div>

### Site 2

![Alt text](images/image-10.png)

<div style="page-break-after: always;"></div>

![Alt text](images/image-11.png)

### Example 3: AstroBin.csv output

## **Example 4: WBPP two-panel mosaic**

![Alt text](images/image-12.png)

![Alt text](images/image-13.png)

![Alt text](images/image-14.png)

### Example 4: Directory structure

    python3 AstroBinUpload.py "/home/steve/Desktop/Current PixInsight Projects/California Nebula (NGC1499)"

### Example 4: Script calling syntax

<div style="page-break-after: always;"></div>    

![Alt text](images/image-15.png)

### Example 4: Summary Output

<div style="page-break-after: always;"></div>

![Alt text](images/image-16.png)

### Example 4: AstroBin.csv output

<div style="page-break-after: always;"></div>

## **Troubleshooting & Installation Tips**

### **Fixing "ModuleNotFoundError" or "pip: command not found"**
If you encounter errors stating that a module (like `pandas` or `astropy`) is missing, or if your terminal does not recognize the `pip` command, follow these steps:

#### **On macOS and Linux**
MacOS often requires you to be explicit about which Python version you are using. If `pip install -r requirements.txt` fails, use the Python module wrapper:
* **The Recommended Command**: 
    `python3 -m pip install -r requirements.txt`
* **Why this works**: This ensures the libraries are installed specifically for the `python3` executable you are using to run the script.

#### **On Windows**
If `pip` is not recognized, ensure Python is added to your System PATH during installation, or use:
* `python -m pip install -r requirements.txt`

### **Common FITS/XISF Header Issues**
* **Missing Keywords**: If the script cannot find specific equipment or location data in your file headers, it will automatically fall back to the values defined in the `[defaults]` section of your `config.ini`.
* **Non-Standard Keywords**: If your capture software uses unique names for standard data, use the `[override]` section in `config.ini` to map them (e.g., mapping `CAMERA_MODEL` to `INSTRUME`).

### **Sky Quality and Site Naming**
* **No network calls**: since v2.1.0 the program contacts no external service. Bortle and SQM come from a matching `[sites]` entry, or from `[defaults]` when no site matches — see [Sky quality](#sky-quality) and [Site names](#site-names-formerly-reverse-geocoding) below.
* **Unexpected site name**: site naming is local and coordinate-clustered. If a session is attributed to the wrong site, check that its `[sites]` latitude and longitude match the frames' headers.

<div style="page-break-after: always;"></div>




# **References**

## **AstroBin's Acquisition CSV File Format**

This section details the required data fields for AstroBin's `acquisition.csv` dialogue.

### **AstroBin Long Exposure Acquisition Fields**

| **Field**        | Description | Validation |
|------------------|-------------|------------|
| **date**         | The date when the acquisition took place | YYYY-MM-DD format |
| **filter**       | Filter used | Numeric ID of a valid filter (found in the URL of the filter's page in the equipment database) |
| **number***      | Number of frames | Whole number |
| **duration***    | Duration of each frame in seconds | Number, Max decimals: 4, Min value: 0.0001, Max value: 999999.9999 |
| **iso**          | ISO setting on the camera | Whole number |
| **binning**      | Binning of pixels | One of [1, 2, 3, 4] |
| **gain**         | Gain setting on the camera | Number, Max decimals: 2 |
| **sensorCooling**| The temperature of the chip in Celsius degrees, e.g., -20 | Whole number, Min value: -274, Max value: 100 |
| **fNumber**      | If a camera lens was used, specify the f-number used for this acquisition session | Number, Max decimals: 2, Min value: 0 |
| **darks**        | The number of dark frames | Whole number, Min value: 0 |
| **flats**        | The number of flat frames | Whole number, Min value: 0 |
| **flatDarks**    | The number of flat dark frames | Whole number, Min value: 0 |
| **bias**         | The number of bias/offset frames | Whole number, Min value: 0 |
| **bortle**       | Bortle dark-sky scale | Whole number, Min value: 1, Max value: 9 |
| **meanSqm**      | Mean SQM mag/arcsec^2 as measured by a Sky Quality Meter | Number, Max decimals: 2, Min value: 0 |
| **meanFwhm**     | Mean Full Width at Half Maximum in arc seconds, a measure of seeing | Number, Max decimals: 2, Min value: 0 |
| **temperature**  | Ambient temperature in Celsius degrees | Number, Max decimals: 2, Min value: -88, Max value: 58 |

## **Astrobin Filter-Code mappings**

The [filters] section of the config.ini file defines the mapping from the filter name to AstroBin's filter code.

The contents of my [filters] section is given below. It shows the names my Astronomik 2 inch filters, as generated by N.I.N.A, and their corresponding AstroBin codes:

[filters]

| **Filter** | **Code** |
|------------|----------|
| Ha         | 4663     |
| SII        | 4844     |
| OIII       | 4752     |
| Red        | 4649     |
| Green      | 4643     |
| Blue       | 4637     |
| Lum        | 2906     |
| CLS        | 4061     |

This is the default filter table in the config.ini. You should this section so that it reflects the filters you use. The filter names should match the names the image capture software generates for your filters.

### **Finding AstroBin's Numeric ID for Filters**

The numeric ID of a filter can be found by examining the URL of the filter's page in the [AstroBin equipment database](https://app.astrobin.com/equipment/explorer/filter?page=1).   

For example, consider a [2-inch H-alpha CCD 6nm filter from Astronomik](https://app.astrobin.com/equipment/explorer/filter/4663/astronomik-h-alpha-ccd-6nm-2). By using [AstroBin's filter explorer](https://app.astrobin.com/equipment/explorer/filter?page=1) to navigate to this filter's page the URL is found to be :

https://app.astrobin.com/equipment/explorer/filter/4663/astronomik-h-alpha-ccd-6nm-2

From this URL, the AstroBin code for this Astronomik 2-inch H-alpha CCD 6nm filter is 4663.

## **Sky quality (Bortle and SQM)**

**As of v2.1.0 the utility makes no network calls.** Bortle and SQM come from
your `config.ini` — either from a matching entry in `[sites]`, or from
`[defaults]` when no site matches.

To fill those values in, look your observing site up by latitude and longitude
at the excellent <https://www.lightpollutionmap.info> and copy the Bortle and
SQM figures it reports into the relevant section. Earlier versions could fetch
this automatically with an API key held in `[secret]`; that path was removed,
along with the section.

## **Site names (formerly reverse geocoding)**

Also removed in v2.1.0. Earlier versions passed each site's coordinates to the
Nominatim / OpenStreetMap API (via `geopy`) to turn them into a postal
address, using an email address from `[secret]` as the courtesy identifier.
`geopy` is no longer a dependency at all.

Site naming is now entirely local, and works like this:

1. Coordinates from every frame's headers are clustered — readings within
   about 110 m of each other are treated as one physical site, which absorbs
   ordinary GPS drift across sessions.
2. Each cluster's centroid is looked up in `[sites]`. A match supplies that
   site's name, Bortle and SQM.
3. No match falls back to `[defaults]` (`SITE`, `SITELAT`, `SITELONG`,
   `BORTLE`, `SQM`).

Multi-site sessions therefore still report per-site correctly, provided each
site has an entry in `[sites]`. Add one by hand using the format shown in the
config walkthrough above — a full postal address makes the nicest summary
heading, but any label works.

## **FWHM Values**

The AstroBin Long Exposure Acquisition Fields has an entry for meanFwhm. This is not directly available from the header file. But N.I.N.A allows for the mean HFR value of an image to be embedded in the image file name. An example of my file naming convention, with HFR embedded, is given below:

'NGC 7822_Panel_1_Date_2023-09-02_Time_21-09-01_Filter_Ha_Exposure_600.00s_HFR_1.64px_FrameNo_0002.fits'

The code will look for the keyword HFR in the image file name. If found it will extract the HFR value and assign it to a variable HFR. As HFR is given in pixels, the script calculates the FWHM from the telescope information held in the FITS header. In particular XPIXSZ the x pixel size in microns and FOCALLEN the telescope focal length in mm.

IMSCALE = XPIXSZ / FOCALLEN * 206.265  
FWHM = 2 * hfr * imscale

The calculations above assume that all stars are circular making FWHM a scaled version of HFR. This is a reasonable approximation as the code averages HFR across all images taken on a particular date with a given filter and gain and then AstroBin further averages HFR across all entries in the uploaded CSV file. Where HFR is not available in the filename it is obtained from [defaults][HFR] in the config.ini file.

## **Data Sources** 
The script was developed to work with the following sources of image files

1. Night Time Imaging N' Astronomy ([N.I.N.A](https://nighttime-imaging.eu/))

2. Sequence Generator Pro ([SGPro](https://www.sequencegeneratorpro.com/sgpro/))

3. [PixInsight](https://pixinsight.com/) for Master calibration frames

FIT, FITS and FTS file headers are accessed in the code using [Astropy's FITS header library](https://docs.astropy.org/en/stable/io/fits/index.html).
To access XISF headers functions were developed based upon the [Pixinsight XISF header specification](https://pixinsight.com/doc/docs/XISF-1.0-spec/XISF-1.0-spec.html#xisf_header).


## **Contributing to AstroBinUpload.py Processing Script**

This script is intended for educational purposes in the field of astrophotography. It is part of an open-source project and contributions or suggestions for improvements are welcome.

To contribute to this project, follow these steps:

1. Fork this repository.
2. Create a branch: `git checkout -b <branch_name>`.
3. Make your changes and commit them: `git commit -m '<commit_message>'`.
4. Push to the original branch: `git push origin <project_name>/<location>`.
5. Create the pull request.

Alternatively, see the GitHub documentation on [creating a pull request](https://docs.github.com/en/github/collaborating-with-issues-and-pull-requests/creating-a-pull-request).

## **Contact**

If you want to contact me, you can reach me at sgreaves139@gmail.com.

## **License**

This project uses the following licence: [GNU General Public Licence v3.0](https://github.com/SteveGreaves/AstroBinUploader/blob/main/LICENSE).
