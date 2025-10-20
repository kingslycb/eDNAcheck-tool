
# eDNA-ObisTool - online tool to assess plausibility of species detections in molecular data

----------------------------------------------------------

# About
The **eDNA-ObisTool** is an online tool that allows to assess the plausibility of species detections in marine molecular biodiversity data.
The backend of the **eDNA-ObisTool** is Python-based. 

### What eDNA_ObisTool does?

**eDNA-ObisTool** compares observed species or genus occurrences—especially those detected through molecular methods such as environmental DNA (eDNA) or bulk-sample metabarcoding—with reference occurrence data. The reference occurrence data can be from global biodiversity repositories such as the Ocean Biodiversity Information System (OBIS), Global Biodiversity Information Facility (GBIF) or custom datasets. It supports spatial and temporal buffering, enabling flexible and ecologically-informed assessments of molecular biodiversity data plausibility. The tool performs the following:

- Loads molecular biodiversity records from user-provided input files.
- Cleans and standardizes taxonomic, spatial, and temporal information.
- Queries the OBIS API for matching occurrence records using optional buffers.
- Filters and aggregates global matches.
- Outputs comparison summaries and generates an interactive map.

### Why it matters?
Molecular biodiversity surveys (e.g., eDNA) often reveal unexpected or ambiguous species detections. eDNAcheck helps contextualize these results by:
- Verifying ecological plausibility of detected species or genera.
- Highlighting range extensions or novel records for further validation.
- Supporting biodiversity monitoring, conservation decisions, and quality control.
- Enabling transparent, reproducible comparison with global biodiversity repositories.
------------------------------------------------------------------

# How to use

The **eDNA-ObisTool** is an online tool and requires no installation. To upload the data for the quality check, following formats are accepted:
.csv, .xlsx, or .tab.  Required columns: Latitude (or Lat), Longitude (or Lon), Species or Genus. Optionally data can contain Year, Date, or EventDate
The output file (.xlsx) summarizes top occurrence matches per sample. Beside this, an interactive map with occurence matches will be displayed.

------------------------------------------------------------------

# Contact and citation information
**eDNA-ObisTool** has been developed as part of marine molecular biodiversity research at the Thünen Institute of Sea Fisheries.

For questions, feedback, or citation:

Email: kingsly-chuo.beng@thuenen.de
GitLab: 

Suggested citation:

"eDNA-ObisTool: Assessing the plausibility of species detections in molecular biodiversity data." Beng, K.C., Atamkeze R.A., Kasmi, Y., Sell, A. F., Akimova, A., Laakmann, S. 2025.


<details><summary>Click to expand</summary>
check whether my changes will be commited

## Docker 

### Build image

`docker build -t edna_obistool .`

### Run container

`docker run -p 8000:8000 --name edna_obistool edna_obistool`

eDNAcheck: Assessing the plausibility of species detections in molecular biodiversity data

1. What the tool does
eDNAcheck is a Python-based tool that compares observed species or genus occurrences—especially those detected through molecular methods such as environmental DNA (eDNA) or bulk-sample metabarcoding—with reference occurrence data. The reference occurrence data can be from global biodiversity repositories such as the Ocean Biodiversity Information System (OBIS), Global Biodiversity Information Facility (GBIF) or custom datasets. It supports spatial and temporal buffering, enabling flexible and ecologically-informed assessments of molecular biodiversity data plausibility. The tool performs the following:

- Loads molecular biodiversity records from user-provided input files.
- Cleans and standardizes taxonomic, spatial, and temporal information.
- Queries the OBIS API for matching occurrence records using optional buffers.
- Filters and aggregates global matches.
- Outputs comparison summaries and generates an interactive map.

2. Why it matters
Molecular biodiversity surveys (e.g., eDNA) often reveal unexpected or ambiguous species detections. eDNAcheck helps contextualize these results by:
- Verifying ecological plausibility of detected species or genera.
- Highlighting range extensions or novel records for further validation.
- Supporting biodiversity monitoring, conservation decisions, and quality control.
- Enabling transparent, reproducible comparison with global biodiversity repositories.

3. How to use it
Run the main function run_obis_comparison() from a Python script or notebook:

from ednacheck import run_obis_comparison

run_obis_comparison(
    file_path="input_data.xlsx",
    file_type="xlsx",                    
    buffer_time_years=5,
    time_buffer_ref="now",              
    buffer_size_km=50,                  
    search_by="species",               
    tops=10,
    output_name="ednacheck_output"
)

The function prints progress updates, saves result files, and generates an interactive HTML map.

4. Requirements and installation
Python Dependencies
Python ≥ 3.8
pandas
openpyxl
requests
folium
Built-in: math, datetime, traceback, os, time

Install via pip
pip install pandas openpyxl requests folium

5. Input/output expectations
Input File
Format: .csv, .xlsx, or .tab
Required columns:
Lat, Lon (or Latitude, Longitude)
Species or Genus
Optionally: Year, Date, or EventDate

Output Files
Excel file summarizing top occurrence matches per sample.
Interactive HTML map displaying:

Input points
Spatially buffered occurence matches

6. Examples
Example 1: Species-level search with buffer

run_obis_comparison(
    file_path="my_species_data.xlsx",
    file_type="xlsx",
    buffer_size_km=100,
    search_by="species",
    output_name="species_buffered_results"
)

Example 2: Genus-level query with temporal filtering

run_obis_comparison(
    file_path="genus_input.csv",
    buffer_size_km=50,
    buffer_time_years=10,
    time_buffer_ref="observation",
    search_by="genus",
    output_name="genus_with_temporal_filter"
)

### Contact and citation information
Developed as part of marine molecular biodiversity research at the Thünen Institute of Sea Fisheries.

For questions, feedback, or citation:
Email: kingsly-chuo.beng@thuenen.de
GitLab: 
Suggested citation:


</details>