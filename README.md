
# eDNAcheck - online tool to assess plausibility of species detections in molecular-based biodiversity data

----------------------------------------------------------

# About
The **eDNAcheck** is an online tool that allows to assess the plausibility of species detections in molecular biodiversity data.
The backend of the **eDNAcheck** is R and Python code provided here. 

### What eDNAcheck does?

**eDNAcheck** compares observed species or genus occurrences—especially those detected through molecular methods such as environmental DNA (eDNA) or bulk-sample metabarcoding—with reference occurrence data. The reference occurrence data can be from global biodiversity repositories such as the Ocean Biodiversity Information System (OBIS), Global Biodiversity Information Facility (GBIF) or custom datasets. It supports spatial and temporal buffering, enabling flexible and ecologically-informed assessments of molecular biodiversity data plausibility. The tool performs the following:

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

The **eDNAcheck** is an online tool and requires no installation. To upload the data for the quality check, following formats are accepted:
.csv, .xlsx, or .tab.  Required columns: Species and/or Genus name, Latitude (or Lat, decimal degrees), Longitude (or Lon, decimal degrees). Optional columns include: collection/observation year (Year, YYYY) or collection/observation date (Date, YYYY-MM-DD). 

The output file (.xlsx) summarizes top occurrence matches per sample. The number of matches  in the output file is user-defined and ranges from 1 to all available occurrence records. The default is 10. 
The output is accompanied by an interactive map with occurence matches.

eDNAcheck provides two validation options:

1. Global repository check – compares your data against global biodiversity databases (already implemented).
1. Custom database check – allows validation against a user-defined database (currently under development).


The required format for the custom database will be: Species name, Genus name, Lat, Lon, Year, Collection/observation date, Optional: any other metadata related to the species/genus occurences

------------------------------------------------------------------

# Contact and citation information
**eDNAcheck** has been developed as part of marine molecular biodiversity research at the Thünen Institute of Sea Fisheries. The development was partially supported by BMBF-Project "CREATE-Phase II"

For questions, feedback, or citation:

Email: kingsly-chuo.beng@thuenen.de

GitLab: 

Suggested citation: "eDNAcheck: Assessing the plausibility of species detections in molecular biodiversity data." Beng, K.C., Atamkeze R.A., Kasmi, Y., Sell, A. F., Akimova, A., Laakmann, S. 2025.



</details>
