# Core Libraries
import pandas as pd
import requests
from geopy.distance import geodesic
import folium
import os
from datetime import datetime, timedelta
import math
import time
import traceback
import io
import shutil #  for potential future use, though not directly used 
from urllib.parse import parse_qs

# Shiny specific imports 
from shiny import App, ui, reactive, render
from shiny.types import FileInfo

# --- Helper: Define a temporary directory for outputs  ---
OUTPUT_DIR = "temp_app_outputs"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# --- Script 1 / InputProcessor Class ---
class InputProcessor:
    def __init__(self, file_path=None, file_content=None, file_type=None,
                 buffer_time_years=None, time_buffer_ref="now",
                 buffer_size_km=None, search_by="species", tops=10,
                 output_name_prefix="obis_comparison_output", # Changed from output_name
                 logger_func=print): # 
        self.file_path = file_path
        self.file_content = file_content # 
        self.file_type = file_type
        self.buffer_time_years = buffer_time_years
        self.time_buffer_ref = time_buffer_ref.lower()
        self.buffer_size_km = buffer_size_km
        self.search_by = search_by.lower()
        self.tops = tops
        self.output_name_prefix = output_name_prefix # 
        self.input_df = None
        self.logger = logger_func # 

        if self.time_buffer_ref not in ["now", "observation"]:
            raise ValueError("time_buffer_ref must be 'now' or 'observation'")

    def _determine_file_type_from_name(self): # 
        if self.file_type and self.file_type != "auto":
            self.logger(f"[InputProcessor] Using explicitly provided file type: {self.file_type}")
            return self.file_type.lower()
        if self.file_path:
            _, ext = os.path.splitext(self.file_path)
            ext = ext.lower()
            self.logger(f"[InputProcessor] Detected file extension: {ext} for path: {self.file_path}")
            if ext == '.xlsx': return 'xlsx'
            if ext == '.csv': return 'csv'
            if ext in ['.tab', '.tsv']: return 'tab'
        self.logger("[InputProcessor] Could not determine file type from name/extension or was 'auto'.")
        return None

    def load_data(self, progress_updater=None): # progress_updater 
        self.logger("[InputProcessor] Starting data loading process...")
        if progress_updater:
            progress_updater.set(detail="Determining file type...")

        effective_file_type = self._determine_file_type_from_name()
        if not effective_file_type and self.file_type and self.file_type != "auto":
             effective_file_type = self.file_type.lower()
        self.logger(f"[InputProcessor] Effective file type for loading: {effective_file_type}")

        try:
            # Handling file_content for pasted data 
            if self.file_content:
                self.logger("[InputProcessor] Loading data from pasted content.")
                if not effective_file_type or effective_file_type not in ['csv', 'tab']:
                    self.logger(f"[InputProcessor] Pasted data type '{effective_file_type}' is ambiguous or not set, defaulting to CSV.")
                    effective_file_type = 'csv' # Default for pasted if not clear
                if progress_updater: progress_updater.set(detail=f"Reading pasted {effective_file_type} data...")
                string_io = io.StringIO(self.file_content)
                if effective_file_type == 'csv': self.input_df = pd.read_csv(string_io)
                elif effective_file_type == 'tab': self.input_df = pd.read_csv(string_io, sep='\t')
                else: raise ValueError(f"Unsupported file_type '{effective_file_type}' for pasted data.")
                self.logger("[InputProcessor] Successfully read data from pasted content.")
            elif self.file_path:
                self.logger(f"[InputProcessor] Loading data from file: {self.file_path}")
                if not effective_file_type: # 
                     raise ValueError(f"Could not determine file type for {self.file_path} (was '{self.file_type}'). Please specify or use a standard extension.")
                if progress_updater: progress_updater.set(detail=f"Reading {effective_file_type} file: {os.path.basename(self.file_path)}...")
                if effective_file_type == 'xlsx':
                    self.input_df = pd.read_excel(self.file_path)
                elif effective_file_type == 'csv':
                    self.input_df = pd.read_csv(self.file_path)
                elif effective_file_type == 'tab':
                    self.input_df = pd.read_csv(self.file_path, sep='\t')
                else:
                    raise ValueError(f"Unsupported file_type: {effective_file_type}")
                self.logger(f"[InputProcessor] Successfully read data from file: {self.file_path}")
            else:
                raise ValueError("No input file path or pasted content provided.")

            if progress_updater: progress_updater.set(detail="Standardizing columns...")
            self.logger("[InputProcessor] Standardizing column names...")
            self.input_df.columns = self.input_df.columns.str.lower().str.strip()
            # Rename map 
            rename_map = {
                'species': 'Species', 'speices': 'Species',
                'genus': 'Genus',
                'lat': 'Lat', 'latitude': 'Lat',
                'lon': 'Lon', 'long': 'Lon', 'longitude': 'Lon',
                'year': 'Year', 'date': 'Year', 'date of observation': 'Year', 'eventdate':'Year'
            }
            self.input_df.rename(columns={k: v for k, v in rename_map.items() if k in self.input_df.columns}, inplace=True)
            self.logger(f"[InputProcessor] Columns after rename attempt: {self.input_df.columns.tolist()}")

            required_cols = ['Lat', 'Lon']
            if self.search_by == 'species':
                required_cols.append('Species')
            elif self.search_by == 'genus':
                required_cols.append('Genus')
            self.logger(f"[InputProcessor] Required columns: {required_cols} for search_by '{self.search_by}'.")

            missing_cols = [col for col in required_cols if col not in self.input_df.columns]
            if missing_cols:
                raise ValueError(f"Input file missing required columns: {', '.join(missing_cols)} for search_by='{self.search_by}'. Check column names (e.g., 'Species', 'Genus', 'Lat', 'Lon').")

            if progress_updater: progress_updater.set(detail="Processing date/year columns...")
            if 'Year' in self.input_df.columns:
                self.logger("[InputProcessor] Processing 'Year' column for 'Observation_Year'.")
                # Date parsing logic 
                parsed_dates = pd.to_datetime(self.input_df['Year'], errors='coerce')
                self.input_df['Observation_Year'] = parsed_dates.dt.year
                # Fallback for purely numeric years if datetime parsing fails for some
                if self.input_df['Observation_Year'].isnull().any():
                    numeric_years = pd.to_numeric(self.input_df['Year'], errors='coerce')
                    self.input_df['Observation_Year'] = self.input_df['Observation_Year'].fillna(numeric_years)
                self.input_df['Observation_Year'] = self.input_df['Observation_Year'].astype('Int64')
                self.logger("[InputProcessor] 'Observation_Year' processed.")
                if 'Observation_Year' in self.input_df.columns and self.input_df['Observation_Year'].notna().any():
                     self.logger(f"[InputProcessor] Observation years range: {self.input_df['Observation_Year'].min()} - {self.input_df['Observation_Year'].max()}")

            else:
                self.logger("[InputProcessor] Warning: 'Year' or similar date column not found. Time buffering based on 'observation' date will not be possible if 'Observation_Year' cannot be derived.")
                self.input_df['Observation_Year'] = pd.NA

            if progress_updater: progress_updater.set(detail="Converting Lat/Lon to numeric...")
            self.input_df['Lat'] = pd.to_numeric(self.input_df['Lat'], errors='coerce')
            self.input_df['Lon'] = pd.to_numeric(self.input_df['Lon'], errors='coerce')
            
            initial_rows = len(self.input_df)
            self.logger(f"[InputProcessor] Initial row count before NA drop: {initial_rows}")
            if progress_updater: progress_updater.set(detail="Dropping rows with missing critical data...")
            self.input_df.dropna(subset=['Lat', 'Lon'], inplace=True)
            if self.search_by == 'species' and 'Species' in self.input_df.columns:
                self.input_df.dropna(subset=['Species'], inplace=True)
            elif self.search_by == 'genus' and 'Genus' in self.input_df.columns:
                self.input_df.dropna(subset=['Genus'], inplace=True)
            
            dropped_rows = initial_rows - len(self.input_df)
            if dropped_rows > 0:
                self.logger(f"[InputProcessor] Warning: Dropped {dropped_rows} rows due to missing critical data (Lat/Lon or taxon name).")
            self.logger(f"[InputProcessor] Row count after NA drop: {len(self.input_df)}")

            if self.input_df.empty:
                raise ValueError("No valid data rows remaining after cleaning.")

            # Add Input_ID for Shiny app 
            self.input_df['Input_ID'] = [f"Sample_{i+1}" for i in range(len(self.input_df))]
            self.logger(f"[InputProcessor] Added 'Input_ID'. Loaded {len(self.input_df)} valid records.")
            if progress_updater: progress_updater.set(detail=f"{len(self.input_df)} valid records processed.")
            return self.input_df

        except FileNotFoundError:
            self.logger(f"[InputProcessor] Error: Input file not found at {self.file_path}")
            raise
        except Exception as e:
            self.logger(f"[InputProcessor] Error loading or processing input file: {e}\n{traceback.format_exc()}")
            raise

# --- Script 2 / ObisProcessor Class ---
class ObisProcessor:
    OBIS_API_URL = "https://api.obis.org/v3/occurrence"
    MAX_RESULTS_PER_PAGE = 10000

    def __init__(self, input_processor_instance, logger_func=print): # input_processor_config renamed to input_processor_instance
        self.config_processor = input_processor_instance # renamed for clarity
        self.logger = logger_func # f

    def _km_to_degree_lat(self, km):
        return km / 111.0

    def _km_to_degree_lon(self, km, latitude_deg):
        R = 6371
        lat_rad = math.radians(latitude_deg)
        deg_lon_km = (math.pi / 180) * R * math.cos(lat_rad)
        if deg_lon_km == 0:
            return km / 111.0
        return km / deg_lon_km

    def _get_bounding_box(self, lat, lon, buffer_km):
        if buffer_km is None or buffer_km <= 0:
            return None
        lat_offset_deg = self._km_to_degree_lat(buffer_km)
        lon_offset_deg = self._km_to_degree_lon(buffer_km, lat)
        min_lon, max_lon = lon - lon_offset_deg, lon + lon_offset_deg
        min_lat, max_lat = lat - lat_offset_deg, lat + lat_offset_deg
        min_lon = max(-180.0, min_lon)
        max_lon = min(180.0, max_lon)
        min_lat = max(-90.0, min_lat)
        max_lat = min(90.0, max_lat)
        if min_lon >= max_lon or min_lat >= max_lat:
            self.logger(f"[ObisProcessor] Warning: Invalid bounding box calculated for ({lat},{lon}) with buffer {buffer_km}km. Querying globally.")
            return None
        return f"POLYGON(({min_lon} {min_lat}, {max_lon} {min_lat}, {max_lon} {max_lat}, {min_lon} {max_lat}, {min_lon} {min_lat}))"

    def fetch_obis_occurrences(self, taxon_name, input_lat, input_lon, progress_updater=None, current_progress_value=0): # Shiny enhancements
        all_records = []
        params = {
            "scientificname": taxon_name,
            "size": self.MAX_RESULTS_PER_PAGE,
            "offset": 0
        }

        if self.config_processor.buffer_size_km and self.config_processor.buffer_size_km > 0: # check >0
            geometry_wkt = self._get_bounding_box(input_lat, input_lon, self.config_processor.buffer_size_km)
            if geometry_wkt:
                params["geometry"] = geometry_wkt
            else:
                 self.logger(f"[ObisProcessor] Warning: Bounding box for {taxon_name} at ({input_lat}, {input_lon}) with buffer {self.config_processor.buffer_size_km} km was not applied. Querying OBIS globally for this item and will filter locally.")
        
        self.logger(f"[ObisProcessor] Fetching OBIS data for: {taxon_name} (Lat: {input_lat}, Lon: {input_lon}). Initial params: {params.get('geometry', 'Global Search')}")
        if progress_updater: progress_updater.set(value=current_progress_value, detail=f"Fetching OBIS for {taxon_name}...")
        
        request_count = 0
        max_requests_per_taxon = 10 # Safety break, good to keep
        initial_offset_for_progress = params["offset"] # For progress in Shiny

        while request_count < max_requests_per_taxon:
            request_count += 1
            self.logger(f"[ObisProcessor] OBIS API request #{request_count} for {taxon_name}, offset {params['offset']}")
            try:
                response = requests.get(self.OBIS_API_URL, params=params, timeout=90) # timeout 
                response.raise_for_status()
                data = response.json()
                
                current_records = [] #
                if isinstance(data, list): current_records = data
                elif isinstance(data, dict) and "results" in data: current_records = data.get("results", [])
                
                all_records.extend(current_records)
                total_reported_by_obis = data.get("total") if isinstance(data, dict) else len(all_records)
                self.logger(f"[ObisProcessor]  Fetched {len(current_records)} records in this request for {taxon_name}. Total so far: {len(all_records)}. OBIS total: {total_reported_by_obis}")

                # Progress update logic 
                if progress_updater and total_reported_by_obis and total_reported_by_obis > 0:
                    fetched_so_far_this_taxon = params["offset"] + len(current_records) - initial_offset_for_progress
                    progress_updater.set(detail=f"Fetching {taxon_name}: {len(all_records)} of ~{total_reported_by_obis or 'many'} recs...")

                if not current_records or len(all_records) >= (total_reported_by_obis if total_reported_by_obis is not None else len(all_records)):
                    break
                params["offset"] += len(current_records)
                if params["offset"] >= (total_reported_by_obis if total_reported_by_obis is not None else params["offset"] + self.MAX_RESULTS_PER_PAGE * max_requests_per_taxon):
                    break
                time.sleep(0.3) # Be polite to the API

            except requests.exceptions.Timeout:
                self.logger(f"[ObisProcessor] API request timed out for {taxon_name}. Params: {params}")
                break
            except requests.exceptions.RequestException as e:
                self.logger(f"[ObisProcessor] API request failed for {taxon_name}: {e}. Params: {params}. Response: {response.text[:500] if 'response' in locals() and response else 'No response'}")
                break
            except ValueError as e: # Includes JSONDecodeError
                self.logger(f"[ObisProcessor] Failed to decode JSON response for {taxon_name}: {e}. Params: {params}. Response: {response.text[:500] if 'response' in locals() and response else 'No response'}")
                break
        
        if request_count >= max_requests_per_taxon and len(all_records) < (data.get("total", 0) if isinstance(data, dict) else len(all_records)):
            self.logger(f"[ObisProcessor] Warning: Reached max requests ({max_requests_per_taxon}) for {taxon_name}. Fetched {len(all_records)} out of {data.get('total', 'N/A')} records.")
        self.logger(f"[ObisProcessor] Finished fetching for {taxon_name}, total records: {len(all_records)}")
        return pd.DataFrame(all_records) if all_records else pd.DataFrame()

    def process_input_data(self, progress_updater=None, base_progress=0, total_progress_span=1): # Shiny enhancements
        self.logger("[ObisProcessor] Starting to process input data for OBIS queries.")
        if self.config_processor.input_df is None or self.config_processor.input_df.empty:
            self.logger("[ObisProcessor] No valid input data to process.")
            if progress_updater: progress_updater.set(value=base_progress + total_progress_span, message="No input data to process.")
            return pd.DataFrame()

        all_results_list = []
        current_year = datetime.now().year
        num_samples = len(self.config_processor.input_df)
        if num_samples == 0:
            if progress_updater: progress_updater.set(value=base_progress + total_progress_span, message="No samples to process.")
            return pd.DataFrame()
        self.logger(f"[ObisProcessor] Processing {num_samples} input samples.")

        for index, row in self.config_processor.input_df.iterrows():
            # Progress update logic 
            current_item_progress = (index + 1) / num_samples
            current_overall_progress = base_progress + (current_item_progress * total_progress_span)
            input_taxon_name = row.get('Species') if self.config_processor.search_by == 'species' else row.get('Genus')
            input_id = row.get('Input_ID', f"Sample_{index+1}") # Using Input_ID 
            
            self.logger(f"\n[ObisProcessor] Processing sample {index+1}/{num_samples}: ID '{input_id}', Taxon '{input_taxon_name}' at (Lat:{row['Lat']:.4f}, Lon:{row['Lon']:.4f})")
            if progress_updater:
                progress_updater.set(value=current_overall_progress, message=f"Sample {index+1}/{num_samples}", detail=f"Querying OBIS for {input_taxon_name} ({input_id})")

            obis_df = self.fetch_obis_occurrences(input_taxon_name, row['Lat'], row['Lon'], progress_updater=progress_updater, current_progress_value=current_overall_progress)

            if obis_df.empty:
                self.logger(f"[ObisProcessor]  No OBIS records found or API error for {input_taxon_name} ({input_id}).")
                continue

            if progress_updater: progress_updater.set(detail=f"Filtering {len(obis_df)} OBIS records for {input_id}...")
            self.logger(f"[ObisProcessor] Initial OBIS records for {input_id}: {len(obis_df)}")
            
            if 'decimalLatitude' not in obis_df.columns or 'decimalLongitude' not in obis_df.columns:
                self.logger(f"[ObisProcessor]  Skipping {input_taxon_name} ({input_id}) OBIS results due to missing 'decimalLatitude' or 'decimalLongitude'. Columns: {obis_df.columns}")
                continue
            
            obis_df['decimalLatitude'] = pd.to_numeric(obis_df['decimalLatitude'], errors='coerce')
            obis_df['decimalLongitude'] = pd.to_numeric(obis_df['decimalLongitude'], errors='coerce')
            obis_df.dropna(subset=['decimalLatitude', 'decimalLongitude'], inplace=True)
            if 'date_year' in obis_df.columns: obis_df['date_year'] = pd.to_numeric(obis_df['date_year'], errors='coerce').astype('Int64') # 
            self.logger(f"[ObisProcessor] Records after Lat/Lon cleanup for {input_id}: {len(obis_df)}")
            if obis_df.empty: self.logger(f"[ObisProcessor]  No OBIS records for {input_id} after lat/lon validation."); continue
            
            # Deduplication logic 
            if 'id' in obis_df.columns:
                original_obis_count_before_dedup = len(obis_df)
                if obis_df['id'].notna().any(): 
                    obis_df.drop_duplicates(subset=['id'], keep='first', inplace=True)
                    deduplicated_count = original_obis_count_before_dedup - len(obis_df)
                    if deduplicated_count > 0:
                        self.logger(f"[ObisProcessor]  Deduplication: Removed {deduplicated_count} duplicate OBIS records based on 'id' for {input_id}. Now {len(obis_df)} records.")
                else:
                    self.logger(f"[ObisProcessor]  Note: 'id' column present but all values are NA for {input_id}. No deduplication based on 'id' performed.")
            else:
                self.logger(f"[ObisProcessor]  Warning: 'id' column not found in OBIS results for {input_id}. Cannot deduplicate by 'id'.")
            if obis_df.empty: self.logger(f"[ObisProcessor]  No OBIS records for {input_id} after deduplication attempt."); continue

            # 1. Time Filtering 
            if self.config_processor.buffer_time_years is not None and self.config_processor.buffer_time_years > 0 and 'date_year' in obis_df.columns:
                original_count = len(obis_df)
                input_obs_year_val = pd.to_numeric(row.get('Observation_Year'), errors='coerce')
                self.logger(f"[ObisProcessor] Applying time buffer for {input_id}: {self.config_processor.buffer_time_years} years, ref: {self.config_processor.time_buffer_ref}, input_obs_year: {input_obs_year_val}")
                
                if pd.isna(input_obs_year_val) and self.config_processor.time_buffer_ref == "observation":
                    self.logger(f"[ObisProcessor]  Warning: Cannot apply 'observation' time buffer for {input_id} as input 'Observation_Year' is missing/invalid. Skipping time filter.")
                else:
                    if self.config_processor.time_buffer_ref == "now":
                        min_year = current_year - self.config_processor.buffer_time_years
                        max_year = current_year
                        obis_df = obis_df[(obis_df['date_year'].notna()) & (obis_df['date_year'] >= min_year) & (obis_df['date_year'] <= max_year)]
                        self.logger(f"[ObisProcessor]  Time filter (now) for {input_id}: {min_year}-{max_year}. Records after: {len(obis_df)}")
                    elif self.config_processor.time_buffer_ref == "observation" and pd.notna(input_obs_year_val):
                        min_year = int(input_obs_year_val - self.config_processor.buffer_time_years)
                        max_year = int(input_obs_year_val)
                        obis_df = obis_df[(obis_df['date_year'].notna()) & (obis_df['date_year'] >= min_year) & (obis_df['date_year'] <= max_year)]
                        self.logger(f"[ObisProcessor]  Time filter (obs yr {int(input_obs_year_val)}) for {input_id}: {min_year}-{max_year}. Records after: {len(obis_df)}")
                self.logger(f"[ObisProcessor]  Time filtering for {input_id}: {original_count} -> {len(obis_df)} records.")
            if obis_df.empty: self.logger(f"[ObisProcessor]  No OBIS records for {input_id} after time filtering."); continue

            # 2. Distance Calculation 
            self.logger(f"[ObisProcessor] Calculating distances for {input_id}...")
            obis_df['distance_km'] = obis_df.apply(
                lambda r: geodesic((row['Lat'], row['Lon']), (r['decimalLatitude'], r['decimalLongitude'])).km if pd.notna(r['decimalLatitude']) and pd.notna(r['decimalLongitude']) else float('inf'),
                axis=1
            )

            # 3. Distance Buffer Filtering 
            if self.config_processor.buffer_size_km is not None and self.config_processor.buffer_size_km > 0:
                original_count = len(obis_df)
                obis_df = obis_df[obis_df['distance_km'] <= self.config_processor.buffer_size_km]
                self.logger(f"[ObisProcessor]  Distance filtering (<= {self.config_processor.buffer_size_km} km) for {input_id}: {original_count} -> {len(obis_df)} records.")
            if obis_df.empty: self.logger(f"[ObisProcessor]  No OBIS records for {input_id} after distance filtering."); continue

            # 4. Sort by distance and select top N 
            obis_df = obis_df.sort_values(by='distance_km').head(self.config_processor.tops)
            self.logger(f"[ObisProcessor] Top {self.config_processor.tops} records for {input_id}: {len(obis_df)}")

            # 5. Format and Store Results 
            for rank, (_idx_obis, obis_row) in enumerate(obis_df.iterrows()):
                result = {
                    'Input_ID': input_id, # 
                    'Input_Species': row.get('Species', pd.NA),
                    'Input_Genus': row.get('Genus', pd.NA),
                    'Input_Lat': row['Lat'],
                    'Input_Lon': row['Lon'],
                    'Input_Observation_Year': row.get('Observation_Year', pd.NA),
                    'OBIS_ID': obis_row.get('id', pd.NA),
                    'OBIS_dataset_id': obis_row.get('dataset_id', pd.NA), # 
                    'OBIS_occurrenceID': obis_row.get('occurrenceID', pd.NA),
                    'OBIS_ScientificName': obis_row.get('scientificName', pd.NA),
                    'OBIS_Lat': obis_row.get('decimalLatitude', pd.NA),
                    'OBIS_Lon': obis_row.get('decimalLongitude', pd.NA),
                    'OBIS_Year': obis_row.get('date_year', pd.NA),
                    'OBIS_EventDate': obis_row.get('eventDate', pd.NA),
                    'Distance_km': round(obis_row['distance_km'], 2) if pd.notna(obis_row['distance_km']) else pd.NA,
                    'Rank': rank + 1,
                    'OBIS_BibliographicCitation': obis_row.get('bibliographicCitation', pd.NA),
                    'OBIS_CollectionCode': obis_row.get('collectionCode', pd.NA),
                    'OBIS_DatasetID': obis_row.get('datasetID', pd.NA), # Duplicate of dataset_id?
                    'OBIS_DatasetName': obis_row.get('datasetName', pd.NA),
                    'OBIS_InstitutionCode': obis_row.get('institutionCode', pd.NA),
                    'OBIS_Node_ID': obis_row.get('node_id', pd.NA), # 
                }
                all_results_list.append(result)
            self.logger(f"[ObisProcessor]  Added {len(obis_df)} records to final results for {input_id}.")

        self.logger(f"[ObisProcessor] Finished processing all samples. Total results collected: {len(all_results_list)}")
        if progress_updater: progress_updater.set(value=base_progress + total_progress_span, message="OBIS processing finished.")
        return pd.DataFrame(all_results_list)

    # save_results_to_excel is not needed here as Shiny handles downloads separately.

# --- Script 3 / MapGenerator Class ---
class MapGenerator:
    def __init__(self, input_df, results_df, selected_input_id="All Inputs", logger_func=print): # 
        # self.config_processor = input_processor_config # not directly needed if input_df is passed
        self.input_df = input_df
        self.results_df = results_df 
        self.selected_input_id = selected_input_id # 
        self.logger = logger_func # 

    def generate_map_html(self, output_map_path, progress_updater=None, detail_message="Generating map..."): # 
        self.logger(f"[MapGenerator] Generating map for '{self.selected_input_id}', output to: {output_map_path}")
        if progress_updater: progress_updater.set(detail=detail_message)

        # Centering logic  but using input_df passed to constructor
        if self.input_df is None or self.input_df.empty:
            self.logger("[MapGenerator] No input data for map.")
            # Create an empty map or message if no input
            m = folium.Map(location=[20, 0], zoom_start=2, tiles="CartoDB positron")
            folium.Html("<h3 style='text-align:center;font-family:sans-serif;'>No input data loaded to generate map.</h3>", script=True).add_to(m.get_root())
            m.save(output_map_path)
            if progress_updater: progress_updater.set(detail="No input data for map.")
            return output_map_path

        map_center_lat, map_center_lon = 0,0; zoom_start_level = 2
        inputs_to_plot = self.input_df
        results_to_plot = self.results_df if self.results_df is not None else pd.DataFrame()

        if self.selected_input_id != "All Inputs" and not self.input_df[self.input_df['Input_ID'] == self.selected_input_id].empty:
            center_row = self.input_df[self.input_df['Input_ID'] == self.selected_input_id].iloc[0]
            map_center_lat, map_center_lon = center_row['Lat'], center_row['Lon']
            zoom_start_level = 6 
            inputs_to_plot = self.input_df[self.input_df['Input_ID'] == self.selected_input_id]
            if not results_to_plot.empty and 'Input_ID' in results_to_plot.columns:
                 results_to_plot = results_to_plot[results_to_plot['Input_ID'] == self.selected_input_id]
        elif not self.input_df.empty: # If "All Inputs" or selected_id not found, use mean
            map_center_lat, map_center_lon = self.input_df['Lat'].mean(), self.input_df['Lon'].mean()
            zoom_start_level = 3
        
        self.logger(f"[MapGenerator] Map center: ({map_center_lat:.2f}, {map_center_lon:.2f}), Zoom: {zoom_start_level}")
        m = folium.Map(location=[map_center_lat, map_center_lon], zoom_start=zoom_start_level, tiles="CartoDB positron")

        # Marker adding logic 
        self.logger(f"[MapGenerator] Adding {len(inputs_to_plot)} input markers and {len(results_to_plot)} result markers.")
        if not inputs_to_plot.empty: # Ensure inputs_to_plot is not None
            for _, row in inputs_to_plot.iterrows():
                 if pd.notna(row['Lat']) and pd.notna(row['Lon']):
                    # Popup text, using Input_ID
                    popup_text = f"<b>Input: {row.get('Input_ID','N/A')}</b><br>"
                    popup_text += f"Taxon: {row.get('Species', row.get('Genus', 'N/A'))}<br>"
                    popup_text += f"Lat: {row['Lat']:.4f}, Lon: {row['Lon']:.4f}"
                    if pd.notna(row.get('Observation_Year')): # use .get for safety
                        popup_text += f"<br>Year: {int(row['Observation_Year'])}"
                    folium.Marker(
                        location=[row['Lat'], row['Lon']],
                        popup=folium.Popup(popup_text, max_width=300),
                        icon=folium.Icon(color='red', icon='location-dot', prefix='fa') # icon
                    ).add_to(m)

        if not results_to_plot.empty:
            for _, row in results_to_plot.iterrows():
                if pd.notna(row['OBIS_Lat']) and pd.notna(row['OBIS_Lon']):
                    # Popup text
                    popup_text = (
                        f"<b>OBIS Record (Rank: {row.get('Rank', 'N/A')})</b><br>"
                        f"Input ID: {row.get('Input_ID', 'N/A')}<br>" # Added Input_ID for context
                        f"ScientificName: {row.get('OBIS_ScientificName', 'N/A')}<br>"
                        f"Distance: {row.get('Distance_km', 'N/A')} km<br>"
                        f"Lat: {row['OBIS_Lat']:.4f}, Lon: {row['OBIS_Lon']:.4f}<br>"
                        f"Year: {row.get('OBIS_Year', 'N/A') if pd.notna(row.get('OBIS_Year')) else 'N/A'}<br>" # Handle NA year
                        f"Dataset: {row.get('OBIS_DatasetName', 'N/A')}"
                    )
                    folium.Marker(
                        location=[row['OBIS_Lat'], row['OBIS_Lon']],
                        popup=folium.Popup(popup_text, max_width=300),
                        icon=folium.Icon(color='blue', icon='circle', prefix='fa') # icon 
                    ).add_to(m)
        
        try:
            m.save(output_map_path)
            self.logger(f"[MapGenerator] Map successfully saved to {output_map_path}")
            if progress_updater: progress_updater.set(detail="Map saved.")
        except Exception as e:
            self.logger(f"[MapGenerator] Error saving map: {e}\n{traceback.format_exc()}")
            raise # Re-raise for Shiny error handling
        return output_map_path

# --- Shiny App UI  ---
app_ui = ui.page_fluid(
    ui.tags.style("""
        body { font-family: 'Arial', sans-serif; }
        .app-title { color: #00508F; text-align: center; margin-bottom: 20px; }
        .footer { text-align: center; margin-top: 30px; padding: 10px; border-top: 1px solid #ccc; font-size: 0.9em; color: #555; }
        .shiny-input-container { margin-bottom: 15px; }
        .btn-primary { background-color: #00508F; border-color: #00407F; }
        .btn-primary:hover { background-color: #003366; border-color: #00264C; }
        .card { margin-bottom: 20px; }
        .card-header { background-color: #f8f9fa; }
        textarea#pasted_data { min-height: 150px; }
    """),
    ui.h1("eDNA Obis Test", class_="app-title"),
    ui.layout_sidebar(
        ui.sidebar(
            ui.h4("Input & Parameters", style="color:#00508F;"),
            ui.input_radio_buttons("input_method", "Input Method:", {"upload": "Upload File", "paste": "Paste Data"}, selected="upload"),
            ui.output_ui("upload_ui_output"),
            ui.output_ui("paste_ui_output"),
            ui.hr(),
            ui.input_text("output_name_prefix_ui", "Output File Name Prefix:", value="eDNA_OBIS_analysis"), # Renamed for clarity
            ui.input_numeric("buffer_time_years", "Time Buffer (years, optional):", value=None, min=0, step=1), # Allow empty
            ui.input_select("time_buffer_ref", "Time Buffer Reference:", {"now": "Now", "observation": "Observation Date"}),
            ui.input_numeric("buffer_size_km", "Distance Buffer (km, optional):", value=None, min=0, step=1.0), # Allow empty
            ui.input_select("search_by", "Search By:", {"species": "Species", "genus": "Genus"}),
            ui.input_numeric("tops", "Top N Results per Sample:", value=10, min=1, step=1),
            ui.hr(),
            ui.input_action_button("run_analysis_btn", "Run Analysis", class_="btn-primary w-100"),
            ui.hr(),
            ui.p("Thünen Institute", style="font-size:0.8em; text-align:left; margin-top:10px; border-top:0px;")
        ),
        ui.navset_card_tab(
            ui.nav_panel("Map & Results",
                ui.row(ui.column(12, ui.output_ui("dynamic_map_ui_selector"))),
                ui.row(ui.column(12, ui.h4("Interactive Map", style="color:#00508F; margin-top:15px;"), ui.output_ui("map_output_embed_ui", height="550px"))),
                ui.row(ui.column(12, ui.h4("Results Table", style="color:#00508F; margin-top:15px;"), ui.output_data_frame("results_table_output"))),
                ui.row(
                    ui.column(6, ui.download_button("download_excel_btn", "Download Results (Excel)", class_="btn-success w-100 mt-3")),
                    ui.column(6, ui.download_button("download_map_btn", "Download Map (HTML)", class_="btn-info w-100 mt-3"))
                )
            ),
            ui.nav_panel("Input Data Preview", ui.h4("Loaded Input Data", style="color:#00508F; margin-top:15px;"), ui.output_data_frame("input_data_preview_table")),
            ui.nav_panel("Log Messages", ui.h4("Processing Log", style="color:#00508F; margin-top:15px;"), ui.output_text_verbatim("log_messages_output", placeholder=True))
        ),
        ui.div(ui.p("eDNA Obis Test App - Thünen Institute"), class_="footer")
    )
)

# --- Shiny App Server Logic ---
def server(input, output, session):
    analysis_state = reactive.Value({
        "input_df": pd.DataFrame(), "results_df": pd.DataFrame(),
        "excel_file_path_for_download": None, "log": []
    })

    def app_logger(message):
        current_s = analysis_state.get()
        new_log_list = current_s.get("log", []) + [f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - {message}"]
        if len(new_log_list) > 300: new_log_list = new_log_list[-300:]
        current_s["log"] = new_log_list
        analysis_state.set(current_s)
        print(f"APP_LOG: {message}")

    @output
    @render.ui
    def upload_ui_output():
        if input.input_method() == "upload":
            return ui.div(
                ui.input_file("input_file_upload", "Choose data file (.xlsx, .csv, .tab/.tsv)", accept=[".csv", ".tsv", ".tab", ".xlsx"], multiple=False),
                ui.input_select("file_type_upload", "Specify File Type:", {"auto": "Auto-detect", "xlsx": "Excel (.xlsx)", "csv": "CSV (.csv)", "tab": "Tab-separated (.tab, .tsv)"}, selected="auto")
            )
        return None

    @output
    @render.ui
    def paste_ui_output():
        if input.input_method() == "paste":
            return ui.div(
                ui.input_text_area("pasted_data", "Paste data here:", placeholder="Species,Lat,Lon,Year\nData,10.5,20.3,2022\n...", rows=8),
                ui.input_select("file_type_paste", "Specify Pasted Data Type:", {"csv": "CSV (comma-separated)", "tab": "TSV (tab-separated)"}, selected="csv")
            )
        return None

    @reactive.Calc
    def current_input_processor():
        file_infos = input.input_file_upload()
        pasted_content = input.pasted_data()
        input_method = input.input_method()
        file_path_val, file_content_val, file_type_val = None, None, None

        if input_method == "upload" and file_infos:
            file_path_val = file_infos[0]["datapath"]
            file_type_val = input.file_type_upload()
        elif input_method == "paste" and pasted_content and pasted_content.strip():
            file_content_val = pasted_content.strip()
            file_type_val = input.file_type_paste()
        
        # Handle potentially empty numeric inputs for buffer_time and buffer_size
        buffer_time_val = input.buffer_time_years()
        buffer_size_val = input.buffer_size_km()

        return InputProcessor(
            file_path=file_path_val, file_content=file_content_val, file_type=file_type_val,
            buffer_time_years=buffer_time_val if buffer_time_val is not None and not math.isnan(buffer_time_val) else None,
            time_buffer_ref=input.time_buffer_ref(),
            buffer_size_km=buffer_size_val if buffer_size_val is not None and not math.isnan(buffer_size_val) else None,
            search_by=input.search_by(), tops=input.tops(),
            output_name_prefix=input.output_name_prefix_ui() or "shiny_obis_analysis", # Use UI input for prefix
            logger_func=app_logger
        )

    @reactive.Effect
    @reactive.event(input.run_analysis_btn)
    def _perform_analysis():
        app_logger("Analysis run triggered by button press.")
        analysis_state.set({"input_df": pd.DataFrame(), "results_df": pd.DataFrame(), "excel_file_path_for_download": None, "log": []})

        processor = current_input_processor()
        if not processor.file_path and not processor.file_content:
            msg = "No input file or pasted data. Please provide input."
            app_logger(f"Error: {msg}")
            ui.notification_show(msg, type="error", duration=5)
            return

        with ui.Progress(min=0, max=100) as p:
            p.set(0, message="Initializing analysis...")
            app_logger("Progress: Initializing analysis...")
            try:
                p.set(value=5, message="Loading input data...")
                # InputProcessor.load_data now uses the combined logic
                input_df = processor.load_data(progress_updater=p)
                p.set(value=20, message="Input data loaded.")
                if input_df.empty:
                    msg = "Loaded input data is empty or invalid after cleaning."
                    app_logger(f"Error: {msg}")
                    ui.notification_show(msg, type="error", duration=7)
                    # Preserve logs on error, reset other data
                    current_s_err = analysis_state.get()
                    current_s_err.update({"input_df": pd.DataFrame(), "results_df": pd.DataFrame(), "excel_file_path_for_download": None})
                    analysis_state.set(current_s_err)
                    return

                app_logger(f"Progress: Input data loaded with {len(input_df)} rows.")
                
                # ObisProcessor uses combined logic
                obis_proc = ObisProcessor(processor, logger_func=app_logger)
                app_logger("Progress: Starting OBIS data processing...")
                results_df = obis_proc.process_input_data(progress_updater=p, base_progress=20, total_progress_span=60)
                p.set(value=80, message="OBIS data processing complete.")
                app_logger(f"Progress: OBIS processing complete. Found {len(results_df)} records.")

                p.set(value=85, message="Preparing output files...", detail="Saving Excel file...")
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                # Use output_name_prefix from the (updated) InputProcessor instance
                base_output_name = f"{processor.output_name_prefix}_{timestamp}"
                excel_file_name = f"{base_output_name}.xlsx"
                excel_file_path_for_download = os.path.join(OUTPUT_DIR, excel_file_name)

                if not results_df.empty:
                    results_df.to_excel(excel_file_path_for_download, index=False, engine='openpyxl')
                    app_logger(f"Results saved to Excel: {excel_file_path_for_download}")
                else:
                    app_logger("No OBIS results for Excel.")
                    excel_file_path_for_download = None # Ensure it's None if no file
                
                p.set(value=95, message="Preparing outputs...", detail="Finalizing...")
                app_logger("Progress: Finalizing outputs.")

                current_s = analysis_state.get()
                current_s.update({"input_df": input_df, "results_df": results_df, "excel_file_path_for_download": excel_file_path_for_download})
                analysis_state.set(current_s)
                p.set(100, message="Analysis complete!")
                app_logger("Progress: Analysis complete!")
                ui.notification_show("Analysis complete! View results and map.", type="default", duration=5)

            except Exception as e:
                error_msg = f"Analysis Error: {str(e)}"
                app_logger(error_msg + f"\n{traceback.format_exc()}") # Log full traceback for server
                ui.notification_show(error_msg, type="error", duration=10, close_button=True)
                current_s_exc = analysis_state.get() # Preserve logs on exception
                current_s_exc.update({"input_df": pd.DataFrame(), "results_df": pd.DataFrame(), "excel_file_path_for_download": None})
                analysis_state.set(current_s_exc)
                if 'p' in locals() and p: p.set(value=100, message="Analysis failed.", detail=str(e)[:100])


    @output
    @render.ui
    def dynamic_map_ui_selector():
        state = analysis_state.get()
        input_df = state.get("input_df")
        if input_df is not None and not input_df.empty:
            choices = {"All Inputs": "All Inputs"}
            if 'Input_ID' in input_df.columns: # Ensure Input_ID exists
                for sample_id in input_df["Input_ID"].unique():
                    sample_row = input_df[input_df["Input_ID"] == sample_id].iloc[0]
                    label_taxon = sample_row.get('Species', sample_row.get('Genus', 'Unknown Taxon'))
                    label = f"{sample_id}: {label_taxon}"
                    choices[sample_id] = label
            return ui.input_select("selected_map_input_id", "Focus Map/Table on Sample:", choices, selected="All Inputs")
        return ui.p("Run analysis to enable map options.")

    @output
    @render.ui
    def map_output_embed_ui():
        state = analysis_state.get()
        input_df = state.get("input_df")
        selected_id = input.selected_map_input_id() if input.selected_map_input_id() else "All Inputs"

        if input_df is None or input_df.empty: # Check if input_df exists and is not empty
             # Triggered if analysis not run yet or failed early
            if not analysis_state.get().get("log", []): # Very first load, no analysis attempted
                return ui.p("Map appears here after analysis.", style="text-align:center; padding:20px; border:1px dashed #ccc;")
            else: # Analysis attempted but input_df might be empty
                return ui.p("Map cannot be displayed: No valid input data loaded or analysis failed.", style="text-align:center; padding:20px; border:1px dashed #ccc;color:red;")


        timestamp = datetime.now().strftime("%Y%m%d%H%M%S%f")
        # Ensure selected_id is filesystem-friendly
        safe_selected_id = "".join(c if c.isalnum() else "_" for c in selected_id)
        map_filename_for_iframe = f"display_map_{safe_selected_id}_{timestamp}.html"
        
        app_logger(f"[Map UI] Requesting map for '{selected_id}'. Iframe will request: {map_filename_for_iframe}")
        return ui.tags.iframe(
            src=f"session/{session.id}/download/serve_dynamic_map?file={map_filename_for_iframe}", # Pass map_filename_for_iframe
            width="100%", height="500px", style="border: 1px solid #ddd;"
        )

    @session.download(filename=lambda: "current_map_display.html")
    async def serve_dynamic_map():
        try:
            query_string = session.http_conn.scope.get('query_string', b'').decode()
            parsed_query = parse_qs(query_string)
            target_short_filename = parsed_query.get('file', [None])[0]

            if not target_short_filename:
                app_logger("[Serve Map Error] 'file' query parameter missing in iframe request.")
                yield "Error: Map identifier missing."
                return

            app_logger(f"[Serve Map] Request received for map file: {target_short_filename}")
            map_full_path = os.path.join(OUTPUT_DIR, target_short_filename) # Use the received filename

            state = analysis_state.get()
            input_df = state.get("input_df")
            results_df = state.get("results_df")
            
            # Determine selected_id for map generation
            # This logic helps make serve_dynamic_map more robust if called directly or with complex IDs
            selected_id_for_map = "All Inputs" # Default
            parts = target_short_filename.replace("display_map_", "").replace(".html","").split('_')
            # Heuristic: last part is timestamp, parts before that (if any) form the ID
            if len(parts) > 1: # At least one ID part and timestamp
                potential_id_parts = parts[:-1] # Exclude timestamp
                parsed_selected_id = "_".join(potential_id_parts)
                # Check if this parsed ID exists in input_df's Input_ID or is "All_Inputs"
                if parsed_selected_id == "All_Inputs" or (input_df is not None and parsed_selected_id in input_df["Input_ID"].unique()):
                    selected_id_for_map = parsed_selected_id.replace("All_Inputs", "All Inputs") # Convert back
                else: # Fallback to current UI selection if parsing seems off or ID not found
                    selected_id_for_map = input.selected_map_input_id() if input.selected_map_input_id() else "All Inputs"
            else: # Fallback if filename format is unexpected
                 selected_id_for_map = input.selected_map_input_id() if input.selected_map_input_id() else "All Inputs"
            app_logger(f"[Serve Map] Effective selected_id for map generation: '{selected_id_for_map}' (derived from '{target_short_filename}')")


            if input_df is None or input_df.empty:
                app_logger("[Serve Map Error] No input data loaded to generate map for iframe.")
                # Create and yield an error HTML for the iframe
                error_html_content = "<p style='color:red; text-align:center; padding:20px;'>Error: No input data available to generate the map.</p>"
                with open(map_full_path, "w") as f_err: f_err.write(error_html_content)
                # Fallthrough to stream this error HTML
            else:
                 # MapGenerator uses combined logic
                map_gen = MapGenerator(input_df, results_df, selected_input_id=selected_id_for_map, logger_func=app_logger)
                map_gen.generate_map_html(map_full_path)
                app_logger(f"[Serve Map] Map generated and saved to: {map_full_path}")

            if os.path.exists(map_full_path):
                app_logger(f"[Serve Map] Streaming map file: {map_full_path}")
                with open(map_full_path, "rb") as f:
                    while True:
                        chunk = f.read(8192)
                        if not chunk: break
                        yield chunk
                # try: os.remove(map_full_path); app_logger(f"[Serve Map] Cleaned up: {map_full_path}")
                # except Exception as e_rm: app_logger(f"[Serve Map Cleanup Error] {e_rm}")
            else:
                app_logger(f"[Serve Map Error] Generated map file not found at: {map_full_path}")
                yield "Error: Map file could not be generated or found."

        except Exception as e_map_serve:
            app_logger(f"[Serve Map Exception] Error during map serving: {e_map_serve}\n{traceback.format_exc()}")
            yield f"Error serving map: {str(e_map_serve)}"


    @output
    @render.data_frame
    def results_table_output():
        state = analysis_state.get()
        results_df = state.get("results_df")
        selected_id = input.selected_map_input_id() if input.selected_map_input_id() else "All Inputs"
        if results_df is not None and not results_df.empty:
            if selected_id != "All Inputs" and 'Input_ID' in results_df.columns:
                return results_df[results_df['Input_ID'] == selected_id]
            return results_df
        return pd.DataFrame([{"Status":"No results yet or analysis failed."}])


    @output
    @render.data_frame
    def input_data_preview_table():
        state = analysis_state.get()
        input_df = state.get("input_df")
        if input_df is not None and not input_df.empty: return input_df
        return pd.DataFrame([{"Status":"No data loaded or analysis failed."}])


    @session.download(filename=lambda: (os.path.basename(analysis_state.get().get("excel_file_path_for_download")) if analysis_state.get().get("excel_file_path_for_download") and os.path.exists(analysis_state.get().get("excel_file_path_for_download")) else f"no_results_{datetime.now().strftime('%Y%m%d%H%M%S')}.txt") )
    async def download_excel_btn():
        state = analysis_state.get()
        excel_path = state.get("excel_file_path_for_download")
        if excel_path and os.path.exists(excel_path):
            app_logger(f"Preparing Excel download: {os.path.basename(excel_path)}")
            with open(excel_path, "rb") as f:
                while True:
                    chunk = f.read(8192)
                    if not chunk: break
                    yield chunk
        else:
            app_logger("Excel download: no file or path is None.")
            yield "No Excel results file available to download. Please run the analysis successfully."

    @session.download(filename=lambda: f"obis_map_{(''.join(c if c.isalnum() else '_' for c in input.selected_map_input_id()) if input.selected_map_input_id() else 'all')}_{datetime.now().strftime('%Y%m%d%H%M')}.html")
    async def download_map_btn():
        state = analysis_state.get()
        input_df, results_df = state.get("input_df"), state.get("results_df")
        selected_id = input.selected_map_input_id() if input.selected_map_input_id() else "All Inputs"
        
        if input_df is None or input_df.empty:
            app_logger("Map download: no input data.")
            yield "No map to download: no input data loaded or analysis failed."
            return

        processor_cfg = current_input_processor() # Get current config for output name prefix
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_selected_id = "".join(c if c.isalnum() else "_" for c in selected_id)
        dl_map_fname = f"{processor_cfg.output_name_prefix}_{safe_selected_id}_map_dl_{timestamp}.html"
        dl_map_path = os.path.join(OUTPUT_DIR, dl_map_fname)
        try:
            app_logger(f"Preparing map download for '{selected_id}' to '{dl_map_fname}'")
            # MapGenerator uses combined logic
            map_gen = MapGenerator(input_df, results_df, selected_input_id=selected_id, logger_func=app_logger)
            map_gen.generate_map_html(dl_map_path) 
            with open(dl_map_path, "rb") as f:
                while True:
                    chunk = f.read(8192)
                    if not chunk: break
                    yield chunk
            # try: os.remove(dl_map_path); app_logger(f"Cleaned up downloaded map: {dl_map_path}")
            # except Exception as e_rm_dl: app_logger(f"Error cleaning up downloaded map {dl_map_path}: {e_rm_dl}")

        except Exception as e:
            error_msg = f"Error generating map for download: {str(e)}"
            app_logger(error_msg + f"\n{traceback.format_exc()}")
            yield error_msg


    @output
    @render.text
    def log_messages_output():
        logs = analysis_state.get().get("log", [])
        return "\n".join(logs) if logs else "No log messages yet. Run analysis to see logs."

# Create the Shiny app instance
app = App(app_ui, server, debug=True)

# Note: To run this app, you would typically save it as a .py file (e.g., app.py) 
# and run `shiny run app.py --reload` in your terminal from the directory where the file is saved.
# Make sure you have all necessary libraries installed (pandas, requests, geopy, folium, openpyxl, shiny).
