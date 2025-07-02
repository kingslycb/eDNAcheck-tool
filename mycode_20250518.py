# Core Libraries
import pandas as pd
import requests
from geopy.distance import geodesic
import folium
import os
from datetime import datetime, timedelta
import math
import time # For potential rate limiting
import traceback # For detailed error logging

# --- Script 1: Input Loading and Parameter Definition ---
class InputProcessor:
    def __init__(self, file_path, file_type=None, buffer_time_years=None, time_buffer_ref="now",
                 buffer_size_km=None, search_by="species", tops=10, output_name="obis_comparison_output"):
        self.file_path = file_path
        self.file_type = file_type
        self.buffer_time_years = buffer_time_years
        self.time_buffer_ref = time_buffer_ref.lower() # 'now' or 'observation'
        self.buffer_size_km = buffer_size_km
        self.search_by = search_by.lower() # 'species' or 'genus'
        self.tops = tops
        self.output_name = output_name
        self.input_df = None

        if self.time_buffer_ref not in ["now", "observation"]:
            raise ValueError("time_buffer_ref must be 'now' or 'observation'")

    def _determine_file_type(self):
        if self.file_type:
            return self.file_type.lower()
        _, ext = os.path.splitext(self.file_path)
        ext = ext.lower()
        if ext == '.xlsx':
            return 'xlsx'
        elif ext == '.csv':
            return 'csv'
        elif ext in ['.tab', '.tsv']:
            return 'tab'
        else:
            raise ValueError(f"Unsupported file extension: {ext}. Please specify file_type ('xlsx', 'csv', 'tab').")

    def load_data(self):
        """Loads data from the input file into a pandas DataFrame."""
        file_type = self._determine_file_type()
        try:
            if file_type == 'xlsx':
                self.input_df = pd.read_excel(self.file_path)
            elif file_type == 'csv':
                self.input_df = pd.read_csv(self.file_path)
            elif file_type == 'tab':
                self.input_df = pd.read_csv(self.file_path, sep='\t')
            else:
                raise ValueError(f"Unsupported file_type: {file_type}")

            self.input_df.columns = self.input_df.columns.str.lower().str.strip()
            rename_map = {
                'species': 'Species', 'speices': 'Species',
                'genus': 'Genus',
                'lat': 'Lat', 'latitude': 'Lat',
                'lon': 'Lon', 'long': 'Lon', 'longitude': 'Lon',
                'year': 'Year', 'date': 'Year', 'date of observation': 'Year', 'eventdate':'Year'
            }
            self.input_df.rename(columns={k: v for k, v in rename_map.items() if k in self.input_df.columns}, inplace=True)

            required_cols = ['Lat', 'Lon']
            if self.search_by == 'species':
                required_cols.append('Species')
            elif self.search_by == 'genus':
                required_cols.append('Genus')

            missing_cols = [col for col in required_cols if col not in self.input_df.columns]
            if missing_cols:
                raise ValueError(f"Input file missing required columns: {', '.join(missing_cols)} for search_by='{self.search_by}'. Check column names (e.g., 'Species', 'Genus', 'Lat', 'Lon').")

            if 'Year' in self.input_df.columns:
                parsed_dates = pd.to_datetime(self.input_df['Year'], errors='coerce')
                self.input_df['Observation_Year'] = parsed_dates.dt.year
                numeric_years = pd.to_numeric(self.input_df['Year'], errors='coerce')
                self.input_df['Observation_Year'] = self.input_df['Observation_Year'].fillna(numeric_years).astype('Int64')
            else:
                print("Warning: 'Year' or similar date column not found in input. Time buffering based on 'observation' date will not be possible if 'Observation_Year' cannot be derived.")
                self.input_df['Observation_Year'] = pd.NA

            self.input_df['Lat'] = pd.to_numeric(self.input_df['Lat'], errors='coerce')
            self.input_df['Lon'] = pd.to_numeric(self.input_df['Lon'], errors='coerce')
            
            initial_rows = len(self.input_df)
            self.input_df.dropna(subset=['Lat', 'Lon'], inplace=True)
            if self.search_by == 'species':
                self.input_df.dropna(subset=['Species'], inplace=True)
            elif self.search_by == 'genus':
                self.input_df.dropna(subset=['Genus'], inplace=True)
            
            dropped_rows = initial_rows - len(self.input_df)
            if dropped_rows > 0:
                print(f"Warning: Dropped {dropped_rows} rows due to missing critical data (Lat/Lon or taxon name).")


            print(f"Loaded {len(self.input_df)} valid records from {self.file_path}")
            #print(f"Columns found and standardized: {self.input_df.columns.tolist()}")
            if 'Observation_Year' in self.input_df.columns and self.input_df['Observation_Year'].notna().any():
                print(f"Observation years range: {self.input_df['Observation_Year'].min()} - {self.input_df['Observation_Year'].max()}")

        except FileNotFoundError:
            raise FileNotFoundError(f"Error: Input file not found at {self.file_path}")
        except Exception as e:
            raise Exception(f"Error loading or processing input file: {e}\n{traceback.format_exc()}")
        return self.input_df

# --- Script 2: OBIS Interaction and Data Processing ---
class ObisProcessor:
    OBIS_API_URL = "https://api.obis.org/v3/occurrence"
    MAX_RESULTS_PER_PAGE = 10000

    def __init__(self, input_processor_config):
        self.config = input_processor_config

    def _km_to_degree_lat(self, km):
        return km / 111.0 # Approx 111 km per degree latitude

    def _km_to_degree_lon(self, km, latitude_deg):
        # Radius of Earth in km
        R = 6371
        # Convert latitude to radians
        lat_rad = math.radians(latitude_deg)
        # Length of a degree of longitude at this latitude
        deg_lon_km = (math.pi / 180) * R * math.cos(lat_rad)
        if deg_lon_km == 0: # Avoid division by zero at poles
            return km / 111.0 # Fallback, less accurate but safe
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
            print(f"Warning: Invalid bounding box calculated for ({lat},{lon}) with buffer {buffer_km}km. Offsets: lat_off={lat_offset_deg}, lon_off={lon_offset_deg}. Querying globally.")
            return None

        return f"POLYGON(({min_lon} {min_lat}, {max_lon} {min_lat}, {max_lon} {max_lat}, {min_lon} {max_lat}, {min_lon} {min_lat}))"

    def fetch_obis_occurrences(self, taxon_name, input_lat, input_lon):
        all_records = []
        params = {
            "scientificname": taxon_name,
            "size": self.MAX_RESULTS_PER_PAGE,
            "offset": 0
        }

        if self.config.buffer_size_km:
            geometry_wkt = self._get_bounding_box(input_lat, input_lon, self.config.buffer_size_km)
            if geometry_wkt:
                params["geometry"] = geometry_wkt
            else: # If bounding box calculation failed, query globally for this item
                 print(f"Warning: Bounding box for {taxon_name} at ({input_lat}, {input_lon}) with buffer {self.config.buffer_size_km} km was not applied. Querying OBIS globally for this item and will filter locally.")


        print(f"Fetching OBIS data for: {taxon_name} (Lat: {input_lat}, Lon: {input_lon}). Initial params: {params.get('geometry', 'Global Search')}")
        
        request_count = 0
        max_requests = 10 # Safety break for pagination to prevent infinite loops on unexpected API behavior
        
        while request_count < max_requests:
            request_count += 1
            try:
                response = requests.get(self.OBIS_API_URL, params=params, timeout=90)
                response.raise_for_status()
                data = response.json()
                
                current_records = []
                if isinstance(data, list): # Case where response is directly a list of records
                    current_records = data
                elif isinstance(data, dict) and "results" in data:
                    current_records = data.get("results", [])
                
                if not current_records and isinstance(data, dict) and data.get("total", 0) > 0 and len(all_records) == 0:
                    pass


                all_records.extend(current_records)
                #print(f"  Fetched {len(current_records)} records in this request. Total so far: {len(all_records)}. OBIS total for query: {data.get('total', 'N/A')}")

                total_reported_by_obis = data.get("total") if isinstance(data, dict) else len(all_records) # if data is list, no total
                
                if not current_records or len(all_records) >= (total_reported_by_obis if total_reported_by_obis is not None else len(all_records)):
                    break # All records fetched or no more records available

                params["offset"] += len(current_records)
                if params["offset"] >= (total_reported_by_obis if total_reported_by_obis is not None else params["offset"]): # Safety break if offset exceeds total
                    break

                #time.sleep(0.3) # Be polite to the API
                time.sleep(0.5) # Be polite to the API
            except requests.exceptions.Timeout:
                print(f"API request timed out for {taxon_name}. Params: {params}")
                break # Stop trying for this taxon on timeout
            except requests.exceptions.RequestException as e:
                print(f"API request failed for {taxon_name}: {e}. Params: {params}")
                print(f"Response content (first 500 chars): {response.text[:500] if response else 'No response'}")
                break # Stop trying for this taxon
            except ValueError as e: # Includes JSONDecodeError
                print(f"Failed to decode JSON response for {taxon_name}: {e}. Params: {params}")
                print(f"Response content (first 500 chars): {response.text[:500] if response else 'No response'}")
                break # Stop trying for this taxon
        
        if request_count >= max_requests and len(all_records) < (data.get("total", 0) if isinstance(data, dict) else len(all_records)):
            print(f"Warning: Reached max requests ({max_requests}) for {taxon_name}. Fetched {len(all_records)} out of {data.get('total', 'N/A')} records.")

        return pd.DataFrame(all_records) if all_records else pd.DataFrame()

    def process_input_data(self):
        if self.config.input_df is None or self.config.input_df.empty:
            print("No valid input data to process.")
            return pd.DataFrame()

        all_results_list = []
        current_year = datetime.now().year

        for index, row in self.config.input_df.iterrows():
            input_taxon_name = row.get('Species') if self.config.search_by == 'species' else row.get('Genus')
            print(f"\nProcessing input sample {index + 1}/{len(self.config.input_df)}: {input_taxon_name} at ({row['Lat']:.4f}, {row['Lon']:.4f})")

            obis_df = self.fetch_obis_occurrences(input_taxon_name, row['Lat'], row['Lon'])

            if obis_df.empty:
                print(f"  No OBIS records found or API error for {input_taxon_name}.")
                continue

            if 'decimalLatitude' not in obis_df.columns or 'decimalLongitude' not in obis_df.columns:
                print(f"  Skipping {input_taxon_name} OBIS results due to missing 'decimalLatitude' or 'decimalLongitude'. Columns: {obis_df.columns}")
                continue
            
            obis_df['decimalLatitude'] = pd.to_numeric(obis_df['decimalLatitude'], errors='coerce')
            obis_df['decimalLongitude'] = pd.to_numeric(obis_df['decimalLongitude'], errors='coerce')
            obis_df.dropna(subset=['decimalLatitude', 'decimalLongitude'], inplace=True)

            if obis_df.empty: # Check after initial lat/lon dropna
                print(f"  No OBIS records for {input_taxon_name} after lat/lon validation.")
                continue
            
            # +++ START OF MODIFIED CODE for deduplication +++
            if 'id' in obis_df.columns:
                original_obis_count_before_dedup = len(obis_df)
                # Ensure 'id' is not all NA before trying to drop duplicates
                if obis_df['id'].notna().any(): 
                    obis_df.drop_duplicates(subset=['id'], keep='first', inplace=True)
                    deduplicated_count = original_obis_count_before_dedup - len(obis_df)
                    if deduplicated_count > 0:
                        print(f"  Deduplication: Removed {deduplicated_count} duplicate OBIS records based on 'id'. Now {len(obis_df)} records.")
                else:
                    print(f"  Note: 'id' column present but all values are NA for {input_taxon_name}. No deduplication based on 'id' performed.")
            else:
                print(f"  Warning: 'id' column not found in OBIS results for {input_taxon_name}. Cannot deduplicate by 'id'.")

            if obis_df.empty: # Check if deduplication made it empty
                print(f"  No OBIS records for {input_taxon_name} after deduplication attempt.")
                continue
            # +++ END OF MODIFIED CODE for deduplication +++

            # 1. Time Filtering
            if self.config.buffer_time_years is not None and 'date_year' in obis_df.columns:
                obis_df['date_year'] = pd.to_numeric(obis_df['date_year'], errors='coerce').astype('Int64')
                original_count = len(obis_df)
                
                input_obs_year_val = pd.to_numeric(row.get('Observation_Year'), errors='coerce')
                if pd.isna(input_obs_year_val) and self.config.time_buffer_ref == "observation":
                    print(f"  Warning: Cannot apply 'observation' time buffer for {input_taxon_name} as input 'Observation_Year' is missing/invalid. Skipping time filter for this record.")
                else:
                    if self.config.time_buffer_ref == "now":
                        min_year = current_year - self.config.buffer_time_years
                        max_year = current_year
                        obis_df = obis_df[(obis_df['date_year'].notna()) & (obis_df['date_year'] >= min_year) & (obis_df['date_year'] <= max_year)]
                    elif self.config.time_buffer_ref == "observation" and pd.notna(input_obs_year_val):
                        min_year = input_obs_year_val - self.config.buffer_time_years
                        max_year = input_obs_year_val 
                        obis_df = obis_df[(obis_df['date_year'].notna()) & (obis_df['date_year'] >= min_year) & (obis_df['date_year'] <= max_year)]
                
                print(f"  Time filtering: {original_count} -> {len(obis_df)} records for {input_taxon_name}.")


            if obis_df.empty:
                print(f"  No OBIS records for {input_taxon_name} after time filtering.")
                continue

            # 2. Distance Calculation
            obis_df['distance_km'] = obis_df.apply(
                lambda r: geodesic((row['Lat'], row['Lon']), (r['decimalLatitude'], r['decimalLongitude'])).km,
                axis=1
            )

            # 3. Distance Buffer Filtering (local filter, supplements API's geometry filter or applies if API query was global)
            if self.config.buffer_size_km is not None:
                original_count = len(obis_df)
                obis_df = obis_df[obis_df['distance_km'] <= self.config.buffer_size_km]
                print(f"  Distance filtering (<= {self.config.buffer_size_km} km): {original_count} -> {len(obis_df)} records for {input_taxon_name}.")


            if obis_df.empty:
                print(f"  No OBIS records for {input_taxon_name} after distance filtering.")
                continue

            # 4. Sort by distance and select top N
            obis_df = obis_df.sort_values(by='distance_km').head(self.config.tops)

            # 5. Format and Store Results
            for rank, (_idx_obis, obis_row) in enumerate(obis_df.iterrows()):
                result = {
                    'Input_Species': row.get('Species', pd.NA),
                    'Input_Genus': row.get('Genus', pd.NA),
                    'Input_Lat': row['Lat'],
                    'Input_Lon': row['Lon'],
                    'Input_Observation_Year': row.get('Observation_Year', pd.NA),
                    'OBIS_ID': obis_row.get('id', pd.NA),
                    'OBIS_dataset_id': obis_row.get('dataset_id', pd.NA),
                    'OBIS_occurrenceID': obis_row.get('occurrenceID', pd.NA),
                    'OBIS_ScientificName': obis_row.get('scientificName', pd.NA),
                    'OBIS_Lat': obis_row.get('decimalLatitude', pd.NA),
                    'OBIS_Lon': obis_row.get('decimalLongitude', pd.NA),
                    'OBIS_Year': obis_row.get('date_year', pd.NA),
                    'OBIS_EventDate': obis_row.get('eventDate', pd.NA),
                    'Distance_km': round(obis_row['distance_km'], 2),
                    'Rank': rank + 1,
                    'OBIS_BibliographicCitation': obis_row.get('bibliographicCitation', pd.NA),
                    'OBIS_CollectionCode': obis_row.get('collectionCode', pd.NA),
                    'OBIS_DatasetID': obis_row.get('datasetID', pd.NA),
                    'OBIS_DatasetName': obis_row.get('datasetName', pd.NA),
                    'OBIS_InstitutionCode': obis_row.get('institutionCode', pd.NA),
                    'OBIS_Node_ID': obis_row.get('node_id', pd.NA),
                }
                all_results_list.append(result)
            print(f"  Added {len(obis_df)} records to final results for {input_taxon_name}.") # Changed message slightly for clarity

        return pd.DataFrame(all_results_list)

    def save_results_to_excel(self, results_df):
        if results_df.empty:
            print("No results to save to Excel.")
            return
        excel_output_path = f"{self.config.output_name}.xlsx"
        try:
            results_df.to_excel(excel_output_path, index=False, engine='openpyxl')
            print(f"Results saved to {excel_output_path}")
        except Exception as e:
            print(f"Error saving results to Excel: {e}\n{traceback.format_exc()}")

# --- Script 3: Generate Interactive Map ---
class MapGenerator:
    def __init__(self, input_processor_config, results_df):
        self.config = input_processor_config
        self.results_df = results_df 

    def generate_map(self):
        map_output_path = f"{self.config.output_name}_map.html"
        
        if self.config.input_df is None or self.config.input_df.empty:
            print("No input data loaded, cannot generate map with input samples.")
            if self.results_df.empty:
                print("Also no OBIS results. Map generation skipped.")
                return
            map_center_lat = self.results_df['OBIS_Lat'].iloc[0] if not self.results_df.empty and 'OBIS_Lat' in self.results_df.columns else 0
            map_center_lon = self.results_df['OBIS_Lon'].iloc[0] if not self.results_df.empty and 'OBIS_Lon' in self.results_df.columns else 0
        else: 
            map_center_lat = self.config.input_df['Lat'].iloc[0]
            map_center_lon = self.config.input_df['Lon'].iloc[0]

        m = folium.Map(location=[map_center_lat, map_center_lon], zoom_start=4)

        if self.config.input_df is not None and not self.config.input_df.empty:
            for _, row in self.config.input_df.iterrows():
                 if pd.notna(row['Lat']) and pd.notna(row['Lon']):
                    popup_text = f"<b>Input Sample</b><br>"
                    popup_text += f"Taxon: {row.get('Species', row.get('Genus', 'N/A'))}<br>"
                    popup_text += f"Lat: {row['Lat']:.4f}, Lon: {row['Lon']:.4f}"
                    if pd.notna(row.get('Observation_Year')):
                        popup_text += f"<br>Year: {int(row['Observation_Year'])}"
                    folium.Marker(
                        location=[row['Lat'], row['Lon']],
                        popup=folium.Popup(popup_text, max_width=300),
                        icon=folium.Icon(color='red', icon='info-sign')
                    ).add_to(m)
            print(f"Added {len(self.config.input_df)} input sample markers to map.")

        if not self.results_df.empty:
            for _, row in self.results_df.iterrows():
                if pd.notna(row['OBIS_Lat']) and pd.notna(row['OBIS_Lon']):
                    popup_text = (
                        f"<b>OBIS Record (Rank: {row.get('Rank', 'N/A')})</b><br>"
                        f"ScientificName: {row.get('OBIS_ScientificName', 'N/A')}<br>"
                        f"Distance: {row.get('Distance_km', 'N/A')} km<br>"
                        f"Lat: {row['OBIS_Lat']:.4f}, Lon: {row['OBIS_Lon']:.4f}<br>"
                        f"Year: {row.get('OBIS_Year', 'N/A')}<br>"
                        f"Dataset: {row.get('OBIS_DatasetName', 'N/A')}"
                    )
                    folium.Marker(
                        location=[row['OBIS_Lat'], row['OBIS_Lon']],
                        popup=folium.Popup(popup_text, max_width=300),
                        icon=folium.Icon(color='blue', icon='cloud')
                    ).add_to(m)
            print(f"Added {len(self.results_df)} OBIS occurrence markers to map.")
        else:
            print("No OBIS results to add to map.")
        
        try:
            m.save(map_output_path)
            print(f"Interactive map saved to {map_output_path}")
        except Exception as e:
            print(f"Error saving map: {e}\n{traceback.format_exc()}")


# --- Main Processing Command Function ---
def run_obis_comparison(file_path, file_type=None, buffer_time_years=None,
                        time_buffer_ref="now", buffer_size_km=None,
                        search_by="species", tops=10,
                        output_name="obis_comparison_output"):
    """
    Main processing function to compare input data with OBIS.

    Args:
        file_path (str): Path to the input data file.
        file_type (str, optional): Type of the input file ('xlsx', 'csv', 'tab').
                                   Defaults to None (auto-detected).
        buffer_time_years (int, optional): Time buffer in years. Defaults to None.
        time_buffer_ref (str, optional): Reference for time buffer ('now' or 'observation').
                                         Defaults to "now".
        buffer_size_km (float, optional): Spatial buffer in kilometers. Defaults to None.
        search_by (str, optional): Taxon level to search by ('species' or 'genus').
                                   Defaults to "species".
        tops (int, optional): Number of top results to fetch per input sample. Defaults to 10.
        output_name (str, optional): Base name for output files.
                                     Defaults to "obis_comparison_output".
    """
    start_time = time.time()
    try:
        print(f"--- Starting OBIS Comparison for: {file_path} ---")
        print(f"Parameters:\n  File Type: {file_type if file_type else 'Auto-detect'}\n  Time Buffer: {buffer_time_years} years ({time_buffer_ref} ref)\n  Distance Buffer: {buffer_size_km} km\n  Search By: {search_by}\n  Tops: {tops}\n  Output Name: {output_name}")

        # Script 1: Input Processing
        input_processor = InputProcessor(
            file_path=file_path,
            file_type=file_type,
            buffer_time_years=buffer_time_years,
            time_buffer_ref=time_buffer_ref,
            buffer_size_km=buffer_size_km,
            search_by=search_by,
            tops=tops,
            output_name=output_name
        )
        input_processor.load_data()

        if input_processor.input_df is None or input_processor.input_df.empty:
            print(f"Critical: Failed to load or no valid data in input file: {file_path}. Aborting process.")
            return

        # Script 2: OBIS Interaction and Data Processing
        obis_processor = ObisProcessor(input_processor_config=input_processor)
        results_df = obis_processor.process_input_data()

        # Script 2f: Save Results to Excel
        if not results_df.empty:
            obis_processor.save_results_to_excel(results_df)
        else:
            print("No results were generated from OBIS matching the criteria. Excel file will not be created (or will be empty if saving empty DFs was intended).")

        # Script 3: Generate Interactive Map
        map_gen = MapGenerator(input_processor_config=input_processor, results_df=results_df if not results_df.empty else pd.DataFrame())
        map_gen.generate_map()

        print(f"--- OBIS Comparison finished for: {file_path} ---")
        print(f"Output files generated with base name: {output_name}")

    except FileNotFoundError:
        print(f"CRITICAL ERROR: Input file not found at {file_path}")
    except ValueError as ve:
        print(f"CRITICAL ERROR: Invalid parameter or data format - {ve}")
    except requests.exceptions.RequestException as re:
        print(f"CRITICAL ERROR: OBIS API request failed - {re}")
    except Exception as e:
        print(f"CRITICAL UNEXPECTED ERROR during the OBIS comparison for {file_path}: {e}")
        print(traceback.format_exc())
    finally:
        end_time = time.time()
        print(f"Total processing time: {end_time - start_time:.2f} seconds.")

