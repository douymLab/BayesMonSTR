import os
import csv
import requests

class MetaInfo:
    def __init__(self, csv_file, sample_column: str, path_column: str, 
                 alias_column: str = None, id_column: str = None, pro_type_column: str = None):
        self.csv_file = csv_file
        self.sample_column = sample_column
        self.path_column = path_column
        self.alias_column = alias_column
        self.id_column = id_column
        self.pro_type_column = pro_type_column
        self.data = []
        self.bam_names = {}
        self.aliases = {}
        self.pro_types = {}
        self.id = {}
        self.bam_file_dir = None
        self.file_existence = {}
        self.ref_path = None
        self.custom_bam_dirs = {}
        self.file_sizes = {}

        self._load_meta_info()
        self._check_files_existence()

    def _load_meta_info(self):
        """Load information from the CSV file and populate the data list and dictionaries."""
        with open(self.csv_file, 'r') as csvfile:
            csvreader = csv.DictReader(csvfile)
            for row in csvreader:
                self.data.append(row)
                sample = row[self.sample_column]
                path = row[self.path_column]
                bam_file_name = os.path.basename(path)
                self.bam_names[sample] = bam_file_name

                if self.alias_column and self.alias_column in row:
                    alias = row[self.alias_column]
                    self.aliases[alias] = sample

                if self.id_column and self.id_column in row:
                    sample_id = row[self.id_column]
                    self.id[sample_id] = sample

                if self.pro_type_column and self.pro_type_column in row:
                    pro_type = row[self.pro_type_column]
                    self.pro_types[sample] = pro_type

                if self.bam_file_dir is None:
                    self.bam_file_dir = os.path.dirname(path)

    def set_custom_bam_dir(self, sample_or_alias, custom_dir):
        """Set a custom directory for a specific BAM file."""
        sample = self.aliases.get(sample_or_alias, sample_or_alias)
        if sample in self.bam_names:
            self.custom_bam_dirs[sample] = custom_dir
            self._check_file_existence(sample)
        else:
            raise ValueError(f"Sample or alias '{sample_or_alias}' not found.")
        
    def get_file_path(self, sample_or_alias):
        """Get the full file path for a given sample or alias."""
        sample = self.aliases.get(sample_or_alias, sample_or_alias)
        if sample not in self.bam_names:
            raise ValueError(f"Sample or alias '{sample_or_alias}' not found.")
        
        bam_file_name = self.bam_names[sample]
        if sample in self.custom_bam_dirs:
            return os.path.join(self.custom_bam_dirs[sample], bam_file_name)
        else:
            return os.path.join(self.bam_file_dir, bam_file_name)
        
    def get_custom_bam_dir(self, sample_or_alias):
        """Get the custom BAM directory for a given sample or alias, if set."""
        sample = self.aliases.get(sample_or_alias, sample_or_alias)
        return self.custom_bam_dirs.get(sample)

    def _check_files_existence(self):
        """Update the file_existence dictionary with the existence status of the BAM files."""
        for sample in self.bam_names:
            self._check_file_existence(sample)

    def _check_file_existence(self, sample):
        """Check existence of a single BAM file, considering custom directory if set."""
        file_path = self.get_file_path(sample)
        exists = self._check_file(file_path)
        self.file_existence[sample] = exists
        if exists:
            self.file_sizes[sample] = self._get_file_size(file_path)
        else:
            self.file_sizes[sample] = 0

    def _check_file(self, file_path):
        """Check if the file exists, whether it's a local file or an HTTP URL."""
        if file_path.startswith('http://') or file_path.startswith('https://'):
            try:
                response = requests.head(file_path, timeout=5)
                return response.status_code == 200
            except requests.RequestException as e:
                print(f"Error checking file at {file_path}: {e}")
                return False
        else:
            return os.path.exists(file_path)

    def file_exists(self, sample_or_alias):
        """Check if the BAM file exists for the given sample or alias."""
        sample = self.aliases.get(sample_or_alias, sample_or_alias)
        return self.file_existence.get(sample, False)

    def get_existing_files_info(self):
        """Return a dictionary with the names of the BAM files that exist."""
        return {sample: bam_name for sample, bam_name in self.bam_names.items() if self.file_existence[sample]}

    def update_bam_file_dir(self, new_dir):
        """Update the default BAM file directory and recheck the existence of the files."""
        self.bam_file_dir = new_dir
        self._check_files_existence()

    def update_ref_file_path(self,new_path):
        """Update the ref path"""
        self.ref_path = new_path

    def select_bam_info(self, identifiers):
        """
        Select the information for the specified BAM files, sample names, or aliases.
        :param identifiers: A list of integers, BAM file names, sample names, or aliases.
        :return: A list of dictionaries with the selected rows.
        """
        if not isinstance(identifiers, list):
            identifiers = [identifiers]

        selected_info = []
        for identifier in identifiers:
            if isinstance(identifier, int):
                if 0 <= identifier < len(self.data):
                    selected_info.append(self.data[identifier])
            else:
                sample = self.aliases.get(identifier, identifier)
                for row in self.data:
                    if sample in row['sample'] or identifier in row['path']:
                        selected_info.append(row)

        return selected_info

    def __repr__(self):
        return (f"MetaInfo(csv_file='{self.csv_file}', bam_file_dir='{self.bam_file_dir}', "
                f"total_samples={len(self.data)}, existing_files={sum(self.file_existence.values())}, "
                f"total unique pro_types={len(set(self.pro_types.values()))}, "
                f"alias_column='{self.alias_column}', id_column='{self.id_column}', "
                f"pro_type_column='{self.pro_type_column}')")
    
    def _get_file_sizes(self):
        """Get and store the sizes of all BAM files."""
        for sample in self.bam_names:
            file_path = self.get_file_path(sample)
        self.file_sizes[sample] = self._get_file_size(file_path)

    def _get_file_size(self, file_path):
        """Get the size of a file, whether it's a local file or an HTTP URL."""
        if file_path.startswith('http://') or file_path.startswith('https://'):
            try:
                response = requests.head(file_path, timeout=5)
                return int(response.headers.get('Content-Length', 0))
            except requests.RequestException as e:
                print(f"Error getting file size for {file_path}: {e}")
                return 0
        else:
            try:
                return os.path.getsize(file_path)
            except OSError as e:
                print(f"Error getting file size for {file_path}: {e}")
                return 0

    def get_file_size(self, sample_or_alias):
        """Get the size of the BAM file for the given sample or alias."""
        sample = self.aliases.get(sample_or_alias, sample_or_alias)
        return self.file_sizes.get(sample, 0)

    def update_file_size(self, sample_or_alias):
        """Update the file size for a specific sample or alias if the file exists."""
        sample = self.aliases.get(sample_or_alias, sample_or_alias)
        if sample in self.bam_names:
            if self.file_existence.get(sample, False):
                file_path = self.get_file_path(sample)
                self.file_sizes[sample] = self._get_file_size(file_path)
            else:
                self.file_sizes[sample] = 0
        else:
            raise ValueError(f"Sample or alias '{sample_or_alias}' not found.")

    def __str__(self):
        info = [f"MetaInfo Object:\nCSV File: {self.csv_file}\nDefault BAM File Directory: {self.bam_file_dir}\n"]
        info.append(f"Total Samples: {len(self.data)}\nExisting Files: {sum(self.file_existence.values())}\n")
        info.append(f"Alias Column: {self.alias_column}\nID Column: {self.id_column}\n")
        info.append(f"Protocol Type Column: {self.pro_type_column}\nSample Info:")
        for sample, exists in self.file_existence.items():
            status = 'Exists' if exists else 'Does not exist'
            alias = next((alias for alias, s in self.aliases.items() if s == sample), None)
            id = next((id for id, s in self.id.items() if s == sample), None)
            pro_type = self.pro_types.get(sample, None)
            file_path = self.get_file_path(sample)
            alias_str = f" (Alias: {alias})" if alias else ""
            ids_str = f" (ID: {id})" if id else ""
            pro_type_str = f" (Pro Type: {pro_type})" if pro_type else ""
            info.append(f"  {sample}{alias_str}{ids_str}: {status}{pro_type_str}")
            info.append(f"    File Path: {file_path}")
        return "\n".join(info)
        
    def save_updated_data(self, output_file):
        """Save the updated data to a new CSV file."""
        if not self.data:
            raise ValueError("No data to save.")

        fieldnames = self.data[0].keys()

        with open(output_file, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for row in self.data:
                writer.writerow(row)

        print(f"Updated data saved to {output_file}")