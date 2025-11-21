from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Union, Tuple
import duckdb

@dataclass
class STRRegion:
    chrom: str
    start: int
    end: int
    str_unit: str
    str_id: str
    str_unit_length: int
    average_length: float
    additional_info: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        # Convert types if needed
        self.start = int(self.start)
        self.end = int(self.end)
        self.str_unit_length = int(self.str_unit_length)
        self.average_length = float(self.average_length)

    def __getitem__(self, key):
        if key in self.__dict__:
            return getattr(self, key)
        return self.additional_info.get(key)

    def __setitem__(self, key, value):
        if key in self.__dict__:
            setattr(self, key, value)
        else:
            self.additional_info[key] = value

    def to_dict(self) -> Dict[str, Any]:
        result = {k: v for k, v in self.__dict__.items() if k != 'additional_info' and v is not None}
        result.update(self.additional_info)
        return result

class STRCollection_duckdb:
    def __init__(self, database_path: str, id_column: str = 'id', mappings: Dict[str, str] = None, additional_columns: List[str] = None):
        self.database_path = database_path
        self.id_column = id_column
        self.mappings = mappings or {
            'chr': 'chrom',
            'start': 'start',
            'end': 'end',
            'motif': 'str_unit',
            'id': 'str_id',
            'len': 'str_unit_length',
            'avg_len': 'average_length'
        }
        self._length = None
        self._table_name = None
        self.additional_columns = additional_columns or []

    def _get_table_name(self):
        if self._table_name is None:
            conn = duckdb.connect(self.database_path, config={'access_mode': 'READ_ONLY'})
            tables = conn.sql("SHOW TABLES").fetchall()
            if len(tables) == 0:
                raise ValueError("No tables found in the database.")
            self._table_name = tables[0][0]  # Assume the first table is the one we want
            conn.close()
        return self._table_name

    def _escape_column_names(self, columns):
        return [f'"{col}"' if col.lower() in ['end', 'start', 'limit'] else col for col in columns]

    def get(self, str_id: Union[str, List[str]]) -> Union[Optional[STRRegion], List[Optional[STRRegion]]]:
        conn = duckdb.connect(self.database_path, config={'access_mode': 'READ_ONLY'})
        table_name = self._get_table_name()
        columns = self._escape_column_names(self.mappings.keys())
        columns_str = ', '.join(columns)
        
        if isinstance(str_id, str):
            query = f'SELECT {columns_str} FROM {table_name} WHERE {self.id_column} = ? LIMIT 1'
            result = conn.execute(query, [str_id]).fetchone()
            conn.close()
            print(f"Query: {query}")
            print(f"Result: {result}")
            return self._parse_str_region(result) if result else None
        else:
            placeholders = ', '.join(['?' for _ in str_id])
            query = f'SELECT {columns_str} FROM {table_name} WHERE {self.id_column} IN ({placeholders})'
            results = conn.execute(query, str_id).fetchall()
            conn.close()
            return [self._parse_str_region(result) for result in results]

    def __iter__(self):
        conn = duckdb.connect(self.database_path, config={'access_mode': 'READ_ONLY'})
        table_name = self._get_table_name()
        columns = self._escape_column_names(list(self.mappings.keys()) + self.additional_columns)
        columns_str = ', '.join(columns)
        query = f'SELECT {columns_str} FROM {table_name}'
        for result in conn.execute(query).fetchall():
            yield self._parse_str_region(result)
        conn.close()

    def __len__(self):
        if self._length is None:
            conn = duckdb.connect(self.database_path, config={'access_mode': 'READ_ONLY'})
            table_name = self._get_table_name()
            self._length = conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
            conn.close()
        return self._length
    
    def _parse_str_region(self, data: Tuple) -> STRRegion:
        print(f"Data: {data}")
        print(f"Mappings: {self.mappings}")
        print(f"Additional columns: {self.additional_columns}")
        kwargs = {}
        additional_info = {}
        for i, (db_column, str_attr) in enumerate(self.mappings.items()):
            kwargs[str_attr] = data[i]
        
        # Add additional columns to additional_info
        for i, column in enumerate(self.additional_columns, start=len(self.mappings)):
            if i < len(data):
                additional_info[column] = data[i]
            else:
                print(f"Warning: Column {column} not found in data")

        # Convert types
        type_conversions = {
            'start': int,
            'end': int,
            'str_unit_length': int,
            'average_length': float,
        }
        for attr, convert in type_conversions.items():
            if attr in kwargs:
                try:
                    kwargs[attr] = convert(kwargs[attr])
                except ValueError:
                    print(f"Warning: Unable to convert {attr} to {convert.__name__}. Using original value.")

        # Extract required positional arguments
        chrom = kwargs.pop('chrom', None)
        start = kwargs.pop('start', None)
        end = kwargs.pop('end', None)

        if chrom is None or start is None or end is None:
            raise ValueError(f"Missing required fields: chrom={chrom}, start={start}, end={end}")

        # Create STRRegion instance
        str_region = STRRegion(chrom, start, end, **kwargs, additional_info=additional_info)

        return str_region

def load_str_regions_from_duckdb(database_path: str, str_ids: Optional[Union[str, List[str]]] = None, limit: Optional[int] = None, mappings: Dict[str, str] = None, additional_columns: List[str] = None) -> Union[STRRegion, List[STRRegion]]:
    str_collection = STRCollection_duckdb(database_path, mappings=mappings, additional_columns=additional_columns)
    
    conn = duckdb.connect(database_path, config={'access_mode': 'READ_ONLY'})
    table_name = str_collection._get_table_name()
    columns = str_collection._escape_column_names(list(str_collection.mappings.keys()) + (additional_columns or []))
    columns_str = ', '.join(columns)
    
    if str_ids:
        if isinstance(str_ids, str):
            str_ids = [str_ids]
        placeholders = ', '.join(['?' for _ in str_ids])
        query = f'SELECT {columns_str} FROM {table_name} WHERE {str_collection.id_column} IN ({placeholders})'
        if limit:
            query += f' LIMIT {limit}'
        results = conn.execute(query, str_ids).fetchall()
    else:
        query = f'SELECT {columns_str} FROM {table_name}'
        if limit:
            query += f' LIMIT {limit}'
        results = conn.sql(query).fetchall()
    
    conn.close()
    return [str_collection._parse_str_region(result) for result in results]

class SNPInfo:
    def __init__(self, database_path: str, chr_column: str = 'chr', hsnp_column: str = 'hSNP_1', start_column: str = 'hSNP_start_1', id_column: str = 'id'):
        self.database_path = database_path
        self.chr_column = chr_column
        self.hsnp_column = hsnp_column
        self.start_column = start_column
        self.id_column = id_column
        self.snp_data: Dict[str, Dict[int, Tuple[str, int]]] = {}
        self.id_to_snp: Dict[str, Tuple[str, int, str, int]] = {}

    def load_snp_info(self, str_ids: Optional[Union[str, List[str]]] = None):
        conn = duckdb.connect(self.database_path, config={'access_mode': 'READ_ONLY'})
        query = f"""
        SELECT {self.chr_column}, {self.hsnp_column}, {self.start_column}, {self.id_column}
        FROM data
        WHERE {self.hsnp_column} IS NOT NULL AND {self.start_column} IS NOT NULL
        """
        
        if str_ids:
            if isinstance(str_ids, str):
                str_ids = [str_ids]
            placeholders = ', '.join(['?' for _ in str_ids])
            query += f" AND {self.id_column} IN ({placeholders})"
            print(f"query: {query}")
            results = conn.execute(query, str_ids).fetchall()
            print(f"Results: {results}")
        else:
            results = conn.execute(query).fetchall()
            print(f"Results: {results}")
        
        conn.close()

        for row in results:
            chrom, hsnp, start, str_id = row
            start = int(start)
            if chrom not in self.snp_data:
                self.snp_data[chrom] = {}
            self.snp_data[chrom][start] = (hsnp, start)
            self.id_to_snp[str_id] = (chrom, start, hsnp)

    def get_snp_info(self, chrom: str, position: int) -> Tuple[Optional[str], Optional[int]]:
        return self.snp_data.get(chrom, {}).get(position, (None, None))

    def get_snp_info_by_id(self, str_id: str) -> Tuple[Optional[str], Optional[int], Optional[str], Optional[int]]:
        return self.id_to_snp.get(str_id, (None, None, None, None))

    def get_all_snps_for_chrom(self, chrom: str) -> List[Tuple[int, str, int]]:
        return [(pos, hsnp, start) for pos, (hsnp, start) in self.snp_data.get(chrom, {}).items()]

    def __str__(self):
        return f"SNPInfo(database={self.database_path}, chromosomes={list(self.snp_data.keys())}, columns=({self.chr_column}, {self.hsnp_column}, {self.start_column}, {self.id_column}))"

