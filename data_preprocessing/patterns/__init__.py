from .base import BasePattern
from .date import DatePattern
from .time import TimePattern
from .datetime import DatetimePattern
from .timestamp import TimestampPattern
from .year import YearPattern
from .month import MonthPattern
from .day import DayPattern
from .week import WeekPattern
from .quarter import QuarterPattern
from .fiscal_year import FiscalYearPattern
from .duration import DurationPattern
from .currency import CurrencyPattern
from .salary import SalaryPattern
from .number_systems import NumberSystemsPattern
from .percentage import PercentagePattern
from .ratio import RatioPattern
from .integer import IntegerPattern
from .float import FloatPattern
# Measurement patterns
from .distance import DistancePattern
from .weight import WeightPattern
from .volume import VolumePattern
from .area import AreaPattern
from .speed import SpeedPattern
from .temperature import TemperaturePattern
from .pressure import PressurePattern
from .energy import EnergyPattern
from .power import PowerPattern
from .capacity import CapacityPattern
from .density import DensityPattern
from .angle import AnglePattern
# Geographical patterns
from .latitude import LatitudePattern
from .longitude import LongitudePattern
from .geo_coordinate import GeoCoordinatePattern
from .timezone import TimezonePattern
# Boolean/Flag patterns
from .boolean import BooleanPattern
# Categorical patterns
from .gender import GenderPattern
# Contact/Identity patterns
from .email import EmailPattern
# Internet/Tech/System patterns
from .url import URLPattern
from .network_addresses import NetworkAddressesPattern
from .file_path import FilePathPattern
from .files import FilesPattern
from .version import VersionPattern
from .varchar import VarcharPattern
from .text import TextPattern

__all__ = [
    'BasePattern',
    'DatePattern',
    'TimePattern',
    'DatetimePattern',
    'TimestampPattern',
    'YearPattern',
    'MonthPattern',
    'DayPattern',
    'WeekPattern',
    'QuarterPattern',
    'FiscalYearPattern',
    'DurationPattern',
    'AgePattern',
    'CurrencyPattern',
    'SalaryPattern',
    'NumberSystemsPattern',
    'PercentagePattern',
    'RatioPattern',
    'IntegerPattern',
    'FloatPattern',
    # Measurements
    'DistancePattern',
    'WeightPattern',
    'VolumePattern',
    'AreaPattern',
    'SpeedPattern',
    'TemperaturePattern',
    'PressurePattern',
    'EnergyPattern',
    'PowerPattern',
    'CapacityPattern',
    'DensityPattern',
    'AnglePattern',
    # Geographical patterns
    'LatitudePattern',
    'LongitudePattern',
    'GeoCoordinatePattern',
    'TimezonePattern',
    # Boolean/Flag patterns
    'BooleanPattern',
    'VarcharPattern',
    'TextPattern',
]


# Pattern registry for easy access
PATTERNS = {
    'date': DatePattern(),
    'time': TimePattern(),
    'datetime': DatetimePattern(),
    'timestamp': TimestampPattern(),
    'year': YearPattern(),
    'month': MonthPattern(),
    'day': DayPattern(),
    'week': WeekPattern(),
    'quarter': QuarterPattern(),
    'fiscal_year': FiscalYearPattern(),
    'duration': DurationPattern(),
    'currency': CurrencyPattern(),
    'salary': SalaryPattern(),
    'number_systems': NumberSystemsPattern(),
    'percentage': PercentagePattern(),
    'ratio': RatioPattern(),
    'integer': IntegerPattern(),
    'float': FloatPattern(),
    # Measurements (only primary patterns, aliases excluded)
    'distance': DistancePattern(),
    # 'length': LengthPattern(),  # Alias of distance, exclude to avoid ambiguity
    # 'width': WidthPattern(),     # Alias of distance, exclude to avoid ambiguity
    # 'height': HeightPattern(),   # Alias of distance, exclude to avoid ambiguity
    'weight': WeightPattern(),
    'volume': VolumePattern(),
    'area': AreaPattern(),
    'speed': SpeedPattern(),
    'temperature': TemperaturePattern(),
    'pressure': PressurePattern(),
    'energy': EnergyPattern(),
    'power': PowerPattern(),
    'capacity': CapacityPattern(),  # Re-enabled for storage capacity (GB, TB)
    'density': DensityPattern(),
    'angle': AnglePattern(),
    # Geographical patterns
    'latitude': LatitudePattern(),
    'longitude': LongitudePattern(),
    'geo_coordinate': GeoCoordinatePattern(),
    'timezone': TimezonePattern(),
    # Boolean/Flag patterns
    'boolean': BooleanPattern(),
    # Categorical patterns
    'gender': GenderPattern(),
    # Contact/Identity patterns
    'email': EmailPattern(),
    # Internet/Tech/System patterns
    'url': URLPattern(),
    'network_addresses': NetworkAddressesPattern(),
    'file_path': FilePathPattern(),
    'files': FilesPattern(),
    'version': VersionPattern(),
    'varchar': VarcharPattern(),
    'text': TextPattern(),
}
