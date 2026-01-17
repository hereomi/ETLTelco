
import sys
import unittest
from unittest.mock import MagicMock
import pandas as pd
import numpy as np
from sqlalchemy.sql import sqltypes as sat

# Adding the path to the project to import modules
sys.path.append(r"e:\PyCode\_WebProject\telcoetl\etl_2\Tetl")

from schema_align.core import DataAligner

# Mock Oracle Types
class OracleNUMBER(sat.Numeric):
    def __init__(self, precision=None, scale=None):
        self.precision = precision
        self.scale = scale

class VerifySchemaPort(unittest.TestCase):
    
    def setUp(self):
        # Patch core module globals
        import schema_align.core as core
        self.original_oracle_avail = core.ORACLE_TYPES_AVAILABLE
        self.original_number = core.NUMBER
        
        core.ORACLE_TYPES_AVAILABLE = True
        core.NUMBER = OracleNUMBER
        
        self.aligner = DataAligner()
        print(f"DEBUG: ORACLE_TYPES_AVAILABLE={core.ORACLE_TYPES_AVAILABLE}, NUMBER={core.NUMBER}")
    
    def tearDown(self):
        import schema_align.core as core
        core.ORACLE_TYPES_AVAILABLE = self.original_oracle_avail
        core.NUMBER = self.original_number
        
    def test_oracle_number_integer_detection(self):
        """Test that Oracle NUMBER with scale 0 is detected as integer."""
        # Case 1: NUMBER(10, 0) -> Integer
        oracle_int = OracleNUMBER(10, 0)
        self.assertTrue(self.aligner._is_int_type(oracle_int), "Oracle NUMBER(10, 0) should be integer")
        # Check precedence
        target_type = self.aligner._determine_target_type(oracle_int)
        self.assertEqual(target_type, 'integer', "Oracle NUMBER(10, 0) should resolve to 'integer'")
        
    def test_oracle_number_float_detection(self):
        """Test that Oracle NUMBER with scale > 0 is detected as float."""
        # Case 2: NUMBER(10, 2) -> Float
        oracle_float = OracleNUMBER(10, 2)
        target_type = self.aligner._determine_target_type(oracle_float)
        self.assertEqual(target_type, 'float', "Oracle NUMBER(10, 2) should resolve to 'float'")
        
    def test_finalize_types_compatibility(self):
        """Test conversion of Pandas nullable types to native Python types."""
        df = pd.DataFrame({
            'int_col': pd.Series([1, 2, None], dtype="Int64"),
            'float_col': pd.Series([1.1, 2.2, None], dtype="Float64"),
            'bool_col': pd.Series([True, False, None], dtype="boolean"),
            'str_col': pd.Series(['a', 'b', None], dtype="string")
        })
        
        finalized = self.aligner._finalize_types(df)
        
        # Check Int64 -> object (int/None)
        self.assertEqual(finalized['int_col'].dtype, 'object')
        self.assertIsInstance(finalized['int_col'][0], int)
        self.assertIsNone(finalized['int_col'][2])
        self.assertNotIsInstance(finalized['int_col'][0], np.integer) # Should be native int
        
        # Check Float64 -> object (float/None)
        # Note: Depending on implementation details, it might be object or float64 with NaNs if using standard logic.
        # But _finalize_types explicitly converts to object for safety.
        self.assertEqual(finalized['float_col'].dtype, 'object')
        self.assertIsInstance(finalized['float_col'][0], float)
        self.assertIsNone(finalized['float_col'][2])
        
        # Check Boolean -> object (bool/None)
        self.assertEqual(finalized['bool_col'].dtype, 'object')
        self.assertIsInstance(finalized['bool_col'][0], (bool, np.bool_)) # numpy bool is usually accepted by drivers, but let's check strict python bool preference if implemented
        # The implementation uses astype(object), so it might preserve numpy bools or python bools depending on upstream.
        # Let's verify it acts like a python bool (is check might fail if it's numpy.bool_ implementation detail, but equality works)
        self.assertEqual(finalized['bool_col'][0], True)
        self.assertIsNone(finalized['bool_col'][2])

if __name__ == '__main__':
    unittest.main()
