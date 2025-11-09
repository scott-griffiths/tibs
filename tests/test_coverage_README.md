# Test Coverage Documentation

This directory contains `test_coverage.py`, a comprehensive set of edge case tests for the bitformat library.

## Purpose

The `test_coverage.py` file focuses on testing edge cases and scenarios that may not be fully covered by existing tests. It uses property-based testing with Hypothesis to automatically discover edge cases.

## Test Categories

### 1. Bits Edge Cases (`TestBitsEdgeCases`)
- Empty bits operations
- Single bit manipulations  
- Large integer roundtrip testing
- Signed integer boundary conditions
- Float special values (NaN, infinity, etc.)
- Bytes roundtrip with edge cases
- Bit manipulation at boundaries

### 2. Array Edge Cases (`TestArrayEdgeCases`)
- Empty array operations
- Single element arrays
- Dtype conversion edge cases
- Extreme values for array dtypes
- Various bit width testing

### 3. Dtype Edge Cases (`TestDtypeEdgeCases`)
- Extreme dtype sizes
- String parsing edge cases
- Array dtype boundaries
- Tuple dtype edge cases

### 4. Format Edge Cases (`TestFormatEdgeCases`)
- Empty field configurations
- Extreme nesting
- Malformed data parsing
- Random data parsing

### 5. Endianness Edge Cases (`TestEndiannesssEdgeCases`)
- Endianness consistency
- Roundtrip testing for all endianness modes

### 6. Error Conditions (`TestErrorConditionsAndBoundaries`)
- Memory efficiency edge cases
- Unicode and encoding edge cases
- Type conversion boundaries
- String input validation

### 7. Hypothesis Property-Based Tests (`TestHypothesisPropertyBasedEdgeCases`)
- Roundtrip properties
- Concatenation properties
- Bytes conversion properties

### 8. Implementation-Specific Tests (`TestImplementationSpecificEdgeCases`)
- Rust/Python integration edge cases
- Memory management across boundaries
- Thread safety indicators

## Key Features

### Property-Based Testing
Uses Hypothesis to generate test cases automatically:
```python
@given(st.integers(min_value=0, max_value=2**64 - 1))
def test_large_integer_roundtrip(self, value):
    # Tests roundtrip for random large integers
```

### Documented Failing Tests
Tests that may legitimately fail are documented with explanations:
```python
# POSSIBLE FAILING TEST: Some signed integer values near boundaries
# might not roundtrip correctly due to two's complement representation
```

### Edge Case Focus
Concentrates on boundary conditions that might not be tested elsewhere:
- Zero-length inputs
- Maximum/minimum values
- Special floating point values
- Memory boundaries
- Encoding edge cases

## Running the Tests

### Prerequisites
The tests require a fully built bitformat package with all dependencies:
- Python 3.11+
- pytest
- hypothesis
- lark parser library
- Built Rust extension module

### Execution
```bash
# Run all coverage tests
pytest tests/test_coverage.py -v

# Run with specific hypothesis settings
pytest tests/test_coverage.py -v --hypothesis-show-statistics

# Run a specific test class
pytest tests/test_coverage.py::TestBitsEdgeCases -v
```

### Demo Version
A runnable demo version is available that shows the testing patterns without requiring the full bitformat build:
```bash
python test_coverage_runnable_demo.py
```

## Expected Outcomes

### Passing Tests
Most tests should pass, indicating robust edge case handling.

### Documented Failing Tests
Some tests may fail and are documented as such. These failures indicate:
1. Legitimate edge cases that reveal implementation boundaries
2. Areas where the library's behavior at edges might need clarification
3. Potential areas for future improvement

**Important**: Failing tests in this file are NOT meant to be "fixed" by modifying the library code without careful consideration. They are meant to document and explore the edge cases.

### Test Skipping
If dependencies are missing, all tests will be skipped with appropriate messages.

## Contributing

When adding new tests to this file:

1. Focus on edge cases and boundary conditions
2. Use Hypothesis for property-based testing where appropriate
3. Document any tests that may legitimately fail
4. Group related tests into logical test classes
5. Include clear docstrings explaining what each test explores

## Integration with CI

These tests are designed to be part of the regular test suite but with understanding that some edge case failures might be acceptable and documented.