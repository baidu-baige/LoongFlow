---
name: "file-processing"
description: "Data file processing utilities for CSV, JSON, and text files. Provides helpers for reading, transforming, and validating structured data."
---

# File Processing Skill

This skill provides utilities and guidance for building robust file processing applications.

## Purpose

Use this skill when your task involves:
- Reading and parsing CSV, JSON, or text files
- Data validation and cleaning
- File format conversions
- Batch processing of multiple files
- Generating reports from data files

## Key Capabilities

### 1. Data Reading
- CSV parsing with header detection
- JSON file handling (single object or array)
- Text file processing line-by-line
- Error handling for malformed files

### 2. Data Validation
- Check for required fields
- Validate data types
- Handle missing values
- Report data quality issues

### 3. Data Transformation
- Filter rows based on conditions
- Calculate statistics (sum, avg, count)
- Format conversions
- Data aggregation

### 4. Output Generation
- Write processed data to new files
- Generate summary reports
- Create multiple output formats

## Best Practices

### Project Structure for File Processing

```
project/
├── main.py              # Entry point with CLI
├── file_reader.py       # File I/O operations
├── data_processor.py    # Core processing logic
├── validator.py         # Data validation
├── config.py            # Configuration constants
└── utils.py             # Helper functions
```

### Error Handling Pattern

```python
def read_file_safely(filepath):
    """Read file with proper error handling"""
    try:
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")

        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        print(f"Error reading file: {e}")
        return None
```

### CSV Processing Template

```python
import csv

def process_csv(input_file, output_file):
    """Process CSV with header detection"""
    with open(input_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)

        processed = []
        for row in reader:
            # Transform each row
            processed_row = transform_row(row)
            processed.append(processed_row)

    # Write results
    with open(output_file, 'w', encoding='utf-8') as f:
        if processed:
            writer = csv.DictWriter(f, fieldnames=processed[0].keys())
            writer.writeheader()
            writer.writerows(processed)
```

### JSON Processing Template

```python
import json

def process_json(input_file, output_file):
    """Process JSON data"""
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Process data (handle both list and dict)
    processed = process_data(data)

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(processed, f, indent=2, ensure_ascii=False)
```

## Common Patterns

### 1. CLI with Argument Parsing

```python
import argparse

def main():
    parser = argparse.ArgumentParser(description='File Processor')
    parser.add_argument('input', help='Input file path')
    parser.add_argument('output', help='Output file path')
    parser.add_argument('--format', choices=['csv', 'json'], default='csv')

    args = parser.parse_args()
    process_file(args.input, args.output, args.format)
```

### 2. Batch Processing

```python
import glob

def process_directory(input_dir, output_dir, pattern='*.csv'):
    """Process all matching files in directory"""
    files = glob.glob(os.path.join(input_dir, pattern))

    for filepath in files:
        filename = os.path.basename(filepath)
        output_path = os.path.join(output_dir, f"processed_{filename}")
        process_file(filepath, output_path)
```

### 3. Progress Reporting

```python
def process_with_progress(items):
    """Process items with progress feedback"""
    total = len(items)
    for i, item in enumerate(items, 1):
        process_item(item)
        print(f"Progress: {i}/{total} ({i*100//total}%)", end='\r')
    print()  # New line when complete
```

## Tools Available

When implementing file processing tasks, you have access to:
- `Read` - Read file contents
- `Write` - Create new files
- `Edit` - Modify existing files
- `Glob` - Find files by pattern
- `Bash` - Run shell commands (e.g., `wc -l`, `head`)

## Testing Tips

Always test your file processor with:
1. **Empty files** - Should handle gracefully
2. **Malformed data** - CSV with wrong column count, invalid JSON
3. **Missing files** - Should provide clear error messages
4. **Large files** - Consider memory usage
5. **Special characters** - Unicode, newlines in CSV fields

## Example Task Breakdown

**Task**: "Create a CSV analyzer that calculates statistics"

**Suggested Steps**:
1. Read CSV file and detect headers
2. Parse data into structured format
3. Calculate statistics (count, sum, average) per column
4. Generate summary report
5. Write results to output file

**Recommended Structure**:
- `csv_analyzer.py` - Main program
- `stats.py` - Statistics calculations
- `report_generator.py` - Format output

## References

For more complex tasks, consider:
- Python's `csv` module for CSV handling
- `json` module for JSON operations
- `pathlib` for cross-platform file paths
- `pandas` for advanced data processing (if allowed)
