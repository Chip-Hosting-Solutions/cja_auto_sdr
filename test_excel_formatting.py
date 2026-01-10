#!/usr/bin/env python3
"""
Test script to generate a sample Excel file verifying the enhanced formatting.
"""

import pandas as pd
import logging
import os
import sys

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# Import the formatting function and DataQualityChecker
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# We need to mock the cjapy import since it's not available
sys.modules['cjapy'] = type(sys)('cjapy')

from cja_sdr_generator import apply_excel_formatting, DataQualityChecker

def create_sample_issues():
    """Create sample data quality issues for testing"""
    checker = DataQualityChecker(logger)

    # Add CRITICAL issues
    checker.add_issue('CRITICAL', 'Schema', 'Metric', 'revenue_total',
                      'Missing required field: id', 'Field "id" is required but not present')
    checker.add_issue('CRITICAL', 'Data', 'Dimension', 'customer_segment',
                      'Empty dataset detected', 'No records found in dimension data')

    # Add HIGH issues
    checker.add_issue('HIGH', 'Duplicates', 'Metric', 'page_views',
                      'Duplicate metric name found', 'Metric "page_views" appears 3 times')
    checker.add_issue('HIGH', 'Validation', 'Dimension', 'product_id',
                      'Invalid ID format', 'ID contains special characters')
    checker.add_issue('HIGH', 'Duplicates', 'Metric', 'conversion_rate',
                      'Duplicate metric name found', 'Metric appears in multiple data views')

    # Add MEDIUM issues
    checker.add_issue('MEDIUM', 'Null Values', 'Metric', 'avg_session_duration',
                      'Null values in critical field', '15% of records have null values')
    checker.add_issue('MEDIUM', 'Null Values', 'Dimension', 'campaign_source',
                      'Null values detected', '8% of records missing campaign source')
    checker.add_issue('MEDIUM', 'Format', 'Metric', 'bounce_rate',
                      'Inconsistent number format', 'Some values as percentage, others as decimal')

    # Add LOW issues
    checker.add_issue('LOW', 'Documentation', 'Metric', 'custom_metric_1',
                      'Missing description', 'No description provided for metric')
    checker.add_issue('LOW', 'Documentation', 'Dimension', 'custom_dim_1',
                      'Missing description', 'No description provided for dimension')
    checker.add_issue('LOW', 'Documentation', 'Metric', 'custom_metric_2',
                      'Missing description', 'Consider adding documentation')
    checker.add_issue('LOW', 'Best Practice', 'Dimension', 'user_type',
                      'Non-standard naming convention', 'Consider using snake_case')

    # Add INFO issue
    checker.add_issue('INFO', 'Summary', 'All', 'N/A',
                      'Validation complete', 'All critical checks passed')

    return checker.get_issues_dataframe()

def create_sample_metadata():
    """Create sample metadata for testing"""
    return pd.DataFrame({
        'Property': ['Data View', 'Generated', 'Total Metrics', 'Total Dimensions', 'Issues Found'],
        'Value': ['Sample Data View', '2026-01-09', '25', '18', '13']
    })

def create_sample_metrics():
    """Create sample metrics for testing (lowercase column names like API)"""
    return pd.DataFrame({
        'id': ['metric_1', 'metric_2', 'metric_3', 'metric_4', 'metric_5'],
        'name': ['Page Views', 'Revenue Total', 'Conversion Rate', 'Average Session Duration', 'Bounce Rate'],
        'type': ['counter', 'currency', 'percentage', 'time', 'percentage'],
        'title': ['Page Views', 'Revenue', 'Conv Rate', 'Avg Session', 'Bounce'],
        'description': [
            'Total page views across all pages',
            'Total revenue in USD from all transactions',
            'Conversion rate as percentage of sessions',
            'Average time spent per session in seconds',
            'Percentage of single-page sessions'
        ]
    })

def create_sample_dimensions():
    """Create sample dimensions for testing (lowercase column names like API)"""
    return pd.DataFrame({
        'id': ['dim_1', 'dim_2', 'dim_3', 'dim_4', 'dim_5'],
        'name': ['Campaign Source', 'Product Category', 'Geographic Region', 'User Segment', 'Device Type'],
        'type': ['string', 'string', 'string', 'string', 'string'],
        'title': ['Campaign', 'Product', 'Region', 'Segment', 'Device'],
        'description': [
            'Marketing campaign source attribution',
            'Product category classification',
            'Geographic region of user',
            'User behavioral segment classification',
            'Device type used for session'
        ]
    })

def main():
    output_file = 'test_excel_output.xlsx'
    logger.info("Generating sample Excel file with enhanced formatting...")
    logger.info("=" * 60)

    # Create sample data
    data_quality_df = create_sample_issues()
    metadata_df = create_sample_metadata()
    metrics_df = create_sample_metrics()
    dimensions_df = create_sample_dimensions()

    # Log the data quality DataFrame to verify sorting
    logger.info("\nData Quality Issues (sorted by severity):")
    logger.info("-" * 40)
    for idx, row in data_quality_df.iterrows():
        logger.info(f"  {row['Severity']:10} | {row['Category']:15} | {row['Item Name']}")

    # Count by severity
    logger.info("\nSeverity Summary:")
    logger.info("-" * 40)
    severity_counts = data_quality_df['Severity'].value_counts()
    for sev in ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW', 'INFO']:
        count = severity_counts.get(sev, 0)
        if count > 0:
            logger.info(f"  {sev}: {count}")

    # Create Excel file
    logger.info(f"\nWriting Excel file: {output_file}")
    logger.info("-" * 40)

    with pd.ExcelWriter(output_file, engine='xlsxwriter') as writer:
        # Apply formatting to each sheet
        sheets = {
            'Metadata': metadata_df,
            'Data Quality': data_quality_df,
            'Metrics': metrics_df,
            'Dimensions': dimensions_df
        }

        for sheet_name, df in sheets.items():
            apply_excel_formatting(writer, df, sheet_name, logger)

    logger.info("=" * 60)
    logger.info(f"SUCCESS: Excel file created at: {os.path.abspath(output_file)}")
    logger.info("\nEnhancements to verify in Excel:")
    logger.info("  Data Quality sheet:")
    logger.info("    1. Summary table at top with severity counts")
    logger.info("    2. Issues sorted: CRITICAL -> HIGH -> MEDIUM -> LOW -> INFO")
    logger.info("    3. Severity column has icons and bold text")
    logger.info("    4. Each severity level has distinct color coding")
    logger.info("  Metrics/Dimensions sheets:")
    logger.info("    5. Columns reordered: name first, then type, id, title, description")
    logger.info("    6. Name column is bold for quick scanning")
    logger.info("    7. Narrower column widths (description capped at 55 chars)")

if __name__ == '__main__':
    main()
