#!/usr/bin/env python3
"""
Script to run all behavior-focused tests in the project.
This provides detailed reporting on test results.
"""

import os
import sys
import unittest
import time
import argparse
from pathlib import Path

# Add the src directory to the path
sys.path.append(str(Path(__file__).resolve().parent.parent))

class DetailedTestResult(unittest.TextTestResult):
    """Custom test result class that provides more detailed output."""
    
    def __init__(self, stream, descriptions, verbosity):
        super().__init__(stream, descriptions, verbosity)
        self.successes = []
        self.start_times = {}
        self.durations = {}
        
    def startTest(self, test):
        """Record the start time of each test."""
        super().startTest(test)
        self.start_times[test] = time.time()
        
    def addSuccess(self, test):
        """Record successful tests."""
        super().addSuccess(test)
        self.successes.append(test)
        self.durations[test] = time.time() - self.start_times[test]
        
    def addError(self, test, err):
        """Record error with duration."""
        super().addError(test, err)
        self.durations[test] = time.time() - self.start_times[test]
        
    def addFailure(self, test, err):
        """Record failure with duration."""
        super().addFailure(test, err)
        self.durations[test] = time.time() - self.start_times[test]
        
    def addSkip(self, test, reason):
        """Record skip with duration."""
        super().addSkip(test, reason)
        self.durations[test] = time.time() - self.start_times[test]

def run_behavior_tests(pattern=None, verbose=False, output_file=None):
    """
    Run behavior-focused tests and report results.
    
    Args:
        pattern: Optional pattern to filter test names
        verbose: Whether to show verbose output
        output_file: Optional file to save results
    """
    # Discover all tests
    test_dir = Path(__file__).parent.parent / 'tests'
    loader = unittest.TestLoader()
    
    # Filter for behavior tests
    if pattern:
        suite = loader.discover(test_dir, pattern=pattern)
    else:
        behavior_tests = unittest.TestSuite()
        
        # Add test files that focus on behavior testing
        behavior_patterns = [
            "test_*_behavior.py",
            "test_veto_behavior.py",
            "test_rl_behavior.py", 
            "test_experiment_behavior.py"
        ]
        
        for pattern in behavior_patterns:
            behavior_tests.addTests(loader.discover(test_dir, pattern=pattern))
            
        suite = behavior_tests
    
    # Create test runner with custom result class
    stream = open(output_file, 'w') if output_file else sys.stdout
    runner = unittest.TextTestRunner(
        resultclass=DetailedTestResult,
        stream=stream,
        verbosity=2 if verbose else 1
    )
    
    print(f"\n{'='*70}")
    print(f"Running behavior tests...")
    print(f"{'='*70}\n")
    
    # Run tests
    result = runner.run(suite)
    
    # Print summary
    print(f"\n{'='*70}")
    print(f"BEHAVIOR TEST SUMMARY:")
    print(f"{'='*70}")
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {len(result.successes)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped)}")
    
    # Calculate total time
    total_time = sum(result.durations.values())
    print(f"\nTotal time: {total_time:.2f} seconds")
    
    # Print slowest tests
    if result.durations:
        print(f"\nSlowest tests:")
        sorted_tests = sorted(result.durations.items(), key=lambda x: x[1], reverse=True)
        for test, duration in sorted_tests[:5]:
            print(f"  {test.id()}: {duration:.2f} seconds")
    
    # Print failures and errors
    if result.failures:
        print(f"\nFAILURES:")
        for test, trace in result.failures:
            print(f"  {test.id()}")
            
    if result.errors:
        print(f"\nERRORS:")
        for test, trace in result.errors:
            print(f"  {test.id()}")
    
    # Close output file if specified
    if output_file:
        stream.close()
        print(f"\nDetailed results saved to {output_file}")
    
    return result

def main():
    parser = argparse.ArgumentParser(description='Run behavior-focused tests')
    parser.add_argument('--pattern', type=str, help='Pattern to filter test names')
    parser.add_argument('--verbose', '-v', action='store_true', help='Show verbose output')
    parser.add_argument('--output', type=str, help='Save results to file')
    
    args = parser.parse_args()
    
    result = run_behavior_tests(args.pattern, args.verbose, args.output)
    
    # Return non-zero exit code if any tests failed
    if result.failures or result.errors:
        sys.exit(1)

if __name__ == '__main__':
    main()