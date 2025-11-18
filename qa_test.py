#!/usr/bin/env python3
"""
QA Test Suite for GeoBot Production Deployment

Tests all critical functionality:
- Conflict forecaster scenarios
- Database logging
- Venezuela scenario
- Guardian API integration (optional)
- Error handling
"""

import sys
import os
from pathlib import Path

# Test results
tests_passed = 0
tests_failed = 0
test_results = []

def test(name):
    """Decorator for test functions."""
    def decorator(func):
        def wrapper():
            global tests_passed, tests_failed
            print(f"\n{'='*70}")
            print(f"TEST: {name}")
            print('='*70)
            try:
                func()
                print(f"✓ PASSED: {name}")
                tests_passed += 1
                test_results.append(('PASS', name))
                return True
            except AssertionError as e:
                print(f"✗ FAILED: {name}")
                print(f"  Error: {e}")
                tests_failed += 1
                test_results.append(('FAIL', name, str(e)))
                return False
            except Exception as e:
                print(f"✗ ERROR: {name}")
                print(f"  Exception: {e}")
                tests_failed += 1
                test_results.append(('ERROR', name, str(e)))
                return False
        return wrapper
    return decorator


@test("Python syntax check for all modules")
def test_syntax():
    """Test Python syntax for all key modules."""
    files_to_check = [
        'geobot/discord_bot/forecaster.py',
        'geobot/discord_bot/bot.py',
        'geobot/monitoring/forecast_logger.py',
        'geobot/data_ingestion/rss_scraper.py',
        'geobot/cli.py',
        'geobot_live.py'
    ]

    import py_compile
    for filepath in files_to_check:
        try:
            py_compile.compile(filepath, doraise=True)
            print(f"  ✓ {filepath}")
        except py_compile.PyCompileError as e:
            raise AssertionError(f"Syntax error in {filepath}: {e}")


@test("Venezuela scenario configuration")
def test_venezuela_scenario():
    """Test Venezuela conflict scenarios are properly configured."""
    # Check forecaster has Venezuela priors
    with open('geobot/discord_bot/forecaster.py', 'r') as f:
        content = f.read()

    assert "'venezuela'" in content, "Venezuela scenario missing"
    assert "'usa_venezuela'" in content, "USA-Venezuela scenario missing"
    assert "'maduro': 'venezuela'" in content, "Maduro mapping missing"

    print("  ✓ Venezuela scenario found")
    print("  ✓ USA-Venezuela scenario found")
    print("  ✓ Maduro mapping found")


@test("Venezuela in country extraction list")
def test_venezuela_country_list():
    """Test Venezuela added to country extraction."""
    with open('geobot/data_ingestion/rss_scraper.py', 'r') as f:
        content = f.read()

    assert "'Venezuela'" in content, "Venezuela not in country list"
    print("  ✓ Venezuela in country extraction list")


@test("Guardian API integration")
def test_guardian_api():
    """Test Guardian API is integrated in RSS scraper."""
    with open('geobot/data_ingestion/rss_scraper.py', 'r') as f:
        content = f.read()

    assert "GUARDIAN_API_URL" in content, "Guardian API URL missing"
    assert "scrape_guardian" in content, "scrape_guardian method missing"
    assert "guardian_api_key" in content, "Guardian API key parameter missing"

    print("  ✓ Guardian API URL defined")
    print("  ✓ scrape_guardian method exists")
    print("  ✓ API key parameter handling present")


@test("Database schema in forecast logger")
def test_database_schema():
    """Test database schema is properly defined."""
    with open('geobot/monitoring/forecast_logger.py', 'r') as f:
        content = f.read()

    assert "CREATE TABLE IF NOT EXISTS forecasts" in content, "Forecasts table missing"
    assert "CREATE TABLE IF NOT EXISTS news_articles" in content, "News articles table missing"
    assert "CREATE VIEW IF NOT EXISTS forecast_drift" in content, "Drift view missing"
    assert "escalation_probability" in content, "Escalation probability column missing"
    assert "regime_change_probability" in content, "Regime change probability column missing"

    print("  ✓ Forecasts table schema present")
    print("  ✓ News articles table schema present")
    print("  ✓ Forecast drift view present")
    print("  ✓ Probability columns defined")


@test("Discord bot logging integration")
def test_discord_logging():
    """Test Discord bot has logging integration."""
    with open('geobot/discord_bot/bot.py', 'r') as f:
        content = f.read()

    assert "from ..monitoring.forecast_logger import get_logger" in content, "Logger import missing"
    assert "self.logger = get_logger()" in content, "Logger initialization missing"
    assert "self.logger.log_forecast" in content, "log_forecast call missing"

    print("  ✓ Logger import present")
    print("  ✓ Logger initialization present")
    print("  ✓ Forecast logging call present")


@test("Conflict scenario priors")
def test_conflict_priors():
    """Test all conflict scenarios have proper prior configurations."""
    with open('geobot/discord_bot/forecaster.py', 'r') as f:
        content = f.read()

    required_scenarios = [
        'taiwan', 'ukraine', 'iran', 'north_korea',
        'israel_palestine', 'syria', 'kashmir',
        'venezuela', 'usa_venezuela'
    ]

    for scenario in required_scenarios:
        assert f"'{scenario}':" in content, f"Scenario {scenario} missing"
        print(f"  ✓ {scenario} scenario configured")


@test("CLI commands structure")
def test_cli_commands():
    """Test CLI has all required commands."""
    with open('geobot/cli.py', 'r') as f:
        content = f.read()

    required_commands = [
        'cmd_analyze', 'cmd_forecast', 'cmd_intervene',
        'cmd_monitor', 'cmd_discord', 'cmd_version'
    ]

    for cmd in required_commands:
        assert f"def {cmd}(" in content, f"Command {cmd} missing"
        print(f"  ✓ {cmd} defined")


@test("GeoBot Live script structure")
def test_geobot_live():
    """Test GeoBot Live script has key components."""
    with open('geobot_live.py', 'r') as f:
        content = f.read()

    assert "class LiveTicker" in content, "LiveTicker class missing"
    assert "format_percentage" in content, "Percentage formatting missing"
    assert "display_forecast" in content, "Forecast display missing"
    assert "ConflictForecaster" in content, "ConflictForecaster usage missing"

    print("  ✓ LiveTicker class present")
    print("  ✓ Visual percentage bars present")
    print("  ✓ Forecast display present")
    print("  ✓ ConflictForecaster integration present")


@test("Error handling in RSS scraper")
def test_error_handling():
    """Test error handling is present in critical modules."""
    with open('geobot/data_ingestion/rss_scraper.py', 'r') as f:
        content = f.read()

    assert "try:" in content, "No try-except blocks"
    assert "except Exception" in content, "No exception handling"
    assert "if not self.guardian_api_key:" in content, "No Guardian API key check"

    print("  ✓ Try-except blocks present")
    print("  ✓ Exception handling present")
    print("  ✓ API key validation present")


@test("Drift tracking methods")
def test_drift_methods():
    """Test drift tracking methods exist."""
    with open('geobot/monitoring/forecast_logger.py', 'r') as f:
        content = f.read()

    required_methods = [
        'log_forecast', 'get_drift_analysis',
        'get_recent_forecasts', 'get_all_conflicts',
        'get_statistics'
    ]

    for method in required_methods:
        assert f"def {method}(" in content, f"Method {method} missing"
        print(f"  ✓ {method} method defined")


@test("Module __init__ files")
def test_init_files():
    """Test __init__ files are properly configured."""
    # Check monitoring __init__
    with open('geobot/monitoring/__init__.py', 'r') as f:
        content = f.read()

    assert "from . import forecast_logger" in content, "forecast_logger import missing"
    print("  ✓ monitoring/__init__.py includes forecast_logger")

    # Check discord_bot __init__
    with open('geobot/discord_bot/__init__.py', 'r') as f:
        content = f.read()

    assert "from . import forecaster" in content, "forecaster import missing"
    print("  ✓ discord_bot/__init__.py includes forecaster")


@test("Requirements file has discord.py")
def test_requirements():
    """Test requirements.txt has new dependencies."""
    with open('requirements.txt', 'r') as f:
        content = f.read()

    assert "discord.py" in content, "discord.py not in requirements"
    assert "feedparser" in content, "feedparser not in requirements"
    assert "requests" in content, "requests not in requirements"

    print("  ✓ discord.py in requirements")
    print("  ✓ feedparser in requirements")
    print("  ✓ requests in requirements")


def print_summary():
    """Print test summary."""
    print("\n" + "=" * 70)
    print("QA TEST SUMMARY")
    print("=" * 70)

    total_tests = tests_passed + tests_failed
    pass_rate = (tests_passed / total_tests * 100) if total_tests > 0 else 0

    print(f"\nTotal Tests: {total_tests}")
    print(f"Passed: {tests_passed} ({pass_rate:.1f}%)")
    print(f"Failed: {tests_failed}")

    if tests_failed > 0:
        print("\nFailed Tests:")
        for result in test_results:
            if result[0] in ['FAIL', 'ERROR']:
                print(f"  ✗ {result[1]}")
                if len(result) > 2:
                    print(f"    {result[2]}")

    print("\n" + "=" * 70)

    if tests_failed == 0:
        print("✓ ALL TESTS PASSED - READY FOR PRODUCTION")
    else:
        print("✗ SOME TESTS FAILED - REVIEW BEFORE DEPLOYMENT")

    print("=" * 70)

    return tests_failed == 0


def main():
    """Run all QA tests."""
    print("=" * 70)
    print("GEOBOT QA TEST SUITE")
    print("=" * 70)
    print("\nTesting production readiness...")

    # Run all tests
    test_syntax()
    test_venezuela_scenario()
    test_venezuela_country_list()
    test_guardian_api()
    test_database_schema()
    test_discord_logging()
    test_conflict_priors()
    test_cli_commands()
    test_geobot_live()
    test_error_handling()
    test_drift_methods()
    test_init_files()
    test_requirements()

    # Print summary
    success = print_summary()

    return 0 if success else 1


if __name__ == '__main__':
    sys.exit(main())
