@echo off
setlocal enabledelayedexpansion
cls

echo ======================================================
echo           EtlTele - Beginner's Interactive Guide
echo ======================================================
echo This tool will help you explore the features of EtlTele
echo using a local SQLite database.
echo.

:: Step 0: Ensure dependencies are ready
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Error: Python is not installed or not in PATH.
    pause
    exit /b
)

:MENU
echo.
echo [1] Setup: Generate "Dirty" Sample Data
echo [2] Feature: Full Load (create-load)
echo [3] Feature: Smart Update (with Where-Build)
echo [4] Feature: Smart Upsert (Conflict Handling)
echo [5] Feature: Auto-ETL (The Smart Controller)
echo [6] View Current Database State
echo [7] Reset/Cleanup
echo [8] Exit
echo.
set /p choice="Choose an option (1-8): "

if "%choice%"=="1" goto GENERATE
if "%choice%"=="2" goto LOAD
if "%choice%"=="3" goto UPDATE
if "%choice%"=="4" goto UPSERT
if "%choice%"=="5" goto AUTO
if "%choice%"=="6" goto VIEW
if "%choice%"=="7" goto RESET
if "%choice%"=="8" goto EXIT

echo Invalid choice, try again.
goto MENU

:GENERATE
echo.
echo Generating 'demo_dirty.csv' with:
echo - Hidden BOM character
echo - Visual homoglyphs (Cyrillic 'A' lookalikes)
echo - Smart quotes and trailing spaces
echo.
python -c "import pandas as pd; df = pd.DataFrame({'ID': [1, 2], 'NАME': ['Jo\u200bhnn', '“Jane” '], 'STATUS': ['Active', 'Pending']}); df.to_csv('demo_dirty.csv', index=False, encoding='utf-8-sig')"
echo ✅ File 'demo_dirty.csv' created.
goto MENU

:LOAD
echo.
echo Step: Loading 'demo_dirty.csv' into 'test_table'...
echo This will create the table and automatically clean the hidden characters.
python main.py --db-uri "sqlite:///demo.db" --rename-column create-load demo_dirty.csv test_table
echo ✅ Data loaded. Check 'where_build.log' for cleaning details.
goto MENU

:UPDATE
echo.
echo Step: Updating row ID=1.
echo We are changing Status to 'Inactive' using a specific WHERE filter.
echo Note: Column names are aligned automatically.
python main.py --db-uri "sqlite:///demo.db" update demo_dirty.csv test_table --where "[[\"ID\", \"=\", 1]]" --dtype "{\"STATUS\": \"Inactive\"}"
echo ✅ Update attempted.
goto MENU

:UPSERT
echo.
echo Step: Upserting data.
echo This will update existing IDs and insert new ones.
python -c "import pandas as pd; df = pd.DataFrame({'ID': [2, 3], 'NАME': ['Jane Fixed', 'New User'], 'STATUS': ['Active', 'New']}); df.to_csv('upsert_data.csv', index=False)"
python main.py --db-uri "sqlite:///demo.db" upsert upsert_data.csv test_table --constrain ID
echo ✅ Upsert completed.
goto MENU

:AUTO
echo.
echo Step: Running Auto-ETL.
echo The engine will detect that the table exists and perform an Upsert automatically.
python main.py --db-uri "sqlite:///demo.db" --add-missing-cols auto-etl upsert_data.csv test_table
echo ✅ Auto-ETL finished.
goto MENU

:VIEW
echo.
echo Current Table Content:
python -c "import sqlite3; conn = sqlite3.connect('demo.db'); cur = conn.cursor(); cur.execute('SELECT * FROM test_table'); [print(row) for row in cur.fetchall()]; conn.close()"
goto MENU

:RESET
echo Cleaning up environment...
if exist demo.db del demo.db
if exist demo_dirty.csv del demo_dirty.csv
if exist upsert_data.csv del upsert_data.csv
if exist where_build.log del where_build.log
echo ✅ Done.
goto MENU

:EXIT
echo Thank you for using EtlTele!
pause
exit /b
