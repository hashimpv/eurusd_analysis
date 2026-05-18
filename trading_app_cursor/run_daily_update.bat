@echo off
REM Daily EURUSD -> PostgreSQL (Alpha Vantage compact, ~100 sessions).
REM If DB is several days behind, run once manually: py -3 -m fetch.daily_update --full
cd /d "%~dp0"
py -3 -m fetch.daily_update
if errorlevel 1 exit /b 1
