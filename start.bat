@echo off
REM Launch the J-Wash server on http://localhost:8381
REM Activate your Python environment first (e.g. `conda activate jwash`),
REM then run this script — or simply `python -X utf8 run.py`.
cd /d %~dp0
python -X utf8 run.py
