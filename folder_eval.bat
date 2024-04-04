@echo off
setlocal

rem Check if the correct number of arguments are provided
if "%~2"=="" (
    echo Usage: %~nx0 MODEL_PATH FOLDER
    exit /b 1
)

rem Set input arguments
set "MODEL_PATH=%~1"
set "FOLDER=%~2"

rem Iterate over each .txt file in the folder
for %%I in ("%FOLDER%\*.txt") do (
    rem Run the evaluation command for the file
    echo Evaluating "%%~nxI"
    python eval.py "%MODEL_PATH%" "%FOLDER%" -f "%%~nxI"
)
