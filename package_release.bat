@echo off
setlocal enabledelayedexpansion

:: ============================================================================
:: CBIR GUI Release Packaging Script
:: Creates a distributable zip file with exe, DLLs, and sample data
:: ============================================================================

set VERSION=1.0.0
set OPENCV_BIN=C:\opencv_build\build\install\x64\vc17\bin
set RELEASE_NAME=cbir_gui_v%VERSION%_win64
set RELEASE_DIR=release\%RELEASE_NAME%

echo.
echo ========================================
echo  CBIR GUI Release Packager v%VERSION%
echo ========================================
echo.

:: Clean previous release
if exist "%RELEASE_DIR%" (
    echo Cleaning previous release...
    rmdir /s /q "%RELEASE_DIR%"
)
mkdir "%RELEASE_DIR%"
mkdir "%RELEASE_DIR%\data"

:: Copy executable
echo Copying executable...
if not exist "bin\cbir_gui.exe" (
    echo ERROR: bin\cbir_gui.exe not found!
    echo Run build.bat first.
    exit /b 1
)
copy "bin\cbir_gui.exe" "%RELEASE_DIR%\"

:: Copy OpenCV DLLs
echo Copying OpenCV DLLs...
if exist "%OPENCV_BIN%\opencv_world4130.dll" (
    copy "%OPENCV_BIN%\opencv_world4130.dll" "%RELEASE_DIR%\"
) else (
    echo Looking for individual OpenCV DLLs...
    for %%f in ("%OPENCV_BIN%\opencv_core*.dll" "%OPENCV_BIN%\opencv_imgproc*.dll" "%OPENCV_BIN%\opencv_imgcodecs*.dll" "%OPENCV_BIN%\opencv_highgui*.dll") do (
        if exist "%%f" copy "%%f" "%RELEASE_DIR%\"
    )
)

:: Copy sample data (just a few images + CSV)
echo Copying sample data...
if exist "data\ResNet18_olym.csv" (
    copy "data\ResNet18_olym.csv" "%RELEASE_DIR%\data\"
)

:: Copy a subset of sample images (first 20)
if exist "data\olympus" (
    mkdir "%RELEASE_DIR%\data\olympus"
    set count=0
    for %%f in ("data\olympus\*.jpg") do (
        if !count! lss 20 (
            copy "%%f" "%RELEASE_DIR%\data\olympus\" >nul
            set /a count+=1
        )
    )
    echo Copied !count! sample images
)

:: Create README
echo Creating README...
(
echo CBIR GUI - Content-Based Image Retrieval
echo =========================================
echo.
echo Version: %VERSION%
echo.
echo QUICK START:
echo 1. Double-click cbir_gui.exe to run
echo 2. Click "Browse..." to select a query image
echo 3. Choose a feature type from the dropdown
echo 4. Click "Search" to find similar images
echo.
echo REQUIREMENTS:
echo - Windows 10/11 (64-bit^)
echo - Visual C++ 2022 Redistributable
echo   Download: https://aka.ms/vs/17/release/vc_redist.x64.exe
echo.
echo FEATURES:
echo - 7 feature extraction methods
echo - Drag and drop image support
echo - Resizable split-panel layout
echo - DNN embedding support (requires CSV file^)
echo.
echo For full documentation, visit:
echo https://github.com/ParkerCai/PRCV-Project2-Content-Based-Image-Retrieval
) > "%RELEASE_DIR%\README.txt"

:: Create zip using PowerShell
echo.
echo Creating zip archive...
if exist "release\%RELEASE_NAME%.zip" del "release\%RELEASE_NAME%.zip"
powershell -Command "Compress-Archive -Path '%RELEASE_DIR%' -DestinationPath 'release\%RELEASE_NAME%.zip'"

echo.
echo ========================================
echo  Release package created successfully!
echo ========================================
echo.
echo Output: release\%RELEASE_NAME%.zip
echo.
echo Contents:
dir /b "%RELEASE_DIR%"
echo.
echo Next steps:
echo 1. Test the release on another machine
echo 2. Upload to GitHub Releases
echo.

endlocal
