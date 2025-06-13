@echo off
REM Enhanced Text Generation Script for AryanAlpha Windows
echo Starting AryanAlpha Text Generation...
echo.

REM Check if trained model exists
if not exist "models\aryan_model.bin" (
    echo ❌ Error: No trained model found!
    echo Please run train.bat first to create a model.
    pause
    exit /b 1
)

REM Use provided prompt or default
set "PROMPT=%~1"
if "%PROMPT%"=="" set "PROMPT=The quick brown fox"
set "MAX_TOKENS=%~2"
if "%MAX_TOKENS%"=="" set "MAX_TOKENS=50"

echo Using trained model: models\aryan_model.bin
echo Vocabulary: models\aryan_vocab.vocab
echo Prompt: "%PROMPT%"
echo Max new tokens: %MAX_TOKENS%
echo.

bin\Aaryan.exe --mode generate --weights_load models\aryan_model.bin --vocab_load models\aryan_vocab.vocab --prompt "%PROMPT%" --max_new %MAX_TOKENS%

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ✅ Text generation completed!
) else (
    echo.
    echo ❌ Text generation failed with error code: %ERRORLEVEL%
)
pause
