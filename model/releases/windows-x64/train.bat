@echo off
REM Enhanced Training Script for AryanAlpha Windows
echo Starting AryanAlpha Enhanced Training...
echo.
echo Using enhanced parameters:
echo - Data: data\test_training_data.txt
echo - Epochs: 10
echo - Batch size: 8
echo - Sequence length: 128
echo - Model dimensions: 256
echo - Layers: 6
echo - Attention heads: 8
echo - Learning rate: 1e-4
echo.

REM Create models directory if it doesn't exist
if not exist models mkdir models

bin\Aaryan.exe --mode train --data data\test_training_data.txt --epochs 10 --batch_size 8 --seq_len 128 --d_model 256 --n_layers 6 --n_heads 8 --lr 1e-4 --weights_save models\aryan_model.bin --vocab_save models\aryan_vocab.vocab

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ✅ Training completed successfully!
    echo 📁 Files saved:
    echo   • Model weights: models\aryan_model.bin
    echo   • Vocabulary: models\aryan_vocab.vocab
) else (
    echo.
    echo ❌ Training failed with error code: %ERRORLEVEL%
)
pause
