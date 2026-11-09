@echo off
REM ============================================================================
REM  FUNDUS AI - Train toan bo cac bai toan
REM
REM  Cach dung (double-click hoac chay trong terminal):
REM    train.bat                              mo bang dieu khien interactive (chon module, resume, kiem tra)
REM    train.bat --status                     xem lich su va tien do cac run (train toi dau)
REM    train.bat --tasks vessels,lesions      chi train mot so bai toan (co the chon patch_lesions SOTA)
REM    train.bat --tasks patch_lesions        train module M4-SOTA Patch-Based IDRiD (2048px)
REM    train.bat --epochs 5                   ghi de so epoch cho moi bai toan
REM    train.bat --batch-size 4               ghi de batch co dinh (mac dinh da toi uu cho RTX 4060 8GB)
REM    train.bat --check-only                 chi kiem tra thiet bi + du lieu, khong train
REM    train.bat --smoke                      chay thu 1 epoch/8 anh moi bai truoc khi train (~1 phut)
REM    train.bat --extra "--lr 0.0001"        them tham so rieng cho script train cua cac bai da chon
REM    train.bat --approve 20261109_093724 --tasks lesions   DUYET: chep checkpoint + bieu do vao checkpoints\
REM    train.bat --promote                    tu duyet: chep ngay khi tung bai train xong
REM    train.bat --new                        train moi tu dau, KHONG tiep tuc run dang do
REM    train.bat --resume 20261107_143800     tiep tuc dung run nay (mac dinh: tu lay run dang do moi nhat)
REM
REM  TAM DUNG / TIEP TUC: bam Ctrl+C hoac dong cua so bat ky luc nao. Lan sau chay lai train.bat
REM    se TU DONG tiep tuc run dang do: bai da xong duoc bo qua, bai dang train chay tiep tu epoch
REM    da xong gan nhat (chi mat phan epoch dang chay do). Trang thai: runs\<ngay_gio>\run_state.json
REM
REM  Ket qua moi lan chay nam trong runs\<ngay_gio>\ (khong ghi de lan truoc):
REM    train_log.txt, SUMMARY.md, run_state.json, device_info.json, batch_sizes.json, preflight.json,
REM    logs\<bai>.epochs.csv / .epochs.jsonl (log tung epoch), logs\<bai>.console.log
REM
REM  Python: mac dinh dung Python global. Doi bang bien moi truong FUNDUS_PYTHON neu can.
REM ============================================================================
setlocal
chcp 65001 >nul
cd /d "%~dp0"
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
set "PYTHONUNBUFFERED=1"

if defined FUNDUS_PYTHON (
    set "PY=%FUNDUS_PYTHON%"
) else if exist ".venv\Scripts\python.exe" (
    set "PY=.venv\Scripts\python.exe"
) else if exist "C:\Program Files\Python310\python.exe" (
    set "PY=C:\Program Files\Python310\python.exe"
) else (
    set "PY=python"
)

"%PY%" --version >nul 2>&1
if errorlevel 1 (
    echo [X] Khong tim thay Python: "%PY%"
    echo     Cai Python hoac dat bien FUNDUS_PYTHON tro toi python.exe
    pause
    exit /b 1
)

echo Dung Python: "%PY%"
"%PY%" -u "scripts\train_all.py" %*
set "RC=%ERRORLEVEL%"

echo.
if "%RC%"=="0" (
    echo [OK] Hoan tat. Xem bao cao trong thu muc runs\ moi nhat ^(SUMMARY.md^).
) else (
    echo [!] Ket thuc voi ma loi %RC%. Xem runs\^<moi nhat^>\train_log.txt de biet chi tiet.
)
pause
exit /b %RC%
