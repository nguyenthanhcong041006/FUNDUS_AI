@echo off
REM ============================================================================
REM  FUNDUS AI - Train / resume / approve cac bai toan M1..M6
REM
REM  M1 optic_disc_cup | M2 vessels | M3 artery_vein | M4 lesions | M5 dr_grading | M6 multi_disease
REM  (M4S = patch_lesions, bai tuy chon, khong nam trong "tat ca")
REM
REM  1) TRAIN
REM    train.bat                       KHONG tham so = train TOAN BO 6 bai M1..M6 (neu dang co run do thi tu TIEP TUC run do)
REM    train.bat --tasks M1,M4         chi train M1 va M4 (cung dung duoc ten: vessels,lesions)
REM    train.bat --epochs 5 --batch-size 4 --image-size 512     ghi de cho TAT CA bai da chon
REM    train.bat --m1 "--epochs 30 --batch-size 4 --norm group --patience 10"
REM                                    cau hinh RIENG tung bai: dung --m1..--m6 (chi bai co --mN duoc train)
REM    train.bat --m1 "--epochs 30 --norm group" --m4 "--epochs 35 --batch-size 2 --lr 0.0002"
REM    train.bat --set M4="--epochs 35"   giong --m4 ; co the lap lai
REM       Trong "ARGS": --epochs / --batch-size / --image-size la tham so cua train.bat, moi tham so khac
REM       (--lr, --patience, --norm, ...) duoc chuyen thang cho script train cua bai do.
REM    train.bat --extra "--patience 8"   them tham so cho moi bai da chon
REM    train.bat --check-only          chi kiem tra thiet bi + du lieu, khong train
REM    train.bat --smoke               chay thu 1 epoch/8 anh moi bai truoc khi train (~1 phut)
REM    train.bat --new                 luon train moi tu dau, KHONG tiep tuc run dang do
REM    Co bat ky cau hinh nao (--tasks, --epochs, --mN, ...) thi train.bat luon tao run MOI.
REM
REM  2) RESUME (tam dung / tiep tuc): bam Ctrl+C hoac dong cua so bat ky luc nao.
REM    train.bat                       tu dong tiep tuc run dang do (bai da xong bo qua, bai dang do chay tiep
REM                                    tu epoch da xong gan nhat - chi mat phan epoch dang chay do)
REM    train.bat --resume              giong tren, chi dinh ro la tiep tuc
REM    train.bat --resume 20261107_143800   tiep tuc dung run nay trong runs\
REM    train.bat --status              xem lich su va tien do cac run
REM
REM  3) APPROVE (duyet): ket qua train nam trong runs\<ngay_gio>\ va CHUA tu ghi de project.
REM     Sau khi xem SUMMARY.md, duyet de copy ra project:
REM       checkpoint -> checkpoints\ | bieu do -> outputs\plots\<bai>\ | log + SUMMARY -> outputs\logs\train\<run>\
REM     (ban cu duoc sao luu trong runs\<run>\approved_backup\)
REM    train.bat --approve             duyet run moi nhat chua duyet (moi bai da DONE), co hoi xac nhan
REM    train.bat --approve 20261109_093724            duyet tat ca bai DONE cua run do
REM    train.bat --approve 20261109_093724 --tasks M1,M4   chi duyet M1 va M4 cua run do
REM    train.bat --approve 20261107_143800 --tasks M2,M3,M5,M6 --yes   khong hoi xac nhan
REM    train.bat --promote             (luc train) tu duyet ngay khi tung bai xong
REM    train.bat --menu                bang dieu khien interactive (train / resume / duyet / xem tien do)
REM
REM  Ket qua moi lan train nam trong runs\<ngay_gio>\ (khong ghi de lan truoc):
REM    train_log.txt, SUMMARY.md, run_state.json, device_info.json, batch_sizes.json, preflight.json,
REM    logs\<bai>.epochs.csv / .epochs.jsonl (log tung epoch), logs\<bai>.console.log
REM
REM  Python: dung FUNDUS_PYTHON neu co, roi .venv (neu chay duoc), roi Python global
REM  (C:\Program Files\Python310), roi "python" trong PATH.
REM ============================================================================
setlocal
chcp 65001 >nul
cd /d "%~dp0"
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
set "PYTHONUNBUFFERED=1"

set "PY="
if defined FUNDUS_PYTHON set "PY=%FUNDUS_PYTHON%"
if not defined PY if exist ".venv\Scripts\python.exe" call :try ".venv\Scripts\python.exe"
if not defined PY if exist "C:\Program Files\Python310\python.exe" call :try "C:\Program Files\Python310\python.exe"
if not defined PY call :try "python"

if not defined PY (
    echo [X] Khong tim thay Python chay duoc ^(.venv, Python310, PATH deu khong dung duoc^).
    echo     Cai Python hoac dat bien FUNDUS_PYTHON tro toi python.exe
    pause
    exit /b 1
)
"%PY%" --version >nul 2>&1
if errorlevel 1 (
    echo [X] Khong chay duoc Python: "%PY%"
    echo     Kiem tra bien FUNDUS_PYTHON
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

:try
REM chon interpreter %~1 neu no chay duoc (.venv copy tu may khac thi bi bo qua)
"%~1" --version >nul 2>&1
if not errorlevel 1 set "PY=%~1"
exit /b 0
