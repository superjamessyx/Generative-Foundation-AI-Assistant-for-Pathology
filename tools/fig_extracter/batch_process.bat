@echo off
echo workdir = %~dp0
@REM echo Work Dir is %workdir%
@REM echo %workdir%
set workdir = %~dp0
for /r %workdir% %%f in (*.docx) do (
    echo %%~xnf
    echo Current Docx: %%~nf
    md %%~nf
    move %%f %%~xnf
    @REM pandoc --extract-media 
)

pause