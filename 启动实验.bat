@echo off
chcp 65001 >nul
title Attention Lab - Launcher
cd /d "%~dp0"

echo ============================================
echo   Attention Lab / 注意实验室 - 本地启动器
echo ============================================
echo.

set PY=
where python >nul 2>nul && set PY=python
if "%PY%"=="" (where py >nul 2>nul && set PY=py)
if "%PY%"=="" (
  echo [错误] 未检测到 Python。请先安装 Python，或改用其他方式启动本地服务器。
  echo 下载地址：https://www.python.org/downloads/
  pause
  exit /b 1
)

echo 正在启动本地服务器（端口 8931）...
start "AttentionLabServer" /min %PY% -m http.server 8931 --bind 127.0.0.1
timeout /t 2 >nul
start "" "http://127.0.0.1:8931/index.html"

echo.
echo 实验页面已在浏览器中打开：http://127.0.0.1:8931/index.html
echo 服务器运行在最小化的窗口 "AttentionLabServer" 中，实验结束后关掉它即可。
echo 请勿用 file:// 方式直接双击 index.html —— 那样眼动无法加载。
echo.
pause
