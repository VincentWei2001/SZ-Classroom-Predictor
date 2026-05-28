# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['C:\\Users\\GIGABYTE\\Desktop\\Test\\Analysis\\新（可用）\\机器学习的模型与开发的程序\\src\\school_app_portable.py'],
    pathex=['C:\\Users\\GIGABYTE\\Desktop\\Test\\Analysis\\新（可用）\\机器学习的模型与开发的程序\\src'],
    binaries=[],
    datas=[('C:\\Users\\GIGABYTE\\Desktop\\Test\\Analysis\\新（可用）\\机器学习的模型与开发的程序\\assets\\app_icon.ico', 'assets')],
    hiddenimports=['sklearn_joblib_compat', 'scenario_registry', 'sklearn.ensemble._forest', 'sklearn.tree._classes', 'sklearn.ensemble._base', 'PySide6.QtCore', 'PySide6.QtGui', 'PySide6.QtWidgets', 'matplotlib.backends.backend_qtagg'],
    hookspath=['C:\\Users\\GIGABYTE\\Desktop\\Test\\Analysis\\新（可用）\\机器学习的模型与开发的程序\\packaging'],
    hooksconfig={},
    runtime_hooks=['C:\\Users\\GIGABYTE\\Desktop\\Test\\Analysis\\新（可用）\\机器学习的模型与开发的程序\\packaging\\pyi_rth_00_qt_dll_path.py'],
    excludes=['PySide6.QtWebEngineCore', 'PySide6.QtWebEngineWidgets', 'PySide6.QtWebEngineQuick', 'PySide6.QtNetwork', 'PySide6.QtQml', 'PySide6.QtQuick', 'PySide6.QtOpenGL', 'PySide6.QtPdf', 'PySide6.QtSvg', 'PySide6.QtVirtualKeyboard'],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='school_app',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=['C:\\Users\\GIGABYTE\\Desktop\\Test\\Analysis\\新（可用）\\机器学习的模型与开发的程序\\assets\\app_icon.ico'],
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name='school_app',
)
