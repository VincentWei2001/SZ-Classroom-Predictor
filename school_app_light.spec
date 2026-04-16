# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['C:\\Users\\GIGABYTE\\Desktop\\Test\\Analysis\\新（可用）\\CSV\\school_app_portable.py'],
    pathex=[],
    binaries=[],
    datas=[('C:\\Users\\GIGABYTE\\Desktop\\Test\\Analysis\\新（可用）\\CSV\\assets\\app_icon.ico', 'assets')],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='school_app_light',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    icon='C:\\Users\\GIGABYTE\\Desktop\\Test\\Analysis\\新（可用）\\CSV\\assets\\app_icon.ico',
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='school_app_light',
)
