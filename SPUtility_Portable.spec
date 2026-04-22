# -*- mode: python ; coding: utf-8 -*-
import os
import sys

from PyInstaller.utils.hooks import collect_data_files, collect_submodules

PROJECT_ROOT = os.path.abspath(globals().get('SPECPATH', os.getcwd()))
SRC_ROOT = os.path.join(PROJECT_ROOT, 'src')

if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

hiddenimports = []
hiddenimports += collect_submodules('sparams_utility')
datas = collect_data_files(
    'sparams_utility',
    includes=[
        'resources/*.svg',
        'resources/*.png',
        'resources/help/*.html',
        'resources/help/images/*',
    ],
) + [
    (os.path.join(PROJECT_ROOT, 'Images', 'Splash_Screen.png'), 'Images'),
    (os.path.join(PROJECT_ROOT, 'Images', 'Icon.png'), 'Images'),
]


a = Analysis(
    ['app.py'],
    pathex=[SRC_ROOT],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
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
    name='SPUtility_Portable',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    icon=os.path.join(PROJECT_ROOT, 'Images', 'Icon.png'),
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
    name='SPUtility_Portable',
)
