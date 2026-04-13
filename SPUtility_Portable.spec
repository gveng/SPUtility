# -*- mode: python ; coding: utf-8 -*-
import os

from PyInstaller.utils.hooks import collect_submodules

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))

hiddenimports = []
hiddenimports += collect_submodules('sparams_utility')
datas = [
    (os.path.join(PROJECT_ROOT, 'src', 'sparams_utility', 'resources', 'help', 'help_en.html'), 'sparams_utility/resources/help'),
    (os.path.join(PROJECT_ROOT, 'Images', 'Splash_Screen.png'), 'Images'),
    (os.path.join(PROJECT_ROOT, 'Images', 'Icon.png'), 'Images'),
]


a = Analysis(
    ['app.py'],
    pathex=['src'],
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
