from __future__ import annotations

from typing import TYPE_CHECKING

from PySide6.QtCore import QEvent, QObject, QPoint, Qt, QTimer, Signal
from PySide6.QtGui import QAction, QIcon
from PySide6.QtWidgets import QApplication, QMainWindow, QMenu, QTabWidget, QWidget


# ── Windows taskbar grouping helpers ────────────────────────────────────
# By default Windows groups taskbar entries by AppUserModelID. Setting a
# *unique* ID on each top-level window forces the OS to display them as
# separate, ungrouped buttons (each with its own icon).
def _set_window_app_user_model_id(window: QWidget, app_id: str) -> None:
    if not __import__("sys").platform.startswith("win"):
        return
    try:
        import ctypes
        from ctypes import wintypes

        hwnd = int(window.winId())
        if hwnd == 0:
            return

        # GUIDs / structures
        class GUID(ctypes.Structure):
            _fields_ = [("Data1", wintypes.DWORD),
                        ("Data2", wintypes.WORD),
                        ("Data3", wintypes.WORD),
                        ("Data4", ctypes.c_ubyte * 8)]

        class PROPERTYKEY(ctypes.Structure):
            _fields_ = [("fmtid", GUID), ("pid", wintypes.DWORD)]

        # PKEY_AppUserModel_ID = {9F4C2855-9F79-4B39-A8D0-E1D42DE1D5F3}, pid=5
        PKEY_AppUserModel_ID = PROPERTYKEY(
            GUID(0x9F4C2855, 0x9F79, 0x4B39,
                 (ctypes.c_ubyte * 8)(0xA8, 0xD0, 0xE1, 0xD4, 0x2D, 0xE1, 0xD5, 0xF3)),
            5,
        )

        # IID_IPropertyStore = {886D8EEB-8CF2-4446-8D02-CDBA1DBDCF99}
        IID_IPropertyStore = GUID(0x886D8EEB, 0x8CF2, 0x4446,
                                  (ctypes.c_ubyte * 8)(0x8D, 0x02, 0xCD, 0xBA, 0x1D, 0xBD, 0xCF, 0x99))

        # PROPVARIANT - we only need the LPWSTR (VT_LPWSTR=31) variant
        class PROPVARIANT(ctypes.Structure):
            _fields_ = [("vt", wintypes.USHORT),
                        ("wReserved1", wintypes.WORD),
                        ("wReserved2", wintypes.WORD),
                        ("wReserved3", wintypes.WORD),
                        ("pwszVal", wintypes.LPWSTR),
                        ("padding", ctypes.c_byte * 8)]

        SHGetPropertyStoreForWindow = ctypes.windll.shell32.SHGetPropertyStoreForWindow
        SHGetPropertyStoreForWindow.argtypes = [
            wintypes.HWND, ctypes.POINTER(GUID), ctypes.POINTER(ctypes.c_void_p),
        ]
        SHGetPropertyStoreForWindow.restype = ctypes.HRESULT

        store = ctypes.c_void_p()
        hr = SHGetPropertyStoreForWindow(hwnd, ctypes.byref(IID_IPropertyStore), ctypes.byref(store))
        if hr != 0 or not store.value:
            return
        try:
            # IPropertyStore vtable: SetValue is at slot 6 (0:QI 1:AddRef 2:Release 3:GetCount 4:GetAt 5:GetValue 6:SetValue 7:Commit)
            vtbl = ctypes.cast(store, ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p))).contents
            SetValue = ctypes.WINFUNCTYPE(
                ctypes.HRESULT, ctypes.c_void_p,
                ctypes.POINTER(PROPERTYKEY), ctypes.POINTER(PROPVARIANT),
            )(vtbl[6])
            Commit = ctypes.WINFUNCTYPE(ctypes.HRESULT, ctypes.c_void_p)(vtbl[7])
            Release = ctypes.WINFUNCTYPE(ctypes.c_ulong, ctypes.c_void_p)(vtbl[2])

            pv = PROPVARIANT()
            pv.vt = 31  # VT_LPWSTR
            pv.pwszVal = ctypes.c_wchar_p(app_id)
            SetValue(store, ctypes.byref(PKEY_AppUserModel_ID), ctypes.byref(pv))
            Commit(store)
        finally:
            try:
                Release(store)
            except Exception:
                pass
    except Exception:
        pass


# Reference aspect ratio (W/H) taken from the layout shown in the user-provided
# circuit screenshot (~1430x745). Used to size newly-opened child windows.
_REFERENCE_ASPECT = 1430.0 / 745.0


def default_child_window_size(category: str) -> tuple[int, int]:
    """Return (width, height) for a freshly opened child window.

    - circuits: ~3/4 of the available screen area
    - everything else (plots, tdr, eye, transient, tables, misc): ~1/5
    Both sized to the reference aspect ratio.
    """
    screen = QApplication.primaryScreen()
    if screen is None:
        return (1280, 760) if category == "circuits" else (640, 380)
    geom = screen.availableGeometry()
    sw, sh = geom.width(), geom.height()
    frac = 0.75 if category == "circuits" else 0.20
    area = frac * sw * sh
    h = int((area / _REFERENCE_ASPECT) ** 0.5)
    w = int(h * _REFERENCE_ASPECT)
    w = max(480, min(w, sw))
    h = max(300, min(h, sh))
    return (w, h)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from sparams_utility.ui.circuit_window import CircuitWindow
    from sparams_utility.ui.eye_diagram_window import EyeDiagramWindow
    from sparams_utility.ui.plot_window import PlotWindow
    from sparams_utility.ui.table_window import TableWindow
    from sparams_utility.ui.tdr_window import TdrWindow
    from sparams_utility.ui.transient_window import TransientResultWindow


class CategoryWindow(QMainWindow):
    """Top-level window grouping children of a single category by tabs."""

    tab_activated = Signal(QWidget)
    closing = Signal()
    detach_requested = Signal(QWidget)

    def __init__(self, category: str, title: str, parent: QWidget | None = None,
                 icon: QIcon | None = None):
        # Top-level window with NO parent so Windows shows it in the taskbar
        # as an independent application window.
        super().__init__(None, Qt.Window)
        self.category = category
        self._default_icon = icon
        self.setWindowTitle(title)
        if icon is not None and not icon.isNull():
            self.setWindowIcon(icon)
        self._tabs = QTabWidget()
        self._tabs.setTabsClosable(True)
        self._tabs.setMovable(True)
        self._tabs.tabCloseRequested.connect(self._on_tab_close_requested)
        self._tabs.currentChanged.connect(self._on_current_changed)
        self._tabs.tabBarDoubleClicked.connect(self._on_tab_double_clicked)
        bar = self._tabs.tabBar()
        bar.setContextMenuPolicy(Qt.CustomContextMenu)
        bar.customContextMenuRequested.connect(self._on_tab_context_menu)
        self.setCentralWidget(self._tabs)
        w, h = default_child_window_size(category)
        self.resize(w, h)

    def add_widget(self, widget: QWidget, tab_title: str) -> int:
        idx = self._tabs.addTab(widget, tab_title)
        self._tabs.setCurrentIndex(idx)
        if self.isMinimized():
            self.showNormal()
        elif not self.isVisible():
            self.show()
        # Apply unique AppUserModelID so Windows shows this category window as
        # its own taskbar button (ungrouped from other categories).
        _set_window_app_user_model_id(self, f"SParamsUtility.App.{self.category}")
        self.raise_()
        self.activateWindow()
        return idx

    def remove_widget(self, widget: QWidget) -> None:
        idx = self._tabs.indexOf(widget)
        if idx >= 0:
            self._tabs.removeTab(idx)
        if self._tabs.count() == 0:
            self.hide()

    def detach_widget(self, widget: QWidget) -> bool:
        """Remove widget from tabs without closing it."""
        idx = self._tabs.indexOf(widget)
        if idx < 0:
            return False
        self._tabs.removeTab(idx)
        if self._tabs.count() == 0:
            self.hide()
        return True

    def widgets(self) -> list[QWidget]:
        return [self._tabs.widget(i) for i in range(self._tabs.count())]

    def current_widget(self) -> QWidget | None:
        return self._tabs.currentWidget()

    def set_current_widget(self, widget: QWidget) -> None:
        idx = self._tabs.indexOf(widget)
        if idx >= 0:
            self._tabs.setCurrentIndex(idx)

    def update_tab_title(self, widget: QWidget, title: str) -> None:
        idx = self._tabs.indexOf(widget)
        if idx >= 0:
            self._tabs.setTabText(idx, title)

    def _on_tab_close_requested(self, idx: int) -> None:
        w = self._tabs.widget(idx)
        if w is not None:
            w.close()

    def _on_current_changed(self, idx: int) -> None:
        widget = self._tabs.widget(idx) if idx >= 0 else None
        # Reflect the active tab's icon on the taskbar entry.
        if widget is not None:
            wicon = widget.windowIcon()
            if wicon is not None and not wicon.isNull():
                self.setWindowIcon(wicon)
            elif self._default_icon is not None and not self._default_icon.isNull():
                self.setWindowIcon(self._default_icon)
        elif self._default_icon is not None and not self._default_icon.isNull():
            self.setWindowIcon(self._default_icon)
        self.tab_activated.emit(widget)

    def _on_tab_double_clicked(self, idx: int) -> None:
        if idx < 0:
            return
        w = self._tabs.widget(idx)
        if w is not None:
            self.detach_requested.emit(w)

    def _on_tab_context_menu(self, pos: QPoint) -> None:
        bar = self._tabs.tabBar()
        idx = bar.tabAt(pos)
        if idx < 0:
            return
        w = self._tabs.widget(idx)
        if w is None:
            return
        menu = QMenu(self)
        act_detach = QAction("Detach to separate window", menu)
        act_detach.triggered.connect(lambda: self.detach_requested.emit(w))
        menu.addAction(act_detach)
        act_close = QAction("Close tab", menu)
        act_close.triggered.connect(w.close)
        menu.addAction(act_close)
        menu.exec(bar.mapToGlobal(pos))

    def closeEvent(self, event):  # noqa: N802
        for w in list(self.widgets()):
            w.close()
        self.closing.emit()
        super().closeEvent(event)


def _resolve_category_for_class():
    # Lazy import to avoid circulars at module import time.
    from sparams_utility.ui.circuit_window import CircuitWindow
    from sparams_utility.ui.eye_diagram_window import EyeDiagramWindow
    from sparams_utility.ui.plot_window import PlotWindow
    from sparams_utility.ui.table_window import TableWindow
    from sparams_utility.ui.tdr_window import TdrWindow
    from sparams_utility.ui.transient_window import TransientResultWindow

    return {
        PlotWindow: "plots",
        TdrWindow: "tdr",
        CircuitWindow: "circuits",
        EyeDiagramWindow: "eye",
        TransientResultWindow: "transient",
        TableWindow: "tables",
    }


class ChildWindowManager(QObject):
    """Replacement for QMdiArea: routes widgets to per-category top-level windows."""

    widget_activated = Signal(object)  # QWidget or None
    widget_closed = Signal(object)     # QWidget

    CATEGORY_TITLES = {
        "plots": "S-Parameter Plots",
        "tdr": "TDR Plots",
        "circuits": "Circuits",
        "eye": "Eye Diagrams",
        "transient": "Transient Results",
        "tables": "Tables",
        "misc": "Other Windows",
    }

    def __init__(self, host: QWidget | None = None,
                 icon_for_category=None,
                 icon_for_widget=None):
        super().__init__(host)
        self._host = host
        self._categories: dict[str, CategoryWindow] = {}
        self._active_widget: QWidget | None = None
        self._category_for_class: dict[type, str] | None = None
        self._detached: dict[int, tuple[QWidget, str]] = {}
        self._icon_for_category = icon_for_category  # callable(category) -> QIcon
        self._icon_for_widget = icon_for_widget      # callable(widget)   -> QIcon

    @property
    def CATEGORY_FOR_CLASS(self) -> dict[type, str]:  # noqa: N802
        if self._category_for_class is None:
            self._category_for_class = _resolve_category_for_class()
        return self._category_for_class

    def category_for_widget(self, widget: QWidget) -> str:
        for cls, cat in self.CATEGORY_FOR_CLASS.items():
            if isinstance(widget, cls):
                return cat
        return "misc"

    def _ensure_category(self, category: str) -> CategoryWindow:
        win = self._categories.get(category)
        if win is None:
            cat_icon = None
            if self._icon_for_category is not None:
                try:
                    cat_icon = self._icon_for_category(category)
                except Exception:
                    cat_icon = None
            win = CategoryWindow(
                category,
                self.CATEGORY_TITLES.get(category, category.title()),
                self._host,
                icon=cat_icon,
            )
            win.tab_activated.connect(self._on_tab_activated)
            win.detach_requested.connect(self.detach)
            self._categories[category] = win
        return win

    def present(self, widget: QWidget, *, tab_title: str | None = None) -> QWidget:
        category = self.category_for_widget(widget)
        cat_win = self._ensure_category(category)
        title = tab_title or (widget.windowTitle() or "Untitled")
        widget.installEventFilter(self)
        # Stash the logical host so child widgets can locate the main window
        # even though the CategoryWindow is now a top-level (parent-less) window.
        try:
            widget._host_main_window = self._host  # type: ignore[attr-defined]
        except Exception:
            pass
        # Tag widget with its per-kind icon so detached windows / taskbar reflect it.
        if self._icon_for_widget is not None:
            try:
                wicon = self._icon_for_widget(widget)
                if wicon is not None and not wicon.isNull():
                    widget.setWindowIcon(wicon)
            except Exception:
                pass
        cat_win.add_widget(widget, title)
        return widget

    def eventFilter(self, obj, event):  # noqa: N802
        try:
            etype = event.type()
        except RuntimeError:
            return super().eventFilter(obj, event)
        if isinstance(obj, QWidget):
            if etype == QEvent.WindowTitleChange:
                cat = self._find_category_of(obj)
                if cat is not None:
                    self._categories[cat].update_tab_title(obj, obj.windowTitle())
            elif etype == QEvent.Close:
                self._on_widget_closing(obj)
        return super().eventFilter(obj, event)

    def _on_widget_closing(self, widget: QWidget) -> None:
        cat = self._find_category_of(widget)
        if cat is not None:
            self._categories[cat].remove_widget(widget)
        # Also drop from detached registry if applicable
        self._detached.pop(id(widget), None)
        if self._active_widget is widget:
            self._active_widget = None
            self.widget_activated.emit(None)
        self.widget_closed.emit(widget)

    # ── Detach / Reattach ─────────────────────────────────────────────

    def is_detached(self, widget: QWidget) -> bool:
        return id(widget) in self._detached

    def detached_widgets(self) -> list[QWidget]:
        return [w for (w, _c) in self._detached.values()]

    def detach(self, widget: QWidget) -> None:
        """Pop widget out of its CategoryWindow tab into a top-level window."""
        cat = self._find_category_of(widget)
        if cat is None:
            return
        cat_win = self._categories[cat]
        if not cat_win.detach_widget(widget):
            return
        widget.setParent(None)
        widget.setWindowFlags(Qt.Window)
        # Ensure detached window has its per-kind icon (taskbar entry).
        if self._icon_for_widget is not None:
            try:
                wicon = self._icon_for_widget(widget)
                if wicon is not None and not wicon.isNull():
                    widget.setWindowIcon(wicon)
            except Exception:
                pass
        # Give detached window a sensible default size from category preset
        try:
            w_d, h_d = default_child_window_size(cat)
            widget.resize(w_d, h_d)
        except Exception:
            pass
        widget.show()
        widget.raise_()
        widget.activateWindow()
        # Unique AppUserModelID so each detached window is its own taskbar button.
        _set_window_app_user_model_id(widget, f"SParamsUtility.App.detached.{id(widget)}")
        self._detached[id(widget)] = (widget, cat)
        self._active_widget = widget
        self.widget_activated.emit(widget)

    def reattach(self, widget: QWidget) -> None:
        """Return a previously detached widget to its CategoryWindow."""
        entry = self._detached.pop(id(widget), None)
        if entry is None:
            return
        _w, cat = entry
        cat_win = self._ensure_category(cat)
        widget.setWindowFlags(Qt.Widget)
        title = widget.windowTitle() or "Untitled"
        cat_win.add_widget(widget, title)
        self._active_widget = widget
        self.widget_activated.emit(widget)

    def reattach_all(self) -> None:
        for w in list(self.detached_widgets()):
            self.reattach(w)

    def _on_tab_activated(self, widget: QWidget | None) -> None:
        if widget is not None:
            self._active_widget = widget
        self.widget_activated.emit(widget)

    def _find_category_of(self, widget: QWidget) -> str | None:
        # Check detached first
        entry = self._detached.get(id(widget))
        if entry is not None:
            return entry[1]
        for cat, win in self._categories.items():
            try:
                if widget in win.widgets():
                    return cat
            except RuntimeError:
                continue
        return None

    def list_widgets(self, category: str | None = None) -> list[QWidget]:
        if category is not None:
            cw = self._categories.get(category)
            tabbed = list(cw.widgets()) if cw is not None else []
            detached = [w for (w, c) in self._detached.values() if c == category]
            return tabbed + detached
        result: list[QWidget] = []
        for cw in self._categories.values():
            result.extend(cw.widgets())
        result.extend(self.detached_widgets())
        return result

    def widgets_of_type(self, *types) -> list[QWidget]:
        return [w for w in self.list_widgets() if isinstance(w, types)]

    def active_widget(self) -> QWidget | None:
        return self._active_widget

    def set_active_widget(self, widget: QWidget | None) -> None:
        if widget is None:
            return
        # Detached: just raise the standalone window
        if self.is_detached(widget):
            widget.show()
            widget.raise_()
            widget.activateWindow()
            self._active_widget = widget
            self.widget_activated.emit(widget)
            return
        cat = self._find_category_of(widget)
        if cat is None:
            return
        cw = self._categories[cat]
        cw.set_current_widget(widget)
        if not cw.isVisible():
            cw.show()
        cw.raise_()
        cw.activateWindow()
        widget.raise_()
        widget.activateWindow()
        self._active_widget = widget
        self.widget_activated.emit(widget)

    def bring_to_front(self, widget: QWidget | None) -> None:
        """Force a result widget to the foreground after a simulation.

        Activates the widget immediately and re-activates it via deferred
        timers so the focus survives the closure of any modal progress
        dialog that may steal focus back to its parent. As a fallback (e.g.
        when the OS denies cross-window focus stealing) the corresponding
        top-level window taskbar entry is flashed via ``QApplication.alert``.
        """
        if widget is None:
            return
        try:
            self.set_active_widget(widget)
        except RuntimeError:
            return

        def _defer() -> None:
            try:
                self.set_active_widget(widget)
                if self.is_detached(widget):
                    target = widget
                else:
                    cat = self._find_category_of(widget)
                    target = self._categories.get(cat) if cat else None
                if target is not None:
                    if target.isMinimized():
                        target.showNormal()
                    target.raise_()
                    target.activateWindow()
                    QApplication.alert(target, 0)
            except RuntimeError:
                pass

        QTimer.singleShot(0, _defer)
        # Second deferred raise defeats late focus-snapback from progress
        # dialogs that may close just after the simulation completes.
        QTimer.singleShot(80, _defer)

    def close_all(self) -> None:
        for w in list(self.detached_widgets()):
            try:
                w.close()
            except RuntimeError:
                pass
        self._detached.clear()
        for cw in list(self._categories.values()):
            cw.close()
        self._categories.clear()
        self._active_widget = None
        self.widget_activated.emit(None)

    def minimize_all(self) -> None:
        for cw in self._categories.values():
            if cw.isVisible():
                cw.showMinimized()
        for w in self.detached_widgets():
            if w.isVisible():
                w.showMinimized()

    def restore_all(self) -> None:
        for cw in self._categories.values():
            cw.showNormal()
            cw.raise_()
        for w in self.detached_widgets():
            w.showNormal()
            w.raise_()

    def category_windows(self) -> dict[str, CategoryWindow]:
        return dict(self._categories)
