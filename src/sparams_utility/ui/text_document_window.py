"""Rich-text document editor window for project notes."""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QAction, QFont, QTextCharFormat, QTextBlockFormat, QKeySequence
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QFontComboBox,
    QMainWindow,
    QSpinBox,
    QTextEdit,
    QToolBar,
    QWidget,
)


class TextDocumentWindow(QMainWindow):
    """A simple rich-text editor tied to a project text-document entry.

    Signals
    -------
    content_changed(doc_id, html)
        Emitted whenever the editor content changes (debounced via document
        contentChanged). ``html`` is the full document HTML.
    """

    content_changed = Signal(str, str)  # doc_id, html

    def __init__(
        self,
        doc_id: str,
        title: str,
        html: str = "",
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._doc_id = doc_id
        self._suppress_change = False

        self.setWindowTitle(title)
        self.resize(720, 540)
        app = QApplication.instance()
        if app is not None:
            self.setWindowIcon(app.windowIcon())

        # ── Editor ────────────────────────────────────────────────────────
        self._editor = QTextEdit()
        self._editor.setAcceptRichText(True)
        self.setCentralWidget(self._editor)

        # ── Toolbar ───────────────────────────────────────────────────────
        tb = QToolBar("Formatting", self)
        tb.setMovable(False)
        self.addToolBar(Qt.TopToolBarArea, tb)

        # Font family
        self._font_box = QFontComboBox()
        self._font_box.setToolTip("Font family")
        self._font_box.setMaximumWidth(180)
        self._font_box.currentFontChanged.connect(self._on_font_family_changed)
        tb.addWidget(self._font_box)

        # Font size
        self._size_box = QSpinBox()
        self._size_box.setToolTip("Font size (pt)")
        self._size_box.setRange(6, 144)
        self._size_box.setValue(10)
        self._size_box.setMaximumWidth(56)
        self._size_box.valueChanged.connect(self._on_font_size_changed)
        tb.addWidget(self._size_box)

        tb.addSeparator()

        # Bold / Italic / Underline
        self._bold_action = QAction("B", self)
        self._bold_action.setCheckable(True)
        self._bold_action.setShortcut(QKeySequence("Ctrl+B"))
        self._bold_action.setToolTip("Bold (Ctrl+B)")
        font_bold = self._bold_action.font()
        font_bold.setBold(True)
        self._bold_action.setFont(font_bold)
        self._bold_action.triggered.connect(self._on_bold_toggled)
        tb.addAction(self._bold_action)

        self._italic_action = QAction("I", self)
        self._italic_action.setCheckable(True)
        self._italic_action.setShortcut(QKeySequence("Ctrl+I"))
        self._italic_action.setToolTip("Italic (Ctrl+I)")
        font_italic = self._italic_action.font()
        font_italic.setItalic(True)
        self._italic_action.setFont(font_italic)
        self._italic_action.triggered.connect(self._on_italic_toggled)
        tb.addAction(self._italic_action)

        self._underline_action = QAction("U", self)
        self._underline_action.setCheckable(True)
        self._underline_action.setShortcut(QKeySequence("Ctrl+U"))
        self._underline_action.setToolTip("Underline (Ctrl+U)")
        font_ul = self._underline_action.font()
        font_ul.setUnderline(True)
        self._underline_action.setFont(font_ul)
        self._underline_action.triggered.connect(self._on_underline_toggled)
        tb.addAction(self._underline_action)

        tb.addSeparator()

        # Alignment
        align_left = QAction("≡L", self)
        align_left.setToolTip("Align left")
        align_left.triggered.connect(lambda: self._set_alignment(Qt.AlignLeft))
        tb.addAction(align_left)

        align_center = QAction("≡C", self)
        align_center.setToolTip("Align center")
        align_center.triggered.connect(lambda: self._set_alignment(Qt.AlignHCenter))
        tb.addAction(align_center)

        align_right = QAction("≡R", self)
        align_right.setToolTip("Align right")
        align_right.triggered.connect(lambda: self._set_alignment(Qt.AlignRight))
        tb.addAction(align_right)

        align_justify = QAction("≡J", self)
        align_justify.setToolTip("Justify")
        align_justify.triggered.connect(lambda: self._set_alignment(Qt.AlignJustify))
        tb.addAction(align_justify)

        tb.addSeparator()

        # Text / background color pickers (simple named-colour dropdowns)
        self._text_color_box = QComboBox()
        self._text_color_box.setToolTip("Text colour")
        self._text_color_box.setMaximumWidth(96)
        for label in ("Black", "Dark grey", "Grey", "White", "Red", "Green", "Blue",
                      "Cyan", "Magenta", "Yellow", "Orange"):
            self._text_color_box.addItem(label)
        self._text_color_box.currentTextChanged.connect(self._on_text_color_changed)
        tb.addWidget(self._text_color_box)

        # ── Load initial content ──────────────────────────────────────────
        self._suppress_change = True
        if html.strip():
            self._editor.setHtml(html)
        self._suppress_change = False

        self._editor.document().contentsChanged.connect(self._on_contents_changed)
        self._editor.cursorPositionChanged.connect(self._sync_toolbar_state)
        self._sync_toolbar_state()

    # ── Public API ────────────────────────────────────────────────────────

    @property
    def doc_id(self) -> str:
        return self._doc_id

    def get_html(self) -> str:
        return self._editor.toHtml()

    def set_html(self, html: str) -> None:
        self._suppress_change = True
        self._editor.setHtml(html)
        self._suppress_change = False

    # ── Slots ─────────────────────────────────────────────────────────────

    def _on_contents_changed(self) -> None:
        if self._suppress_change:
            return
        self.content_changed.emit(self._doc_id, self._editor.toHtml())

    def _sync_toolbar_state(self) -> None:
        """Update toolbar checkboxes to reflect current cursor format."""
        fmt = self._editor.currentCharFormat()
        self._bold_action.setChecked(fmt.fontWeight() == QFont.Bold)
        self._italic_action.setChecked(fmt.fontItalic())
        self._underline_action.setChecked(fmt.fontUnderline())

        # Sync font box / size box without triggering edits
        family = fmt.fontFamilies()
        if isinstance(family, list) and family:
            family = family[0]
        if isinstance(family, str) and family:
            self._font_box.blockSignals(True)
            self._font_box.setCurrentFont(QFont(family))
            self._font_box.blockSignals(False)

        size = fmt.fontPointSize()
        if size > 0:
            self._size_box.blockSignals(True)
            self._size_box.setValue(int(size))
            self._size_box.blockSignals(False)

    def _on_font_family_changed(self, font: QFont) -> None:
        fmt = QTextCharFormat()
        fmt.setFontFamilies([font.family()])
        self._editor.mergeCurrentCharFormat(fmt)

    def _on_font_size_changed(self, size: int) -> None:
        fmt = QTextCharFormat()
        fmt.setFontPointSize(float(size))
        self._editor.mergeCurrentCharFormat(fmt)

    def _on_bold_toggled(self, checked: bool) -> None:
        fmt = QTextCharFormat()
        fmt.setFontWeight(QFont.Bold if checked else QFont.Normal)
        self._editor.mergeCurrentCharFormat(fmt)

    def _on_italic_toggled(self, checked: bool) -> None:
        fmt = QTextCharFormat()
        fmt.setFontItalic(checked)
        self._editor.mergeCurrentCharFormat(fmt)

    def _on_underline_toggled(self, checked: bool) -> None:
        fmt = QTextCharFormat()
        fmt.setFontUnderline(checked)
        self._editor.mergeCurrentCharFormat(fmt)

    def _set_alignment(self, alignment: Qt.AlignmentFlag) -> None:
        self._editor.setAlignment(alignment)

    _COLOR_MAP: dict[str, str] = {
        "Black": "#000000",
        "Dark grey": "#444444",
        "Grey": "#888888",
        "White": "#ffffff",
        "Red": "#cc0000",
        "Green": "#006600",
        "Blue": "#0000cc",
        "Cyan": "#007777",
        "Magenta": "#880088",
        "Yellow": "#aaaa00",
        "Orange": "#cc6600",
    }

    def _on_text_color_changed(self, label: str) -> None:
        hex_color = self._COLOR_MAP.get(label, "#000000")
        fmt = QTextCharFormat()
        from PySide6.QtGui import QColor as _QColor
        fmt.setForeground(_QColor(hex_color))
        self._editor.mergeCurrentCharFormat(fmt)
