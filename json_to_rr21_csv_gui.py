"""json_to_rr21_csv_gui.py

PyQt5 utility – combine OpenPose *body‑25* JSON files into a single CSV
in **RR‑21** order.

Run with:
    python json_to_rr21_csv_gui.py
"""

import csv
import json
import sys
from pathlib import Path

from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtWidgets import (
    QApplication,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QProgressBar,
    QVBoxLayout,
    QWidget,
)

# ---------------------------------------------------------------------------
#  Configuration
# ---------------------------------------------------------------------------

BODY25_NAMES = [
    "NOSE", "NECK", "RIGHT_SHOULDER", "RIGHT_ELBOW", "RIGHT_WRIST",
    "LEFT_SHOULDER", "LEFT_ELBOW", "LEFT_WRIST", "MID_HIP",
    "RIGHT_HIP", "RIGHT_KNEE", "RIGHT_ANKLE", "LEFT_HIP", "LEFT_KNEE",
    "LEFT_ANKLE", "RIGHT_EYE", "LEFT_EYE", "RIGHT_EAR", "LEFT_EAR",
    "LEFT_BIG_TOE", "LEFT_SMALL_TOE", "LEFT_HEEL", "RIGHT_BIG_TOE",
    "RIGHT_SMALL_TOE", "RIGHT_HEEL",
]

RR21_NAMES = [
    "NOSE", "LEFT_EYE", "RIGHT_EYE", "LEFT_EAR", "RIGHT_EAR",
    "LEFT_SHOULDER", "RIGHT_SHOULDER", "LEFT_ELBOW", "RIGHT_ELBOW",
    "LEFT_WRIST", "RIGHT_WRIST", "LEFT_HIP", "RIGHT_HIP",
    "LEFT_KNEE", "RIGHT_KNEE", "LEFT_ANKLE", "RIGHT_ANKLE",
    "LEFT_HEEL", "RIGHT_HEEL", "LEFT_FOOT", "RIGHT_FOOT",
]

# Map RR‑21 point → body‑25 equivalent (None → zero‑fill)
RR21_TO_BODY25_MAP = {
    "LEFT_FOOT": "LEFT_BIG_TOE",
    "RIGHT_FOOT": "RIGHT_BIG_TOE",
}

RR21_IDX: list[int] = []
for name in RR21_NAMES:
    mapped = RR21_TO_BODY25_MAP.get(name, name)
    try:
        RR21_IDX.append(BODY25_NAMES.index(mapped))
    except ValueError:
        RR21_IDX.append(-1)  # zero‑fill sentinel

# ---------------------------------------------------------------------------
#  Background worker
# ---------------------------------------------------------------------------

class ConverterWorker(QThread):
    progress = pyqtSignal(int)
    finished = pyqtSignal(bool, str)

    def __init__(self, json_root: Path, csv_path: Path):
        super().__init__()
        self.json_root = json_root
        self.csv_path = csv_path

    # ---------------------------------------------------------------------
    @staticmethod
    def _collect_json_files(root: Path):
        return sorted(root.rglob("*.json"))

    # ---------------------------------------------------------------------
    def run(self):
        try:
            json_paths = self._collect_json_files(self.json_root)
            if not json_paths:
                self.finished.emit(False, "No JSON files found.")
                return

            total = len(json_paths)
            with self.csv_path.open("w", newline="") as f:
                writer = csv.writer(f)

                # header: 42 entries (kp_x/kp_y)
                header = []
                for name in RR21_NAMES:
                    header.extend([f"{name}_x", f"{name}_y"])
                writer.writerow(header)

                for i, jp in enumerate(json_paths, start=1):
                    with jp.open() as jf:
                        data = json.load(jf)

                    if data.get("people"):
                        kps = data["people"][0]["pose_keypoints_2d"]
                        row = []
                        for idx in RR21_IDX:
                            if idx == -1:
                                row.extend([0, 0])
                            else:
                                base = idx * 3
                                row.extend(kps[base : base + 2])  # x, y
                    else:
                        row = [0] * (21 * 2)

                    writer.writerow(row)
                    self.progress.emit(int(i / total * 100))

            self.finished.emit(True, f"CSV written to: {self.csv_path}")
        except Exception as e:
            self.finished.emit(False, f"Error: {e}")

# ---------------------------------------------------------------------------
#  GUI
# ---------------------------------------------------------------------------

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("OpenPose → RR‑21 CSV Converter (42‑col)")
        self.setMinimumWidth(600)

        # Widgets
        self.json_edit = QLineEdit(); self.json_edit.setReadOnly(True)
        btn_json = QPushButton("Select JSON Folder …"); btn_json.clicked.connect(self.pick_json)

        self.csv_edit = QLineEdit(); self.csv_edit.setReadOnly(True)
        btn_csv = QPushButton("Select Output CSV …"); btn_csv.clicked.connect(self.pick_csv)

        self.btn_convert = QPushButton("Convert"); self.btn_convert.setEnabled(False); self.btn_convert.clicked.connect(self.convert)
        self.progress = QProgressBar(); self.status = QLabel("Ready.")

        # Layout
        layout = QVBoxLayout()
        layout.addWidget(QLabel("1· Base folder containing OpenPose JSONs:"))
        hl1 = QHBoxLayout(); hl1.addWidget(self.json_edit); hl1.addWidget(btn_json); layout.addLayout(hl1)
        layout.addWidget(QLabel("2· Save combined CSV as:"))
        hl2 = QHBoxLayout(); hl2.addWidget(self.csv_edit); hl2.addWidget(btn_csv); layout.addLayout(hl2)
        layout.addSpacing(20)
        layout.addWidget(self.btn_convert); layout.addWidget(self.progress); layout.addWidget(self.status)
        container = QWidget(); container.setLayout(layout); self.setCentralWidget(container)
        self.worker = None

    # ------------------------------------------------------------------
    def pick_json(self):
        folder = QFileDialog.getExistingDirectory(self, "Select base JSON folder")
        if folder:
            self.json_edit.setText(folder)
        self._update_state()

    def pick_csv(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save CSV", "pose_data.csv", "CSV (*.csv)")
        if path and not path.lower().endswith(".csv"):
            path += ".csv"
        if path:
            self.csv_edit.setText(path)
        self._update_state()

    def _update_state(self):
        self.btn_convert.setEnabled(bool(self.json_edit.text() and self.csv_edit.text()))

    def convert(self):
        json_root = Path(self.json_edit.text())
        csv_path = Path(self.csv_edit.text())
        if not json_root.exists():
            QMessageBox.warning(self, "Folder not found", "The selected JSON folder does not exist.")
            return
        try:
            csv_path.parent.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            QMessageBox.critical(self, "Cannot write file", str(e))
            return

        self.progress.setValue(0)
        self.status.setText("Processing …")
        self.btn_convert.setEnabled(False)

        self.worker = ConverterWorker(json_root, csv_path)
        self.worker.progress.connect(self.progress.setValue)
        self.worker.finished.connect(self.done)
        self.worker.start()

    def done(self, ok: bool, msg: str):
        self.btn_convert.setEnabled(True)
        self.status.setText(msg)
        (QMessageBox.information if ok else QMessageBox.critical)(self, "Result", msg)
        self.worker = None


# ---------------------------------------------------------------------------
#  Entry‑point
# ---------------------------------------------------------------------------

def main():
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
