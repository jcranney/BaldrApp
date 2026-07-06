#!/usr/bin/env python3

import json
import time
from pathlib import Path

import zmq
from PyQt5 import QtCore, QtGui, QtWidgets


# ============================================================
# Defaults
# ============================================================

CONTROL_ADDR = "tcp://127.0.0.1:6670"
MDS_ADDR = "tcp://127.0.0.1:5555"

DEFAULT_MASK = "J3"
BEAMS = [1, 2, 3, 4]

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_LOG_PATH = SCRIPT_DIR / ".logs" / "baldr_sim.log"


# ============================================================
# ZMQ worker
# ============================================================

class ZmqCommandWorker(QtCore.QThread):
    finished_signal = QtCore.pyqtSignal(str, str, bool)

    def __init__(self, addr, command, timeout_ms=30000, parent=None):
        super().__init__(parent)
        self.addr = addr
        self.command = command
        self.timeout_ms = int(timeout_ms)

    def run(self):
        ctx = zmq.Context.instance()
        sock = ctx.socket(zmq.REQ)
        sock.setsockopt(zmq.LINGER, 0)
        sock.setsockopt(zmq.RCVTIMEO, self.timeout_ms)
        sock.setsockopt(zmq.SNDTIMEO, self.timeout_ms)

        try:
            sock.connect(self.addr)
            t0 = time.time()
            sock.send_string(self.command)
            reply = sock.recv_string()
            dt = time.time() - t0
            self.finished_signal.emit(
                self.command,
                f"{reply}\n(reply time: {dt:.2f} s)",
                True,
            )
        except Exception as exc:
            self.finished_signal.emit(
                self.command,
                f"{type(exc).__name__}: {exc}",
                False,
            )
        finally:
            sock.close(0)


# ============================================================
# Main GUI
# ============================================================

class BaldrSimControlGui(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Baldr simulator control")
        self.resize(1150, 800)

        self.workers = []
        self.last_log_text = ""

        self._build_ui()
        self._connect_signals()

        self.log_timer = QtCore.QTimer(self)
        self.log_timer.timeout.connect(self.refresh_log)
        self.log_timer.start(500)

        self.status_timer = QtCore.QTimer(self)
        self.status_timer.timeout.connect(self.auto_status)
        self.status_timer.start(5000)

        self.refresh_log()

    # ------------------------------------------------------------
    # UI
    # ------------------------------------------------------------

    def _build_ui(self):
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)

        main_layout = QtWidgets.QVBoxLayout(central)

        # ---------------- paths / status ----------------
        top_group = QtWidgets.QGroupBox("Simulator connection")
        top_layout = QtWidgets.QGridLayout(top_group)

        self.control_addr_edit = QtWidgets.QLineEdit(CONTROL_ADDR)
        self.mds_addr_edit = QtWidgets.QLineEdit(MDS_ADDR)
        self.log_path_edit = QtWidgets.QLineEdit(str(DEFAULT_LOG_PATH))

        self.status_button = QtWidgets.QPushButton("Refresh status")
        self.auto_status_check = QtWidgets.QCheckBox("Auto status")
        self.auto_status_check.setChecked(True)

        top_layout.addWidget(QtWidgets.QLabel("Control ZMQ:"), 0, 0)
        top_layout.addWidget(self.control_addr_edit, 0, 1)
        top_layout.addWidget(QtWidgets.QLabel("MDS ZMQ:"), 1, 0)
        top_layout.addWidget(self.mds_addr_edit, 1, 1)
        top_layout.addWidget(QtWidgets.QLabel("Log file:"), 2, 0)
        top_layout.addWidget(self.log_path_edit, 2, 1)
        top_layout.addWidget(self.status_button, 0, 2)
        top_layout.addWidget(self.auto_status_check, 1, 2)

        main_layout.addWidget(top_group)

        # ---------------- controls row ----------------
        controls_layout = QtWidgets.QHBoxLayout()

        # Mode controls.
        mode_group = QtWidgets.QGroupBox("Source / atmosphere mode")
        mode_layout = QtWidgets.QGridLayout(mode_group)

        self.preset_onsky_button = QtWidgets.QPushButton("Preset: on-sky")
        self.preset_internal_button = QtWidgets.QPushButton("Preset: internal")
        self.phase_on_button = QtWidgets.QPushButton("Atmos phase ON")
        self.phase_off_button = QtWidgets.QPushButton("Atmos phase OFF")
        self.ao_on_button = QtWidgets.QPushButton("First-stage AO ON")
        self.ao_off_button = QtWidgets.QPushButton("First-stage AO OFF")
        self.scint_on_button = QtWidgets.QPushButton("Scintillation ON")
        self.scint_off_button = QtWidgets.QPushButton("Scintillation OFF")

        mode_layout.addWidget(self.preset_onsky_button, 0, 0)
        mode_layout.addWidget(self.preset_internal_button, 0, 1)
        mode_layout.addWidget(self.phase_on_button, 1, 0)
        mode_layout.addWidget(self.phase_off_button, 1, 1)
        mode_layout.addWidget(self.ao_on_button, 2, 0)
        mode_layout.addWidget(self.ao_off_button, 2, 1)
        mode_layout.addWidget(self.scint_on_button, 3, 0)
        mode_layout.addWidget(self.scint_off_button, 3, 1)

        controls_layout.addWidget(mode_group)

        # Phase-mask controls.
        fpm_group = QtWidgets.QGroupBox("Phase mask / fake MDS")
        fpm_layout = QtWidgets.QGridLayout(fpm_group)

        self.mask_combo = QtWidgets.QComboBox()
        self.mask_combo.addItems(["J1", "J2", "J3", "J4", "J5", "H1", "H2", "H3", "H4", "H5"])
        self.mask_combo.setCurrentText(DEFAULT_MASK)

        self.beam_combo = QtWidgets.QComboBox()
        self.beam_combo.addItems([str(b) for b in BEAMS])

        self.fpm_in_button = QtWidgets.QPushButton("Mask IN selected")
        self.fpm_out_button = QtWidgets.QPushButton("Mask OUT selected")
        self.fpm_in_all_button = QtWidgets.QPushButton("Mask IN all")
        self.fpm_out_all_button = QtWidgets.QPushButton("Mask OUT all")
        self.fpm_where_button = QtWidgets.QPushButton("Where is mask?")

        fpm_layout.addWidget(QtWidgets.QLabel("Beam:"), 0, 0)
        fpm_layout.addWidget(self.beam_combo, 0, 1)
        fpm_layout.addWidget(QtWidgets.QLabel("Mask:"), 1, 0)
        fpm_layout.addWidget(self.mask_combo, 1, 1)
        fpm_layout.addWidget(self.fpm_in_button, 2, 0)
        fpm_layout.addWidget(self.fpm_out_button, 2, 1)
        fpm_layout.addWidget(self.fpm_in_all_button, 3, 0)
        fpm_layout.addWidget(self.fpm_out_all_button, 3, 1)
        fpm_layout.addWidget(self.fpm_where_button, 4, 0, 1, 2)

        controls_layout.addWidget(fpm_group)

        # Fresnel offset controls.
        offset_group = QtWidgets.QGroupBox("Fresnel runtime offsets")
        offset_layout = QtWidgets.QGridLayout(offset_group)

        self.edge_offset_spin = QtWidgets.QDoubleSpinBox()
        self.edge_offset_spin.setRange(-20.0, 20.0)
        self.edge_offset_spin.setDecimals(4)
        self.edge_offset_spin.setSingleStep(0.1)
        self.edge_offset_spin.setSuffix(" mm")

        self.coldstop_x_spin = QtWidgets.QDoubleSpinBox()
        self.coldstop_x_spin.setRange(-1000.0, 1000.0)
        self.coldstop_x_spin.setDecimals(2)
        self.coldstop_x_spin.setSingleStep(5.0)
        self.coldstop_x_spin.setSuffix(" um")

        self.coldstop_y_spin = QtWidgets.QDoubleSpinBox()
        self.coldstop_y_spin.setRange(-1000.0, 1000.0)
        self.coldstop_y_spin.setDecimals(2)
        self.coldstop_y_spin.setSingleStep(5.0)
        self.coldstop_y_spin.setSuffix(" um")

        self.apply_edge_button = QtWidgets.QPushButton("Apply edge offset")
        self.apply_coldstop_button = QtWidgets.QPushButton("Apply cold-stop offset")
        self.reset_offsets_button = QtWidgets.QPushButton("Reset offsets")

        # offset_layout.addWidget(QtWidgets.QLabel("D mirror edge:"), 0, 0)
        # offset_layout.addWidget(self.edge_offset_spin, 0, 1)
        # offset_layout.addWidget(self.apply_edge_button, 0, 2)

        # offset_layout.addWidget(QtWidgets.QLabel("Cold stop X:"), 1, 0)
        # offset_layout.addWidget(self.coldstop_x_spin, 1, 1)
        # offset_layout.addWidget(QtWidgets.QLabel("Cold stop Y:"), 2, 0)
        # offset_layout.addWidget(self.coldstop_y_spin, 2, 1)
        # offset_layout.addWidget(self.apply_coldstop_button, 1, 2, 2, 1)

        # pupil mis-conjugation 
        self.pupil_misconj_spin = QtWidgets.QDoubleSpinBox()
        self.pupil_misconj_spin.setRange(-500.0, 500.0)
        self.pupil_misconj_spin.setDecimals(3)
        self.pupil_misconj_spin.setSingleStep(1.0)
        self.pupil_misconj_spin.setSuffix(" mm")

        self.apply_pupil_misconj_button = QtWidgets.QPushButton("Apply pupil misconj.")

        offset_layout.addWidget(QtWidgets.QLabel("D mirror edge:"), 0, 0)
        offset_layout.addWidget(self.edge_offset_spin, 0, 1)
        offset_layout.addWidget(self.apply_edge_button, 0, 2)

        offset_layout.addWidget(QtWidgets.QLabel("Cold stop X:"), 1, 0)
        offset_layout.addWidget(self.coldstop_x_spin, 1, 1)
        offset_layout.addWidget(QtWidgets.QLabel("Cold stop Y:"), 2, 0)
        offset_layout.addWidget(self.coldstop_y_spin, 2, 1)
        offset_layout.addWidget(self.apply_coldstop_button, 1, 2, 2, 1)

        offset_layout.addWidget(QtWidgets.QLabel("Pupil misconj.:"), 3, 0)
        offset_layout.addWidget(self.pupil_misconj_spin, 3, 1)
        offset_layout.addWidget(self.apply_pupil_misconj_button, 3, 2)

        offset_layout.addWidget(self.reset_offsets_button, 4, 0, 1, 3)

        # offset_layout.addWidget(QtWidgets.QLabel("D mirror edge:"), 0, 0)
        # offset_layout.addWidget(self.edge_offset_spin, 0, 1)
        # offset_layout.addWidget(self.apply_edge_button, 0, 2)

        # offset_layout.addWidget(QtWidgets.QLabel("Cold stop X:"), 1, 0)
        # offset_layout.addWidget(self.coldstop_x_spin, 1, 1)
        # offset_layout.addWidget(QtWidgets.QLabel("Cold stop Y:"), 2, 0)
        # offset_layout.addWidget(self.coldstop_y_spin, 2, 1)
        # offset_layout.addWidget(self.apply_coldstop_button, 1, 2, 2, 1)

        # offset_layout.addWidget(QtWidgets.QLabel("Pupil misconj.:"), 3, 0)
        # offset_layout.addWidget(self.pupil_misconj_spin, 3, 1)
        # offset_layout.addWidget(self.apply_pupil_misconj_button, 3, 2)

        # offset_layout.addWidget(self.reset_offsets_button, 4, 0, 1, 3)


        # offset_layout.addWidget(self.reset_offsets_button, 3, 0, 1, 3)

        controls_layout.addWidget(offset_group)

        main_layout.addLayout(controls_layout)

        # ---------------- status + command output + log ----------------
        split = QtWidgets.QSplitter(QtCore.Qt.Vertical)

        self.status_text = QtWidgets.QPlainTextEdit()
        self.status_text.setReadOnly(True)
        self.status_text.setMaximumBlockCount(500)

        self.command_text = QtWidgets.QPlainTextEdit()
        self.command_text.setReadOnly(True)
        self.command_text.setMaximumBlockCount(1000)

        self.log_text = QtWidgets.QPlainTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumBlockCount(3000)

        font = QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.FixedFont)
        self.status_text.setFont(font)
        self.command_text.setFont(font)
        self.log_text.setFont(font)

        split.addWidget(self.status_text)
        split.addWidget(self.command_text)
        split.addWidget(self.log_text)
        split.setSizes([180, 180, 440])

        main_layout.addWidget(split)

    def _connect_signals(self):
        self.status_button.clicked.connect(lambda: self.send_control("status"))

        self.preset_onsky_button.clicked.connect(lambda: self.send_control("preset onsky"))
        self.preset_internal_button.clicked.connect(lambda: self.send_control("preset internal"))

        self.phase_on_button.clicked.connect(lambda: self.send_control("phase on"))
        self.phase_off_button.clicked.connect(lambda: self.send_control("phase off"))
        self.ao_on_button.clicked.connect(lambda: self.send_control("ao on"))
        self.ao_off_button.clicked.connect(lambda: self.send_control("ao off"))
        self.scint_on_button.clicked.connect(lambda: self.send_control("scint on"))
        self.scint_off_button.clicked.connect(lambda: self.send_control("scint off"))

        self.apply_edge_button.clicked.connect(self.apply_edge_offset)
        self.apply_coldstop_button.clicked.connect(self.apply_coldstop_offset)
        self.reset_offsets_button.clicked.connect(lambda: self.send_control("reset_offsets"))
        self.apply_pupil_misconj_button.clicked.connect(
            self.apply_pupil_misconjugation
        )
        self.fpm_in_button.clicked.connect(self.fpm_in_selected)
        self.fpm_out_button.clicked.connect(self.fpm_out_selected)
        self.fpm_in_all_button.clicked.connect(self.fpm_in_all)
        self.fpm_out_all_button.clicked.connect(self.fpm_out_all)
        self.fpm_where_button.clicked.connect(self.fpm_where_selected)

    # ------------------------------------------------------------
    # Log/status
    # ------------------------------------------------------------

    def refresh_log(self):
        path = Path(self.log_path_edit.text()).expanduser()

        if not path.exists():
            text = f"Log file not found:\n{path}"
        else:
            try:
                max_bytes = 120_000
                with open(path, "rb") as f:
                    f.seek(0, 2)
                    size = f.tell()
                    f.seek(max(0, size - max_bytes), 0)
                    text = f.read().decode(errors="replace")
            except Exception as exc:
                text = f"Could not read log:\n{exc}"

        if text != self.last_log_text:
            self.last_log_text = text
            self.log_text.setPlainText(text)
            self.log_text.moveCursor(QtGui.QTextCursor.End)

    def auto_status(self):
        if self.auto_status_check.isChecked():
            self.send_control("status", quiet=True)

    # ------------------------------------------------------------
    # ZMQ send helpers
    # ------------------------------------------------------------

    def send_control(self, command, quiet=False):
        self.send_zmq(
            addr=self.control_addr_edit.text().strip(),
            command=command,
            target="control",
            quiet=quiet,
        )

    def send_mds(self, command):
        self.send_zmq(
            addr=self.mds_addr_edit.text().strip(),
            command=command,
            target="mds",
            quiet=False,
        )

    def send_zmq(self, addr, command, target, quiet=False):
        if not quiet:
            self.append_command(f">>> [{target}] {command}")

        worker = ZmqCommandWorker(addr=addr, command=command, timeout_ms=30000)
        worker.finished_signal.connect(
            lambda cmd, reply, ok, target=target, quiet=quiet:
            self.handle_command_reply(cmd, reply, ok, target, quiet)
        )
        worker.finished.connect(lambda: self.cleanup_worker(worker))

        self.workers.append(worker)
        worker.start()

    def handle_command_reply(self, command, reply, ok, target, quiet):
        prefix = "OK" if ok else "ERR"

        if target == "control" and command == "status" and ok:
            pretty = self.pretty_json(reply)
            self.status_text.setPlainText(pretty)
            self.status_text.moveCursor(QtGui.QTextCursor.End)

        if not quiet:
            self.append_command(f"[{prefix}] {reply}")

    def cleanup_worker(self, worker):
        try:
            self.workers.remove(worker)
        except ValueError:
            pass
        worker.deleteLater()

    def append_command(self, text):
        self.command_text.appendPlainText(text)
        self.command_text.moveCursor(QtGui.QTextCursor.End)

    @staticmethod
    def pretty_json(text):
        # Strip optional reply-time line.
        raw = text.split("\n(reply time:")[0]
        try:
            return json.dumps(json.loads(raw), indent=2, sort_keys=True)
        except Exception:
            return text

    # ------------------------------------------------------------
    # Control callbacks
    # ------------------------------------------------------------

    def apply_edge_offset(self):
        value_mm = self.edge_offset_spin.value()
        self.send_control(f"set edge_offset_mm {value_mm:.6f}")

    def apply_coldstop_offset(self):
        x_um = self.coldstop_x_spin.value()
        y_um = self.coldstop_y_spin.value()
        self.send_control(f"set coldstop_offset_um {x_um:.6f} {y_um:.6f}")

    def apply_pupil_misconjugation(self):
        value_mm = self.pupil_misconj_spin.value()
        self.send_control(f"set pupil_misconjugation_mm {value_mm:.6f}")
        
    def selected_beam(self):
        return int(self.beam_combo.currentText())

    def selected_mask(self):
        return self.mask_combo.currentText()

    def fpm_in_selected(self):
        beam = self.selected_beam()
        mask = self.selected_mask()
        self.send_mds(f"fpm_move {beam} {mask}")

    def fpm_out_selected(self):
        beam = self.selected_beam()

        # Requires fake_asgard_ZMQ_CRED1_server.py to support this.
        # If it returns NOK, add fpm_out handling to the fake MDS server next.
        #self.send_mds(f"fpm_out {beam}")
        self.send_mds(f"moverel BMX{beam} 10") # just does a small offset, the MDS registers that this is off the mask so changes state to phasemask "out"
    def fpm_in_all(self):
        mask = self.selected_mask()
        for beam in BEAMS:
            self.send_mds(f"fpm_move {beam} {mask}")

    def fpm_out_all(self):
        for beam in BEAMS:
            #self.send_mds(f"fpm_out {beam}")
            self.send_mds(f"moverel BMX{beam} 10") # just does a small offset, the MDS registers that this is off the mask so changes state to phasemask "out"

    def fpm_where_selected(self):
        beam = self.selected_beam()
        self.send_mds(f"fpm_whereami {beam}")


# ============================================================
# Entrypoint
# ============================================================

def main():
    app = QtWidgets.QApplication([])
    win = BaldrSimControlGui()
    win.show()
    app.exec_()


if __name__ == "__main__":
    main()
