#!/usr/bin/env python3

import argparse
import json
import mmap
import os
import shutil
import struct
import sys
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Deque, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
import pyqtgraph as pg


MAGIC = b"BIRDLOG1"
VERSION = 1
HEADER_SIZE = 128
DESCRIPTOR_SIZE = 96
DTYPE_FLOAT32 = 1
RAD2DEG = np.float32(180.0 / np.pi)
AXES = ("x", "y", "z")
SEGMENT_NAMES = ("RH", "RR", "RM", "LH", "LR", "LM")
AGGREGATE_NAMES = SEGMENT_NAMES + ("Body",)
PAPER_FONT = "Times New Roman"
INITIAL_JOINT_DEG = (
  6.87549, -5.72958, 11.25, -40.52, 10.8862, 29.64,
  6.87549, -5.72958, 11.25, -40.52, 10.8862, 29.64
)


@dataclass(frozen=True)
class ChannelDescriptor:
  name: str
  unit: str
  frame: str
  offset: int
  rows: int
  cols: int
  dtype: int

  @property
  def sample_shape(self) -> Tuple[int, ...]:
    if self.rows == 1 and self.cols == 1: return ()
    if self.cols == 1: return (self.rows,)
    return (self.rows, self.cols)


@dataclass(frozen=True)
class Header:
  version: int
  descriptor_count: int
  sample_size: int
  slot_size: int
  capacity: int
  log_hz: int
  nh: int
  nr: int
  nm: int
  strip_order: int
  start_time_ns: int
  schema_hash: int
  session_id: int
  sim_dt_ns: int
  log_dt_ns: int
  data_offset: int


class MMapReader:
  def __init__(self, path: str):
    self.path = path
    self.fd: Optional[int] = None
    self.mm: Optional[mmap.mmap] = None
    self.header: Optional[Header] = None
    self.descriptors: List[ChannelDescriptor] = []
    self._inode: Optional[int] = None
    self._size: Optional[int] = None

  def open(self) -> None:
    if self.mm is not None: return
    self.fd = os.open(self.path, os.O_RDONLY)
    stat = os.fstat(self.fd)
    self._inode = int(stat.st_ino)
    self._size = int(stat.st_size)
    self.mm = mmap.mmap(self.fd, stat.st_size, access=mmap.ACCESS_READ)
    try:
      self._parse_header()
    except Exception:
      self.close()
      raise

  def close(self) -> None:
    if self.mm is not None:
      self.mm.close()
      self.mm = None
    if self.fd is not None:
      os.close(self.fd)
      self.fd = None
    self.header = None
    self.descriptors = []
    self._inode = None
    self._size = None

  def changed_on_disk(self) -> bool:
    if self.mm is None: return False
    try: stat = os.stat(self.path)
    except OSError: return True
    return self._inode != int(stat.st_ino) or self._size != int(stat.st_size)

  def _parse_header(self) -> None:
    if self.mm is None or len(self.mm) < HEADER_SIZE: raise RuntimeError("mmap header is incomplete")
    if self.mm[0:8] != MAGIC: raise RuntimeError(f"bad mmap magic: {self.mm[0:8]!r}")

    values = struct.unpack_from("<13I", self.mm, 8)
    version, header_size, descriptor_size, descriptor_count, sample_size, slot_size, capacity, log_hz, nh, nr, nm, strip_order, _ = values
    qvalues = struct.unpack_from("<8Q", self.mm, 64)
    _, start_time_ns, schema_hash, session_id, sim_dt_ns, log_dt_ns, data_offset, _ = qvalues
    if version != VERSION: raise RuntimeError(f"unsupported mmap version: {version}")
    if header_size != HEADER_SIZE: raise RuntimeError(f"header size mismatch: {header_size}")
    if descriptor_size != DESCRIPTOR_SIZE: raise RuntimeError(f"descriptor size mismatch: {descriptor_size}")
    if data_offset + capacity*slot_size > len(self.mm): raise RuntimeError("mmap file is smaller than its declared layout")

    self.header = Header(version, descriptor_count, sample_size, slot_size, capacity, log_hz, nh, nr, nm, strip_order, start_time_ns, schema_hash, session_id, sim_dt_ns, log_dt_ns, data_offset)
    self.descriptors = []
    for i in range(descriptor_count):
      offset = HEADER_SIZE + i*DESCRIPTOR_SIZE
      name = bytes(self.mm[offset:offset+40]).split(b"\0", 1)[0].decode("utf-8")
      unit = bytes(self.mm[offset+40:offset+56]).split(b"\0", 1)[0].decode("utf-8")
      frame = bytes(self.mm[offset+56:offset+72]).split(b"\0", 1)[0].decode("utf-8")
      field_offset, rows, cols, dtype = struct.unpack_from("<IHHB", self.mm, offset+72)
      if dtype != DTYPE_FLOAT32: raise RuntimeError(f"unsupported dtype for {name}: {dtype}")
      if field_offset + 4*rows*cols > sample_size: raise RuntimeError(f"channel outside sample: {name}")
      self.descriptors.append(ChannelDescriptor(name, unit, frame, field_offset, rows, cols, dtype))

  def write_count(self) -> int:
    if self.mm is None: return 0
    return int(struct.unpack_from("<Q", self.mm, 64)[0])

  def session_id(self) -> int:
    if self.mm is None: return 0
    return int(struct.unpack_from("<Q", self.mm, 88)[0])

  def _read_slot(self, logical: int) -> Optional[bytes]:
    header = self.header
    if self.mm is None or header is None: return None
    slot_offset = header.data_offset + (logical % header.capacity)*header.slot_size
    for _ in range(12):
      seq_a = struct.unpack_from("<Q", self.mm, slot_offset)[0]
      if seq_a & 1: continue
      sample = bytes(self.mm[slot_offset+8:slot_offset+8+header.sample_size])
      seq_b = struct.unpack_from("<Q", self.mm, slot_offset)[0]
      if seq_a == seq_b and not (seq_b & 1): return sample
    return None

  def read_range(self, wc_from: int, wc_to: int) -> Tuple[np.ndarray, Dict[str, np.ndarray], int, int]:
    header = self.header
    if header is None: return np.empty((0,), np.float64), {}, 0, wc_from
    wc_from = int(wc_from)
    wc_to = int(wc_to)
    if wc_to <= wc_from: return np.empty((0,), np.float64), {}, 0, wc_from

    dropped = max(0, wc_to-wc_from-header.capacity)
    effective_from = max(wc_from, wc_to-header.capacity)
    count = wc_to-effective_from
    time = np.full((count,), np.nan, dtype=np.float64)
    channels: Dict[str, np.ndarray] = {
      "__step": np.zeros((count,), dtype=np.uint64),
      "__reset_epoch": np.zeros((count,), dtype=np.uint64),
      "__flags": np.zeros((count,), dtype=np.uint32)
    }
    for descriptor in self.descriptors:
      channels[descriptor.name] = np.full((count,) + descriptor.sample_shape, np.nan, dtype=np.float32)

    valid = np.zeros((count,), dtype=bool)
    for out_index, logical in enumerate(range(effective_from, wc_to)):
      sample = self._read_slot(logical)
      if sample is None: continue
      time[out_index] = struct.unpack_from("<d", sample, 0)[0]
      channels["__step"][out_index] = struct.unpack_from("<Q", sample, 8)[0]
      channels["__reset_epoch"][out_index] = struct.unpack_from("<Q", sample, 16)[0]
      channels["__flags"][out_index] = struct.unpack_from("<I", sample, 24)[0]
      for descriptor in self.descriptors:
        values = np.frombuffer(sample, dtype="<f4", count=descriptor.rows*descriptor.cols, offset=descriptor.offset)
        channels[descriptor.name][out_index] = values.reshape(descriptor.sample_shape) if descriptor.sample_shape else values[0]
      valid[out_index] = True

    if not np.all(valid):
      time = time[valid]
      channels = {name: values[valid] for name, values in channels.items()}
    return time, channels, dropped, effective_from

  def read_all(self) -> Tuple[np.ndarray, Dict[str, np.ndarray], int]:
    header = self.header
    if header is None: return np.empty((0,), np.float64), {}, 0
    wc = self.write_count()
    time, channels, _, _ = self.read_range(max(0, wc-header.capacity), wc)
    return time, channels, wc


class HistoryBuffer:
  def __init__(self, capacity: int):
    self.capacity = int(capacity)
    self.chunks: Deque[Tuple[np.ndarray, Dict[str, np.ndarray]]] = deque()
    self.count = 0

  def clear(self) -> None:
    self.chunks.clear()
    self.count = 0

  def append(self, time: np.ndarray, channels: Dict[str, np.ndarray]) -> None:
    if time.size == 0: return
    self.chunks.append((time.copy(), {name: values.copy() for name, values in channels.items()}))
    self.count += int(time.size)
    while self.count > self.capacity and self.chunks:
      first_time, first_channels = self.chunks[0]
      excess = self.count-self.capacity
      if first_time.size <= excess:
        self.chunks.popleft()
        self.count -= int(first_time.size)
      else:
        self.chunks[0] = (first_time[excess:], {name: values[excess:] for name, values in first_channels.items()})
        self.count -= excess

  def get(self, names: Sequence[str]) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    if not self.chunks: return np.empty((0,), np.float64), {}
    time = np.concatenate([chunk[0] for chunk in self.chunks])
    channels = {name: np.concatenate([chunk[1][name] for chunk in self.chunks]) for name in names}
    return time, channels


class LogRecorder:
  def __init__(self, mmap_path: str, log_dir: Path, metadata: Dict):
    self.mmap_path = mmap_path
    self.log_dir = log_dir
    self.metadata = metadata
    self.times: List[np.ndarray] = []
    self.channel_chunks: Dict[str, List[np.ndarray]] = {}
    self.dropped = 0

  def append(self, time: np.ndarray, channels: Dict[str, np.ndarray], dropped: int = 0) -> None:
    if time.size == 0: return
    self.times.append(time.copy())
    for name, values in channels.items(): self.channel_chunks.setdefault(name, []).append(values.copy())
    self.dropped += int(dropped)

  def save(self) -> Optional[Path]:
    if not self.times: return None
    timestamp = datetime.now().strftime("%m%d_%H%M%S_%f")
    npz_dir = self.log_dir / "npz"
    mmap_dir = self.log_dir / "mmap"
    npz_dir.mkdir(parents=True, exist_ok=True)
    mmap_dir.mkdir(parents=True, exist_ok=True)
    output = npz_dir / f"{timestamp}.npz"
    payload = {"time": np.concatenate(self.times)}
    for name, chunks in self.channel_chunks.items(): payload[name] = np.concatenate(chunks)
    meta = dict(self.metadata)
    meta["dropped"] = self.dropped
    payload["__metadata__"] = np.asarray(json.dumps(meta, separators=(",", ":")))
    np.savez_compressed(output, **payload)
    try:
      if os.path.exists(self.mmap_path): shutil.copy2(self.mmap_path, mmap_dir / f"{timestamp}.mmap")
    except OSError:
      pass
    self.times.clear()
    self.channel_chunks.clear()
    self.dropped = 0
    return output


def descriptor_metadata(reader: MMapReader) -> Dict:
  header = reader.header
  return {
    "version": header.version,
    "schema_hash": header.schema_hash,
    "log_hz": header.log_hz,
    "nh": header.nh,
    "nr": header.nr,
    "nm": header.nm,
    "strip_order": header.strip_order,
    "channels": [descriptor.__dict__ for descriptor in reader.descriptors]
  }


def rotation_to_rpy(rotation: np.ndarray) -> np.ndarray:
  pitch = np.arcsin(np.clip(-rotation[:, 2, 0], -1.0, 1.0))
  roll = np.arctan2(rotation[:, 2, 1], rotation[:, 2, 2])
  yaw = np.unwrap(np.arctan2(rotation[:, 1, 0], rotation[:, 0, 0]))
  return np.column_stack((roll, pitch, yaw)).astype(np.float32)*RAD2DEG


def paper_text(text: str, size: int = 12) -> str:
  return f"<span style=\"font-family:'{PAPER_FONT}'; font-size:{size}pt; color:#000000;\">{text}</span>"


def channel_title(name: str, frame: str) -> str:
  text = name.replace("_", " ").replace(".", " &middot; ")
  return f"{text} [{frame}]" if frame else text


def set_common_y_scale(plots: Sequence[pg.PlotItem], series: Sequence[Sequence[np.ndarray]]) -> None:
  centers: List[float] = []
  half_spans: List[float] = []
  for axis_series in series:
    low = np.inf
    high = -np.inf
    for values in axis_series:
      finite = values[np.isfinite(values)]
      if finite.size == 0: continue
      low = min(low, float(np.min(finite)))
      high = max(high, float(np.max(finite)))
    if np.isfinite(low) and np.isfinite(high):
      centers.append(0.5*(low+high))
      half_spans.append(0.5*(high-low))
    else:
      centers.append(0.0)
      half_spans.append(0.0)

  common_half_span = max(half_spans)
  if common_half_span <= 0.0:
    common_half_span = max(1.0, 0.05*max(abs(center) for center in centers))
  else:
    common_half_span *= 1.05
  for plot, center in zip(plots, centers):
    plot.setYRange(center-common_half_span, center+common_half_span, padding=0.0)


def style_plot(plot: pg.PlotItem, title: str, unit: str = "", show_time_values: bool = True) -> None:
  plot.setTitle(paper_text(title))
  plot.showGrid(x=True, y=True, alpha=0.25)
  if unit: plot.setLabel("left", paper_text(unit, 10))
  tick_font = QtGui.QFont(PAPER_FONT, 10)
  plot.getAxis("bottom").setStyle(showValues=show_time_values, tickFont=tick_font)
  plot.getAxis("left").setStyle(tickFont=tick_font)
  plot.getAxis("bottom").enableAutoSIPrefix(False)
  plot.getAxis("left").enableAutoSIPrefix(False)
  plot.setClipToView(True)
  plot.setDownsampling(auto=True, mode="peak")


def configure_light_theme(app: QtWidgets.QApplication) -> None:
  app.setStyle("Fusion")
  font = QtGui.QFont(PAPER_FONT, 11)
  font.setStyleHint(QtGui.QFont.Times)
  app.setFont(font)
  app.setStyleSheet(f"* {{font-family: '{PAPER_FONT}';}}")
  palette = QtGui.QPalette()
  palette.setColor(QtGui.QPalette.Window, QtGui.QColor(255, 255, 255))
  palette.setColor(QtGui.QPalette.WindowText, QtGui.QColor(20, 20, 20))
  palette.setColor(QtGui.QPalette.Base, QtGui.QColor(255, 255, 255))
  palette.setColor(QtGui.QPalette.AlternateBase, QtGui.QColor(242, 242, 242))
  palette.setColor(QtGui.QPalette.ToolTipBase, QtGui.QColor(255, 255, 255))
  palette.setColor(QtGui.QPalette.ToolTipText, QtGui.QColor(20, 20, 20))
  palette.setColor(QtGui.QPalette.Text, QtGui.QColor(20, 20, 20))
  palette.setColor(QtGui.QPalette.Button, QtGui.QColor(242, 242, 242))
  palette.setColor(QtGui.QPalette.ButtonText, QtGui.QColor(20, 20, 20))
  palette.setColor(QtGui.QPalette.BrightText, QtGui.QColor(255, 255, 255))
  palette.setColor(QtGui.QPalette.Highlight, QtGui.QColor(30, 90, 210))
  palette.setColor(QtGui.QPalette.HighlightedText, QtGui.QColor(255, 255, 255))
  app.setPalette(palette)


class MonitorWindow(QtWidgets.QMainWindow):
  def __init__(self, mmap_path: str, update_ms: int, log_dir: Path, replay_path: Optional[str], record: bool):
    super().__init__()
    self.setWindowTitle("Bird Simulation Monitor")
    self.resize(1800, 1050)
    self.mmap_path = mmap_path
    self.update_ms = update_ms
    self.log_dir = log_dir
    self.replay_path = replay_path
    self.record_enabled = record and replay_path is None
    self.reader = MMapReader(mmap_path)
    self.history: Optional[HistoryBuffer] = None
    self.recorder: Optional[LogRecorder] = None
    self.descriptors: List[ChannelDescriptor] = []
    self.channel_names: List[str] = []
    self.header: Optional[Header] = None
    self.wc_last = 0
    self.session_epoch: Optional[int] = None
    self.last_time: Optional[float] = None
    self.dropped_total = 0
    self.replay_time = np.empty((0,), np.float64)
    self.replay_channels: Dict[str, np.ndarray] = {}
    self.curves: Dict[str, pg.PlotDataItem] = {}
    self.load_plots: Dict[Tuple[str, int], pg.PlotItem] = {}
    self.aggregate_plots: Dict[Tuple[str, int], pg.PlotItem] = {}
    self.strip_wing = 0
    self.strip_segment = 0
    self.strip_local = 0
    self.strip_controls: List[Tuple[QtWidgets.QComboBox, QtWidgets.QComboBox, QtWidgets.QSpinBox]] = []
    self.max_points = 4000
    self._build_ui()

    if replay_path is not None:
      self._load_replay(Path(replay_path))
    else:
      self.timer = QtCore.QTimer(self)
      self.timer.timeout.connect(self._on_timer)
      self.timer.start(update_ms)

  def _build_ui(self) -> None:
    pg.setConfigOptions(antialias=False, background="w", foreground="k")
    root = QtWidgets.QWidget()
    self.setCentralWidget(root)
    layout = QtWidgets.QVBoxLayout(root)

    top = QtWidgets.QHBoxLayout()
    self.path_label = QtWidgets.QLabel(self.replay_path or self.mmap_path)
    self.status_label = QtWidgets.QLabel("waiting...")
    self.status_label.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
    top.addWidget(self.path_label, 1)
    top.addWidget(self.status_label, 1)
    layout.addLayout(top)

    self.replay_controls = QtWidgets.QWidget()
    replay_layout = QtWidgets.QHBoxLayout(self.replay_controls)
    replay_layout.setContentsMargins(0, 0, 0, 0)
    self.range_from = QtWidgets.QDoubleSpinBox()
    self.range_to = QtWidgets.QDoubleSpinBox()
    for spin in (self.range_from, self.range_to):
      spin.setDecimals(3)
      spin.setSingleStep(0.5)
    apply_button = QtWidgets.QPushButton("Apply replay range")
    apply_button.clicked.connect(self._refresh_active_tab)
    replay_layout.addWidget(QtWidgets.QLabel("Replay from [s]:"))
    replay_layout.addWidget(self.range_from)
    replay_layout.addWidget(QtWidgets.QLabel("to [s]:"))
    replay_layout.addWidget(self.range_to)
    replay_layout.addWidget(apply_button)
    replay_layout.addStretch(1)
    self.replay_controls.setVisible(self.replay_path is not None)
    layout.addWidget(self.replay_controls)

    self.tabs = QtWidgets.QTabWidget()
    self.tabs.currentChanged.connect(self._refresh_active_tab)
    layout.addWidget(self.tabs, 1)
    self._build_flight_tab()
    self._build_joint_tab()
    self._build_strip_flow_tab()
    self._build_strip_load_tab()
    self._build_aggregate_tab()
    self._build_browser_tab()

  def _plot(self, graphics: pg.GraphicsLayoutWidget, row: int, col: int, title: str, unit: str = "", legend: bool = True, show_time_values: bool = True) -> pg.PlotItem:
    plot = graphics.addPlot(row=row, col=col)
    style_plot(plot, title, unit, show_time_values)
    if legend: plot.addLegend(offset=(8, 8))
    return plot

  def _add_strip_controls(self, layout: QtWidgets.QVBoxLayout) -> None:
    controls = QtWidgets.QHBoxLayout()
    wing_combo = QtWidgets.QComboBox()
    wing_combo.addItems(("Right", "Left"))
    segment_combo = QtWidgets.QComboBox()
    segment_combo.addItems(("Humerus", "Radius", "Manus"))
    strip_spin = QtWidgets.QSpinBox()
    strip_spin.setMinimum(0)
    selector = (wing_combo, segment_combo, strip_spin)
    self.strip_controls.append(selector)
    wing_combo.currentIndexChanged.connect(lambda value: self._strip_selection_changed(0, value))
    segment_combo.currentIndexChanged.connect(lambda value: self._strip_selection_changed(1, value))
    strip_spin.valueChanged.connect(lambda value: self._strip_selection_changed(2, value))
    controls.addWidget(QtWidgets.QLabel("Wing:"))
    controls.addWidget(wing_combo)
    controls.addWidget(QtWidgets.QLabel("Segment:"))
    controls.addWidget(segment_combo)
    controls.addWidget(QtWidgets.QLabel("Strip:"))
    controls.addWidget(strip_spin)
    controls.addStretch(1)
    layout.addLayout(controls)
    self._sync_strip_controls()

  def _strip_selection_changed(self, field: int, value: int) -> None:
    if field == 0: self.strip_wing = value
    elif field == 1: self.strip_segment = value
    else: self.strip_local = value
    self._sync_strip_controls()
    self._refresh_active_tab()

  def _sync_strip_controls(self) -> None:
    maximum = 0
    if self.header is not None:
      maximum = max(0, (self.header.nh, self.header.nr, self.header.nm)[self.strip_segment]-1)
    self.strip_local = min(self.strip_local, maximum)
    for wing_combo, segment_combo, strip_spin in self.strip_controls:
      blockers = (QtCore.QSignalBlocker(wing_combo), QtCore.QSignalBlocker(segment_combo), QtCore.QSignalBlocker(strip_spin))
      wing_combo.setCurrentIndex(self.strip_wing)
      segment_combo.setCurrentIndex(self.strip_segment)
      strip_spin.setMaximum(maximum)
      strip_spin.setValue(self.strip_local)
      del blockers

  def _build_flight_tab(self) -> None:
    graphics = pg.GraphicsLayoutWidget()
    self.tabs.addTab(graphics, "Flight")
    blue = pg.mkPen((30, 90, 210), width=2)
    red = pg.mkPen((220, 50, 50), width=2, style=QtCore.Qt.DashLine)
    groups = (
      ("pos", ("<i>x</i>", "<i>y</i>", "<i>z</i>"), "m"),
      ("vel", ("<i>v</i><sub>x</sub>", "<i>v</i><sub>y</sub>", "<i>v</i><sub>z</sub>"), "m s<sup>&minus;1</sup>"),
      ("rpy", ("<i>&phi;</i>", "<i>&theta;</i>", "<i>&psi;</i>"), "&deg;"),
      ("w", ("<i>p</i>", "<i>q</i>", "<i>r</i>"), "&deg; s<sup>&minus;1</sup>")
    )
    for row, (group, titles, unit) in enumerate(groups):
      for axis, title in enumerate(titles):
        plot = self._plot(graphics, row, axis, title, unit, False, False)
        self.curves[f"flight.{group}.{axis}.state"] = plot.plot(pen=blue, name="state")
        self.curves[f"flight.{group}.{axis}.cmd"] = plot.plot(pen=red, name="cmd")
    for axis, axis_name in enumerate(AXES):
      plot = self._plot(graphics, 4, axis, f"<i>F</i><sub>{axis_name}</sub><sup>W</sup>", "N", False, True)
      if axis < 2: plot.setYRange(-50.0, 50.0, padding=0.0)
      else: plot.setYRange(-80.0, 10.0, padding=0.0)
      self.curves[f"flight.aero_force.{axis}"] = plot.plot(pen=blue)

  def _build_joint_tab(self) -> None:
    joint_tabs = QtWidgets.QTabWidget()
    self.tabs.addTab(joint_tabs, "Joint")
    blue = pg.mkPen((30, 90, 210), width=2)
    red = pg.mkPen((220, 50, 50), width=2, style=QtCore.Qt.DashLine)
    torque_pen = pg.mkPen((20, 150, 90), width=2)
    for wing in range(2):
      graphics = pg.GraphicsLayoutWidget()
      joint_tabs.addTab(graphics, "Right" if wing == 0 else "Left")
      for local_joint in range(6):
        joint = 6*wing+local_joint
        show_time_values = local_joint == 5
        angle_plot = self._plot(graphics, local_joint, 0, f"<i>&theta;</i><sub>{joint+1}</sub>", "&deg;", False, show_time_values)
        torque_plot = self._plot(graphics, local_joint, 1, f"<i>&tau;</i><sub>{joint+1}</sub>", "N m", False, show_time_values)
        angle_plot.setYRange(INITIAL_JOINT_DEG[joint]-100.0, INITIAL_JOINT_DEG[joint]+100.0, padding=0.0)
        torque_plot.setYRange(-20.0, 20.0, padding=0.0)
        self.curves[f"joint.{joint}.state"] = angle_plot.plot(pen=blue, name="state")
        self.curves[f"joint.{joint}.cmd"] = angle_plot.plot(pen=red, name="cmd")
        self.curves[f"joint.{joint}.torque"] = torque_plot.plot(pen=torque_pen)

  def _build_strip_flow_tab(self) -> None:
    widget = QtWidgets.QWidget()
    layout = QtWidgets.QVBoxLayout(widget)
    self._add_strip_controls(layout)
    graphics = pg.GraphicsLayoutWidget()
    layout.addWidget(graphics, 1)
    self.tabs.addTab(widget, "Strip Flow")
    blue = (30, 90, 210)
    green = (20, 150, 90)
    purple = (160, 70, 180)
    specs = (
      ("alpha", "<i>&alpha;</i>", "&deg;"),
      ("alpha_dot", "<i>&alpha;&#775;</i>", "&deg; s<sup>&minus;1</sup>"),
      ("speed", "<i>U</i>", "m s<sup>&minus;1</sup>"),
      ("Re", "<i>Re</i>", "&times;10<sup>5</sup>"),
      ("stall", "<i>X</i>", ""),
      ("coeff", "<i>C</i>", "")
    )
    for index, (name, title, unit) in enumerate(specs):
      plot = self._plot(graphics, index//3, index%3, title, unit, True, index//3 == 1)
      if name == "alpha": plot.setYRange(-30.0, 80.0, padding=0.0)
      elif name == "Re": plot.setYRange(0.0, 3.0, padding=0.0)
      if name == "stall":
        self.curves["flow.X"] = plot.plot(pen=pg.mkPen(blue, width=2), name=paper_text("<i>X</i>", 10))
        self.curves["flow.X_eq"] = plot.plot(pen=pg.mkPen((230, 160, 160), width=1.5), name=paper_text("<i>X</i><sub>eq</sub>", 10))
        self.curves["flow.X_target"] = plot.plot(pen=pg.mkPen((145, 205, 170), width=1.5), name=paper_text("<i>X</i><sub>t</sub>", 10))
        self.curves["flow.stall_active"] = plot.plot(pen=pg.mkPen((*purple, 70), width=1), fillLevel=0.0, brush=pg.mkBrush(*purple, 35), name=paper_text("stall", 10))
      elif name == "coeff":
        for curve_name, label, color in (("Cd", "<i>C</i><sub>D</sub>", blue), ("Cl_lut", "<i>C</i><sub>L,LUT</sub>", (135, 205, 165)), ("Cl_dynamic", "<i>C</i><sub>L,DS</sub>", green), ("Cl_wagner", "&Delta;<i>C</i><sub>L,WJ</sub>", (220, 120, 30)), ("Cl_total", "<i>C</i><sub>L</sub>", (0, 0, 0)), ("Cm", "<i>C</i><sub>M</sub>", purple)):
          self.curves[f"flow.{curve_name}"] = plot.plot(pen=pg.mkPen(color, width=2), name=paper_text(label, 10))
      else:
        self.curves[f"flow.{name}"] = plot.plot(pen=pg.mkPen(blue, width=2), name=paper_text(title, 10))

  def _build_strip_load_tab(self) -> None:
    widget = QtWidgets.QWidget()
    layout = QtWidgets.QVBoxLayout(widget)
    self._add_strip_controls(layout)
    graphics = pg.GraphicsLayoutWidget()
    layout.addWidget(graphics, 1)
    self.tabs.addTab(widget, "Strip Loads")
    pens = {
      "lut": pg.mkPen((30, 90, 210), width=2),
      "dynamic": pg.mkPen((220, 50, 50), width=2),
      "wagner": pg.mkPen((150, 80, 190), width=2),
      "added_bias": pg.mkPen((230, 140, 20), width=2),
      "added_full": pg.mkPen((20, 150, 90), width=2, style=QtCore.Qt.DashLine)
    }
    for row, kind in enumerate(("force", "moment")):
      for axis, axis_name in enumerate(AXES):
        symbol = "F" if kind == "force" else "M"
        plot = self._plot(graphics, row, axis, f"<i>{symbol}</i><sub>{axis_name}</sub><sup>B</sup>", "N" if kind == "force" else "N m", True, row == 1)
        self.load_plots[(kind, axis)] = plot
        labels = {"lut": "LUT", "dynamic": "DS", "wagner": "WJ", "added_bias": "AM<sub>bias</sub>", "added_full": "AM<sub>full</sub>"}
        for contribution, pen in pens.items(): self.curves[f"load.{kind}.{axis}.{contribution}"] = plot.plot(pen=pen, name=paper_text(labels[contribution], 10))
        self.curves[f"load.{kind}.{axis}.total"] = plot.plot(pen=pg.mkPen((0, 0, 0), width=3), name=paper_text("&Sigma;", 10))

  def _build_aggregate_tab(self) -> None:
    widget = QtWidgets.QWidget()
    layout = QtWidgets.QVBoxLayout(widget)
    controls = QtWidgets.QHBoxLayout()
    self.aggregate_combo = QtWidgets.QComboBox()
    self.aggregate_combo.addItems(AGGREGATE_NAMES)
    self.aggregate_combo.currentIndexChanged.connect(self._refresh_active_tab)
    controls.addWidget(QtWidgets.QLabel("Segment:"))
    controls.addWidget(self.aggregate_combo)
    controls.addStretch(1)
    layout.addLayout(controls)
    graphics = pg.GraphicsLayoutWidget()
    layout.addWidget(graphics, 1)
    self.tabs.addTab(widget, "Strip Segment")
    self.aggregate_segment_pen = pg.mkPen((30, 90, 210), width=2)
    self.aggregate_body_pen = pg.mkPen((220, 50, 50), width=2, style=QtCore.Qt.DashLine)
    for row, kind in enumerate(("force", "torque")):
      for axis, axis_name in enumerate(AXES):
        symbol = "F" if kind == "force" else "M"
        plot = self._plot(graphics, row, axis, f"<i>{symbol}</i><sub>{axis_name}</sub><sup>B</sup>", "N" if kind == "force" else "N m", False, row == 1)
        self.aggregate_plots[(kind, axis)] = plot
        self.curves[f"aggregate.{kind}.{axis}"] = plot.plot(pen=self.aggregate_segment_pen)

  def _build_browser_tab(self) -> None:
    widget = QtWidgets.QWidget()
    layout = QtWidgets.QVBoxLayout(widget)
    controls = QtWidgets.QHBoxLayout()
    self.browser_combo = QtWidgets.QComboBox()
    self.browser_row = QtWidgets.QSpinBox()
    self.browser_col = QtWidgets.QSpinBox()
    self.browser_combo.currentIndexChanged.connect(self._browser_channel_changed)
    self.browser_row.valueChanged.connect(self._refresh_active_tab)
    self.browser_col.valueChanged.connect(self._refresh_active_tab)
    controls.addWidget(QtWidgets.QLabel("Channel:"))
    controls.addWidget(self.browser_combo, 1)
    controls.addWidget(QtWidgets.QLabel("row:"))
    controls.addWidget(self.browser_row)
    controls.addWidget(QtWidgets.QLabel("col:"))
    controls.addWidget(self.browser_col)
    layout.addLayout(controls)
    self.browser_plot = pg.PlotWidget()
    style_plot(self.browser_plot.getPlotItem(), "Channel")
    self.browser_curve = self.browser_plot.plot(pen=pg.mkPen((30, 90, 210), width=2))
    layout.addWidget(self.browser_plot, 1)
    self.tabs.addTab(widget, "Channel Browser")

  def _configure_schema(self, header: Header, descriptors: List[ChannelDescriptor]) -> None:
    self.header = header
    self.descriptors = descriptors
    self.channel_names = [descriptor.name for descriptor in descriptors]
    self.history = HistoryBuffer(header.capacity)
    self.browser_combo.blockSignals(True)
    self.browser_combo.clear()
    self.browser_combo.addItems(self.channel_names)
    self.browser_combo.blockSignals(False)
    self._browser_channel_changed()
    self._update_strip_limit()

  def _update_strip_limit(self) -> None:
    self._sync_strip_controls()

  def _strip_index(self) -> int:
    header = self.header
    wing = self.strip_wing
    local = self.strip_local
    segment = self.strip_segment
    if segment == 0: return wing*header.nh+local
    if segment == 1: return 2*header.nh+wing*header.nr+local
    return 2*(header.nh+header.nr)+wing*header.nm+local

  def _browser_channel_changed(self) -> None:
    if not self.descriptors or self.browser_combo.currentIndex() < 0: return
    descriptor = self.descriptors[self.browser_combo.currentIndex()]
    self.browser_row.setMaximum(max(0, descriptor.rows-1))
    self.browser_col.setMaximum(max(0, descriptor.cols-1))
    self.browser_plot.setLabel("left", paper_text(descriptor.unit, 10))
    self.browser_plot.setTitle(paper_text(channel_title(descriptor.name, descriptor.frame)))
    self._refresh_active_tab()

  def _data(self, names: Sequence[str]) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    if self.replay_path is not None:
      if self.replay_time.size == 0: return self.replay_time, {}
      relative = self.replay_time-self.replay_time[0]
      mask = (relative >= self.range_from.value()) & (relative <= self.range_to.value())
      return self.replay_time[mask], {name: self.replay_channels[name][mask] for name in names}
    if self.history is None: return np.empty((0,), np.float64), {}
    return self.history.get(names)

  def _view(self, time: np.ndarray, channels: Dict[str, np.ndarray]) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    if time.size == 0: return time, channels
    relative = (time-time[0]).astype(np.float32)
    stride = max(1, (time.size+self.max_points-1)//self.max_points)
    return relative[::stride], {name: values[::stride] for name, values in channels.items()}

  def _set_curve(self, name: str, time: np.ndarray, values: np.ndarray) -> None:
    self.curves[name].setData(time, values)

  def _update_flight(self) -> None:
    names = ("state.pos", "state.vel", "state.R", "state.w", "cmd.pos", "cmd.vel", "cmd.R", "cmd.w", "segment.force", "body.ellipsoid_force")
    time, channels = self._data(names)
    time, channels = self._view(time, channels)
    if time.size == 0: return
    state_rpy = rotation_to_rpy(channels["state.R"])
    cmd_rpy = rotation_to_rpy(channels["cmd.R"])
    body_aero_force = np.sum(channels["segment.force"], axis=1)+channels["body.ellipsoid_force"]
    world_aero_force = np.einsum("nij,nj->ni", channels["state.R"], body_aero_force)
    for axis in range(3):
      self._set_curve(f"flight.pos.{axis}.state", time, channels["state.pos"][:, axis])
      self._set_curve(f"flight.pos.{axis}.cmd", time, channels["cmd.pos"][:, axis])
      self._set_curve(f"flight.vel.{axis}.state", time, channels["state.vel"][:, axis])
      self._set_curve(f"flight.vel.{axis}.cmd", time, channels["cmd.vel"][:, axis])
      self._set_curve(f"flight.rpy.{axis}.state", time, state_rpy[:, axis])
      self._set_curve(f"flight.rpy.{axis}.cmd", time, cmd_rpy[:, axis])
      self._set_curve(f"flight.w.{axis}.state", time, channels["state.w"][:, axis]*RAD2DEG)
      self._set_curve(f"flight.w.{axis}.cmd", time, channels["cmd.w"][:, axis]*RAD2DEG)
      self._set_curve(f"flight.aero_force.{axis}", time, world_aero_force[:, axis])

  def _update_joints(self) -> None:
    names = ("joint.theta", "joint.theta_cmd", "servo.torque")
    time, channels = self._data(names)
    time, channels = self._view(time, channels)
    if time.size == 0: return
    for joint in range(12):
      self._set_curve(f"joint.{joint}.state", time, channels["joint.theta"][:, joint]*RAD2DEG)
      self._set_curve(f"joint.{joint}.cmd", time, channels["joint.theta_cmd"][:, joint]*RAD2DEG)
      self._set_curve(f"joint.{joint}.torque", time, channels["servo.torque"][:, joint])

  def _update_strip_flow(self) -> None:
    flow_names = ["alpha", "alpha_dot", "speed", "Re", "X", "X_eq", "X_target", "stall_active", "Cd", "Cl_lut", "Cl_dynamic", "Cm"]
    has_wagner = "strip.Cl_wagner" in self.channel_names
    if has_wagner: flow_names.append("Cl_wagner")
    names = tuple(f"strip.{name}" for name in flow_names)
    time, channels = self._data(names)
    time, channels = self._view(time, channels)
    if time.size == 0: return
    index = self._strip_index()
    self._set_curve("flow.alpha", time, channels["strip.alpha"][:, index]*RAD2DEG)
    self._set_curve("flow.alpha_dot", time, channels["strip.alpha_dot"][:, index]*RAD2DEG)
    self._set_curve("flow.speed", time, channels["strip.speed"][:, index])
    self._set_curve("flow.Re", time, channels["strip.Re"][:, index]*1.0e-5)
    for name in ("X", "X_eq", "X_target", "stall_active", "Cd", "Cl_lut", "Cl_dynamic", "Cm"):
      self._set_curve(f"flow.{name}", time, channels[f"strip.{name}"][:, index])
    wagner_cl = channels["strip.Cl_wagner"][:, index] if has_wagner else np.zeros(time.shape, dtype=np.float32)
    self._set_curve("flow.Cl_wagner", time, wagner_cl)
    self._set_curve("flow.Cl_total", time, channels["strip.Cl_dynamic"][:, index]+wagner_cl)

  def _update_strip_loads(self) -> None:
    force_contributions = ("lut", "dynamic", "wagner", "added_bias", "added_full")
    moment_contributions = ("lut", "added_bias", "added_full")
    has_wagner = "strip.wagner_force" in self.channel_names
    recorded_forces = tuple(name for name in force_contributions if name != "wagner" or has_wagner)
    names = tuple([f"strip.{name}_force" for name in recorded_forces] + [f"strip.{name}_moment" for name in moment_contributions])

    time, channels = self._data(names)
    time, channels = self._view(time, channels)
    if time.size == 0: return
    index = self._strip_index()

    for kind in ("force", "moment"):
      row_series: List[List[np.ndarray]] = []

      for axis in range(3):
        axis_series: List[np.ndarray] = []
        total = np.zeros(time.shape, dtype=np.float32)

        for contribution in force_contributions:
          if (kind == "moment" and contribution in ("dynamic", "wagner")) or (contribution == "wagner" and not has_wagner): values = np.zeros(time.shape, dtype=np.float32)
          else: values = channels[f"strip.{contribution}_{kind}"][:, index, axis]
          self._set_curve(f"load.{kind}.{axis}.{contribution}", time, values)
          axis_series.append(values)
          total += values

        self._set_curve(f"load.{kind}.{axis}.total", time, total)
        axis_series.append(total)
        row_series.append(axis_series)

      set_common_y_scale([self.load_plots[(kind, axis)] for axis in range(3)], row_series)

  def _update_aggregate(self) -> None:
    names = ("segment.force", "segment.torque", "body.ellipsoid_force", "body.ellipsoid_torque")
    time, channels = self._data(names)
    time, channels = self._view(time, channels)
    if time.size == 0: return
    selection = self.aggregate_combo.currentIndex()
    body_selected = selection == len(SEGMENT_NAMES)
    pen = self.aggregate_body_pen if body_selected else self.aggregate_segment_pen
    for kind in ("force", "torque"):
      row_series: List[List[np.ndarray]] = []
      for axis in range(3):
        if body_selected: values = channels[f"body.ellipsoid_{kind}"][:, axis]
        else: values = channels[f"segment.{kind}"][:, selection, axis]
        self.curves[f"aggregate.{kind}.{axis}"].setPen(pen)
        self._set_curve(f"aggregate.{kind}.{axis}", time, values)
        row_series.append([values])
      set_common_y_scale([self.aggregate_plots[(kind, axis)] for axis in range(3)], row_series)

  def _update_browser(self) -> None:
    if self.browser_combo.currentIndex() < 0: return
    name = self.browser_combo.currentText()
    time, channels = self._data((name,))
    time, channels = self._view(time, channels)
    if time.size == 0: return
    values = channels[name]
    if values.ndim == 1: selected = values
    elif values.ndim == 2: selected = values[:, self.browser_row.value()]
    else: selected = values[:, self.browser_row.value(), self.browser_col.value()]
    self.browser_curve.setData(time, selected)

  @QtCore.pyqtSlot()
  def _refresh_active_tab(self) -> None:
    if self.header is None: return
    index = self.tabs.currentIndex()
    if index == 0: self._update_flight()
    elif index == 1: self._update_joints()
    elif index == 2: self._update_strip_flow()
    elif index == 3: self._update_strip_loads()
    elif index == 4: self._update_aggregate()
    elif index == 5: self._update_browser()

  def _new_recorder(self) -> None:
    if self.record_enabled and self.reader.header is not None:
      self.recorder = LogRecorder(self.mmap_path, self.log_dir, descriptor_metadata(self.reader))

  def _rotate_session(self, reason: str) -> None:
    output = self.recorder.save() if self.recorder is not None else None
    if self.history is not None: self.history.clear()
    self._new_recorder()
    self.last_time = None
    if output is not None: self.status_label.setText(f"saved ({reason}): {output.name}")

  def _consume(self, time: np.ndarray, channels: Dict[str, np.ndarray], dropped: int) -> None:
    if time.size == 0: return
    epochs = channels["__reset_epoch"]
    start = 0
    while start < time.size:
      epoch = int(epochs[start])
      end = start+1
      while end < time.size and int(epochs[end]) == epoch: end += 1
      if self.session_epoch is None: self.session_epoch = epoch
      elif epoch != self.session_epoch:
        self._rotate_session("simulation reset")
        self.session_epoch = epoch
      elif self.last_time is not None and float(time[start]) < self.last_time:
        self._rotate_session("time reset")
      segment_channels = {name: values[start:end] for name, values in channels.items()}
      self.history.append(time[start:end], segment_channels)
      if self.recorder is not None: self.recorder.append(time[start:end], segment_channels, dropped if start == 0 else 0)
      self.last_time = float(time[end-1])
      start = end

  @QtCore.pyqtSlot()
  def _on_timer(self) -> None:
    try:
      session_changed = self.reader.mm is not None and self.reader.header is not None and self.reader.session_id() not in (0, self.reader.header.session_id)
      if self.reader.mm is not None and (self.reader.changed_on_disk() or session_changed):
        self._rotate_session("mmap session changed")
        self.reader.close()
        self.header = None
      if self.reader.mm is None:
        if not os.path.exists(self.mmap_path):
          self.status_label.setText(f"waiting: {self.mmap_path}")
          return
        self.reader.open()
        self._configure_schema(self.reader.header, self.reader.descriptors)
        self.wc_last = max(0, self.reader.write_count()-self.reader.header.capacity)
        self._new_recorder()

      wc = self.reader.write_count()
      if wc < self.wc_last:
        self._rotate_session("write count reset")
        self.wc_last = max(0, wc-self.reader.header.capacity)
      time, channels, dropped, _ = self.reader.read_range(self.wc_last, wc)
      self.wc_last = wc
      self.dropped_total += dropped
      self._consume(time, channels, dropped)
      self._refresh_active_tab()
      full_age = ""
      if time.size and "strip.full_added_time" in channels:
        full_age = f" | full-added age={max(0.0, float(time[-1]-channels['strip.full_added_time'][-1]))*1000.0:.1f} ms"
      self.status_label.setText(f"250 Hz | samples={self.history.count} | wc={wc} | dropped={self.dropped_total}{full_age}")
    except FileNotFoundError:
      self.status_label.setText(f"waiting: {self.mmap_path}")
    except Exception as error:
      self.status_label.setText(f"error: {error}")

  def _load_replay(self, path: Path) -> None:
    try:
      if path.suffix.lower() == ".npz":
        with np.load(path, allow_pickle=False) as loaded:
          self.replay_time = loaded["time"].astype(np.float64, copy=True)
          metadata = json.loads(str(loaded["__metadata__"]))
          self.replay_channels = {name: loaded[name].copy() for name in loaded.files if name not in ("time", "__metadata__")}
        descriptors = [ChannelDescriptor(**item) for item in metadata["channels"]]
        header = Header(metadata.get("version", 1), len(descriptors), 0, 0, max(1, self.replay_time.size), metadata.get("log_hz", 250), metadata["nh"], metadata["nr"], metadata["nm"], metadata.get("strip_order", 1), 0, metadata.get("schema_hash", 0), 0, 0, 0, 0)
      else:
        replay_reader = MMapReader(str(path))
        replay_reader.open()
        self.replay_time, self.replay_channels, _ = replay_reader.read_all()
        header = replay_reader.header
        descriptors = list(replay_reader.descriptors)
        replay_reader.close()

      self._configure_schema(header, descriptors)
      if self.replay_time.size:
        duration = float(self.replay_time[-1]-self.replay_time[0])
        for spin in (self.range_from, self.range_to): spin.setRange(0.0, max(0.0, duration))
        self.range_from.setValue(0.0)
        self.range_to.setValue(duration)
      self.status_label.setText(f"replay: {path.name} | samples={self.replay_time.size}")
      self._refresh_active_tab()
    except Exception as error:
      self.status_label.setText(f"replay error: {error}")

  def closeEvent(self, event) -> None:
    if self.recorder is not None:
      try:
        output = self.recorder.save()
        if output is not None: self.status_label.setText(f"saved: {output.name}")
      except Exception:
        pass
    self.reader.close()
    super().closeEvent(event)


def main() -> None:
  parser = argparse.ArgumentParser(description="Bird simulation mmap monitor and replay viewer")
  parser.add_argument("replay", nargs="?", default=None, help="Replay .npz or .mmap file")
  parser.add_argument("--path", dest="replay_path", default=None, help="Replay .npz or .mmap file")
  parser.add_argument("--mmap", default="/tmp/bird_sim.mmap", help="Live mmap path")
  parser.add_argument("--update-ms", type=int, default=40, help="GUI refresh interval")
  parser.add_argument("--log-dir", type=Path, default=Path("log"), help="Recording directory")
  parser.add_argument("--no-record", action="store_true", help="Disable live NPZ recording")
  args = parser.parse_args()
  replay_path = args.replay_path or args.replay

  app = QtWidgets.QApplication(sys.argv)
  configure_light_theme(app)
  window = MonitorWindow(args.mmap, max(10, args.update_ms), args.log_dir, replay_path, not args.no_record)
  window.show()
  sys.exit(app.exec_())


if __name__ == "__main__":
  main()
