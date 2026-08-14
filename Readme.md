# Bird simulation monitor

The simulator publishes a 20-second, 250 Hz telemetry ring to
`/tmp/bird_sim.mmap`. The mmap schema is stored in the file, so the Python
reader does not contain hard-coded C++ payload offsets.

Install the monitor dependencies:

```bash
python3 -m pip install -r requirements-monitor.txt
```

Run the simulator and live monitor:

```bash
./build/run
python3 apps/bird_monitor.py
```

The live monitor automatically saves full-rate `.npz` recordings and mmap
snapshots below `bird_logs/` when it closes or detects a simulation reset.
Recording can be disabled with `--no-record`.

Replay either format:

```bash
python3 apps/bird_monitor.py bird_logs/npz/0814_210000_000000.npz
python3 apps/bird_monitor.py bird_logs/mmap/0814_210000_000000.mmap
```

Use `--mmap`, `--log-dir`, and `--update-ms` to override the live mmap path,
recording directory, and GUI refresh interval.
