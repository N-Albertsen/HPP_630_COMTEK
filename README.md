
---


## Installation

For MPI support:
```bash
# Ubuntu/Debian
sudo apt install openmpi-bin libopenmpi-dev
```

```bash
python3 -m venv .venv
```

```bash
source .venv/bin/activate
```

```bash
pip install -r requirements.txt
```

---

## Usage Breadth-First Search

### Run all benchmarks (Tasks IV–VII)
```bash
cd BFS           
python3 main.py
```

---

## Usage JPEG image compression

### Run all benchmarks (Tasks IV–VII)
```bash
cd JPEG
python run_benchmark.py                     # uses synthetic test image
python run_benchmark.py my_photo.jpg        # uses custom image
```