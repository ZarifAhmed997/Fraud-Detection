Transaction-Anomaly-Detection
realtime-anomaly-engine/
├── README.md
├── CMakeLists.txt
├── .gitignore

├── src/
│   ├── main.cpp                # Entry point (wiring + config)
│   │
│   ├── engine/
│   │   ├── engine.hpp
│   │   ├── engine.cpp          # Core event-processing loop
│   │
│   ├── ingest/
│   │   ├── parser.hpp
│   │   ├── parser.cpp          # CSV / binary event parsing
│   │   ├── replay.hpp
│   │   └── replay.cpp          # Deterministic replay + rate control
│   │
│   ├── features/
│   │   ├── feature_store.hpp
│   │   ├── feature_store.cpp   # Sliding windows, aggregations
│   │   ├── statistics.hpp
│   │   └── statistics.cpp      # Mean, variance, entropy, velocity
│   │
│   ├── ml/
│   │   ├── inference.hpp
│   │   └── inference.cpp       # Lightweight model inference
│   │
│   ├── concurrency/
│   │   ├── ring_buffer.hpp     # Lock-free queues
│   │   └── thread_pool.hpp
│   │
│   ├── memory/
│   │   ├── object_pool.hpp     # Custom allocator
│   │   └── aligned_alloc.hpp   # Cache-line alignment
│   │
│   ├── utils/
│   │   ├── timestamp.hpp
│   │   ├── config.hpp
│   │   └── logging.hpp
│
├── ml/
│   ├── data/
│   │   └── features.csv        # Exported features from C++
│   │
│   ├── train.py                # Train anomaly model
│   ├── evaluate.py             # Metrics, ROC, precision/recall
│   └── export_model.py         # Export to ONNX / lightweight format
│
├── data/
│   ├── raw/
│   │   └── transactions.csv    # Synthetic or public dataset
│   │
│   └── generated/
│       └── events.bin          # Optional binary replay format
│
├── benchmarks/
│   ├── latency.cpp             # End-to-end latency tests
│   └── throughput.cpp          # Events/sec benchmarks
│
├── tests/
│   ├── test_features.cpp
│   ├── test_engine.cpp
│   └── test_parser.cpp
│
├── scripts/
│   ├── generate_data.py        # Synthetic data generator
│   └── run_benchmarks.sh
│
└── docs/
    ├── architecture.md         # High-level design
    ├── performance.md          # Latency + throughput results
    └── ml_pipeline.md          # Feature + ML explanation

