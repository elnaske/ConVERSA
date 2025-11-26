# ConVERSA
A framework for evaluating neural speech codecs under bitrate, latency, and computational constraints, built for compatibility with ESPNet and VERSA.

Supported metrics:
- Bitrate
  - [x] kbps
  - [x] codebook size
- Size
  - [x] Number of parameters
  - [x] Memory
- Number of operations
  - [x] FLOPs (total number of operations)
  - [ ] FLOPS (number of operations per second)
  - [x] MACS
- Latency
  - [ ] Inference time
  - [ ] RTF
- Reconstruction
  - [ ] VERSA metrics
