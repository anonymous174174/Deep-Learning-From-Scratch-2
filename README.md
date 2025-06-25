# Deep_Learning_From_Scratch-2

ffmpeg \
  -hwaccel cuda \                             # Use GPU for decoding
  -threads 0 \                                # Use all available CPU threads
  -ss 00:01:30 \                              # Start time
  -to 00:03:45 \                              # End time
  -i input.mp4 \                              # Input file
  -vf "atadenoise=3:3:6:6,hqdn3d=4:4:6:6,unsharp=7:7:0.5" \  # CPU-based filters
  -c:v h264_nvenc \                           # NVIDIA GPU-based H.264 encoding
  -preset fast \                              # Encoding preset: fast, default for NVENC
  -cq 22 \                                    # Constant quality mode (like CRF for NVENC)
  -rc vbr \                                   # Rate control mode (variable bitrate)
  -c:a copy \                                 # Copy audio without re-encoding
  output_slice.mp4                            # Output file
