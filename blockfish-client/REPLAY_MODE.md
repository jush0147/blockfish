# Blockfish Native GUI — BC-4 Replay Mode

Native SDL2 replay player built directly into `blockfish-client` for viewing recorded BC-4 Mass Legal 100L Cheese Race replays (`bc4_mass_legal_v1` schema).

- Reuses Blockfish's native SDL2 board, Hold, 5-piece Next Queue, ghost piece, progress (`X/100L`), and stats rendering.
- **AI Search & AutoPlay are completely disabled** in Replay Mode (`blockfish::ai::AI` is never constructed and `best_move()` is never called).
- Reconstructs every step from `block-stacker` Guideline rules + `record.seed`, pre-caching all `Stacker` states for instant O(1) stepping, rewinding, and jumping.
- Strictly validates `step`, `cur`, `hold_before`, `queue_p5`, `did_hold`, `placed_piece`, `lines_cleared`, `ds_cleared`, `ds_total`, and final summary totals on every step; halts and displays an explicit error on any mismatch.

## Quick Start

1. Build `blockfish-client`:
   ```bash
   cargo build --release -p blockfish-client
   ```
2. Launch Replay Mode with any BC-4 Replay JSON (e.g. the 168-piece PB from Release `bc4-mass-legal-m5005`):
   ```bash
   ./target/release/blockfish-client --replay path/to/seed_102015_pcs_168.replay.json
   ```

### Optional CLI Flags
- `--start-step <N>`: Jump immediately to step `N` on startup (default: `0`)
- `--speed <PPS>`: Initial playback speed in pieces/sec (default: `2.5`)
- `--paused`: Start paused instead of auto-playing
- `--verify-only`: Headlessly verify every step and exit `0` on exact match / `1` on mismatch (no window opened)
- `--screenshot <path.bmp>`: Render a single native SDL2 frame at `--start-step`, save BMP, and exit
- `--export-video <path.mp4>`: Offline deterministic frame-stepped export to `1920x1080` H.264 (`yuv420p`, `+faststart`, no audio) via FFmpeg (preserves aspect ratio with centered black padding; default `--speed 5.0`, `--fps 60`, `--lead-sec 2.0`, `--tail-sec 2.0`)
- `--final-png <path.png>`: Save the final `1920x1080` frame as PNG during `--export-video`

### Example: Offline 1080p60 MP4 Export (Headless Linux)
```bash
xvfb-run -a -s "-screen 0 1280x800x24" \
  env SDL_VIDEODRIVER=x11 SDL_RENDER_DRIVER=software \
  ./target/release/blockfish-client \
  --replay seed_102015_pcs_168.replay.json \
  --export-video bc4_seed_102015_168p.mp4 \
  --final-png bc4_seed_102015_final.png \
  --fps 60 --speed 5.0 --lead-sec 2.0 --tail-sec 2.0
```

## Keyboard Controls (Replay Mode)

| Key | Action |
|---|---|
| `Space` / `P` | Play / Pause toggle (auto-restarts if at end) |
| `Right` / `.` | Step forward 1 piece |
| `Left` / `,` | Step backward 1 piece |
| `Shift+Right` / `PageDown` | Jump forward 10 pieces |
| `Shift+Left` / `PageUp` | Jump backward 10 pieces |
| `Up` / `+` / `=` | Increase playback speed (`0.5` .. `25.0` pps) |
| `Down` / `-` | Decrease playback speed |
| `R` / `Home` | Restart to step `0` |
| `End` | Jump to final step |
| `G` | Goto step (`0`–`9` to enter step number, `Enter` to jump, `Esc` to cancel) |
| `Esc` / `Q` | Quit |
