use block_stacker::{Config as BSConfig, Ruleset, Stacker};
use blockfish::{
    ai::{
        fast::{color_to_idx, FastBoard, FastPlace, FastPlaceGen, FastShapeTable},
        spatial_val::SpatialValueModel,
        BagTracker, BeamConfig, FastAI,
    },
    StackerExt as _,
};
use std::fs::File;
use std::io::{BufWriter, Write};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

fn garbage_rows(stacker: &Stacker) -> u16 {
    let mut max_g = 0u16;
    let mut any_g = false;
    for ((r, _), cc) in stacker.matrix() {
        if cc == 'G' {
            any_g = true;
            max_g = max_g.max(r + 1);
        }
    }
    if any_g { max_g } else { 0 }
}

fn next_rng(rng: &mut u64) -> u64 {
    *rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    *rng
}

fn avail_mask_5(held: Option<u8>, q: &[u8], nqc: usize) -> u8 {
    let mut m = 0u8;
    if let Some(h) = held {
        m |= 1u8 << h;
    }
    let end = (nqc + 5).min(q.len());
    for &p in &q[nqc.min(q.len())..end] {
        m |= 1u8 << p;
    }
    if m == 0 { 0x7F } else { m }
}

fn next5_at(q: &[u8], qc: usize) -> [u8; 5] {
    let mut out = [7u8; 5];
    for k in 0..5 {
        if let Some(&p) = q.get(qc + k) {
            out[k] = p;
        }
    }
    out
}

/// Fixed 60-byte RL transition record (BFRL0001):
///   0..44 : standard 44-byte MoveRec (afterstate board + queue + flags)
///  44..48 : base_v (f32, spatial_bc4 predicted remaining pieces)
///  48..52 : true_rem_pcs (f32, actual remaining Legal pieces from this afterstate to clear 100L)
///  52..56 : game_id (u32)
///  56..58 : step (u16)
///  58     : flags (bit 0 = is_val, bit 1 = is_explore_move, bit 2 = game_topout)
///  59     : reserved (0)
#[derive(Clone, Copy)]
struct RlStateRec {
    raw44: [u8; 44],
    base_v: f32,
    true_rem_pcs: f32,
    game_id: u32,
    step: u16,
    flags: u8,
}

impl RlStateRec {
    fn write_to<W: Write>(&self, w: &mut W) -> std::io::Result<()> {
        let mut buf = [0u8; 60];
        buf[0..44].copy_from_slice(&self.raw44);
        buf[44..48].copy_from_slice(&self.base_v.to_le_bytes());
        buf[48..52].copy_from_slice(&self.true_rem_pcs.to_le_bytes());
        buf[52..56].copy_from_slice(&self.game_id.to_le_bytes());
        buf[56..58].copy_from_slice(&self.step.to_le_bytes());
        buf[58] = self.flags;
        buf[59] = 0;
        w.write_all(&buf)
    }
}

fn pack_raw44(
    b: &FastBoard,
    rem_g: u16,
    held: Option<u8>,
    next5: [u8; 5],
    bag_mask: u8,
    lines: u8,
    ds_d: u8,
) -> [u8; 44] {
    let mut rec = [0u8; 44];
    for r in 0..16 {
        let le = (b.data[r] & 0x3FF).to_le_bytes();
        rec[2 * r] = le[0];
        rec[2 * r + 1] = le[1];
    }
    rec[32] = b.rows;
    rec[33] = b.garbage_rows;
    rec[34] = rem_g.min(255) as u8;
    rec[35] = held.unwrap_or(7);
    rec[36..41].copy_from_slice(&next5);
    rec[41] = bag_mask;
    rec[42] = lines;
    rec[43] = ds_d;
    rec
}

fn make_fast_legal_cfg(sm: Arc<SpatialValueModel>, beam_w: usize) -> BeamConfig {
    let mut c = BeamConfig::default();
    c.spatial_model = Some(sm);
    c.rollout_top_k = 0;
    c.beam_width = beam_w;
    c.max_per_root = (beam_w / 4).max(4);
    c.bag_det_extend = true;
    c.endgame_solver = true;
    c.simulate_cheese_spawn = true;
    c.spatial_rerank_only = true;
    c.spatial_no_pe_internal = true;
    c.root_pimc_worlds = 0;
    c
}

/// Reconstruct the remaining 7-bag mask at index 0 of `vis_q` from `after_vis_mask`
/// (exact same semantics as `bmasks[nqc]` in `eval_resnet.rs` and `bag_mask_at` in `export_dataset.rs`).
fn bag_mask_at_head(vis_q: &[u8], after_vis_mask: u8) -> u8 {
    let mut bm = if after_vis_mask == 0x7F { 0u8 } else { after_vis_mask };
    for &p in vis_q.iter().rev() {
        let bit = 1u8 << p;
        if (bm & bit) != 0 {
            bm = bit;
        } else {
            bm |= bit;
        }
    }
    if bm == 0 { 0x7F } else { bm }
}

fn play_rl_game(
    seed: u64,
    is_val: bool,
    sm: Arc<SpatialValueModel>,
    beam_w: usize,
    eps_pct: usize,
) -> (usize, bool, Vec<RlStateRec>) {
    let fst = FastShapeTable::srs();
    let cfg = make_fast_legal_cfg(sm.clone(), beam_w);
    let mut ai = FastAI::new(cfg);
    let mut tracker = BagTracker::new();
    let mut pgen = FastPlaceGen::new();
    let mut tmp_places: Vec<FastPlace> = Vec::with_capacity(64);

    let mut game_cfg = BSConfig::default();
    game_cfg.garbage.total_lines = Some(100);
    game_cfg.prng_seed = Some(seed);
    let mut stacker = Stacker::new(Ruleset::guideline().into(), game_cfg);

    // Observe initial snapshot ONCE before the loop so BagTracker is never double-called!
    if let Some(snap0) = stacker.snapshot() {
        let _ = tracker.observe(snap0.hold, &snap0.queue);
    }

    let mut ds_total = 0usize;
    let mut pieces = 0usize;
    let mut rng = seed.wrapping_mul(0x9E3779B97F4A7C15).wrapping_add(31);
    let mut traj: Vec<RlStateRec> = Vec::with_capacity(240);

    while ds_total < 100 && pieces < 340 && stacker.current_piece_type().is_some() {
        let snap = match stacker.snapshot() {
            Some(s) => s,
            None => break,
        };
        let g = garbage_rows(&stacker);
        let rem_g = 100usize.saturating_sub(ds_total) as u16;

        // With probability eps_pct (only on training games & rem_g > 6), explore a random top-6 move!
        let do_explore = !is_val && rem_g > 6 && eps_pct > 0 && (((next_rng(&mut rng) >> 33) as usize) % 100) < eps_pct;

        let mut chosen_inputs = None;
        if do_explore {
            let b0 = FastBoard::from_basic_matrix(&snap.matrix, g);
            let vis_q: Vec<u8> = snap.queue.iter().map(|&c| color_to_idx(c) as u8).collect();
            let cur_held: Option<u8> = snap.hold.map(|c| color_to_idx(c) as u8);
            let (c_nh, c_h) = if cur_held.is_some() {
                (vis_q.get(0).copied(), cur_held)
            } else {
                (vis_q.get(0).copied(), vis_q.get(1).copied())
            };
            pgen.generate_for_state(&fst, &b0, c_nh, c_h, &mut tmp_places);
            let mut scored: Vec<(i64, FastPlace)> = Vec::with_capacity(tmp_places.len());
            for &fp in &tmp_places {
                let mut b = b0;
                let o = &fst.shapes[fp.color_idx as usize].orients[fp.tf.2 as usize];
                let (_, ds_d, _) = b.place_and_sift(o, fp.norm_i, fp.norm_j);
                if b.rows >= 19 {
                    continue;
                }
                let nrem = rem_g.saturating_sub(ds_d as u16);
                let (nh, nqc) = if cur_held.is_some() {
                    if fp.did_hold { (c_nh, 1usize) } else { (cur_held, 1usize) }
                } else if fp.did_hold {
                    (c_nh, 2usize)
                } else {
                    (None, 1usize)
                };
                let in_h = nh.or_else(|| vis_q.get(nqc).copied());
                let av = avail_mask_5(nh, &vis_q, nqc);
                let sc = sm.evaluate_board_with_bag(&b, nrem, in_h, av);
                scored.push((sc, fp));
            }
            if !scored.is_empty() {
                scored.sort_unstable_by_key(|x| x.0);
                let top_k = scored.len().min(6);
                let pick = ((next_rng(&mut rng) >> 33) as usize) % top_k;
                if let Some(inp) = ai.inputs_for_place(&snap.matrix, scored[pick].1) {
                    chosen_inputs = Some(inp);
                }
            }
        }

        let mut inputs = match chosen_inputs {
            Some(inp) => inp,
            None => match ai.best_move(&snap, Some(g), Some(rem_g)) {
                Some(res) => res.inputs,
                None => break,
            },
        };
        inputs.pop();
        stacker.run(inputs);
        let (lines_c, ds_c) = stacker.hard_drop();
        ds_total += ds_c;
        pieces += 1;

        // Observe the post-move snapshot ONCE and record its exact synchronized bag_mask!
        if ds_total < 100 {
            if let Some(snap_a) = stacker.snapshot() {
                let after_vis_a = tracker.observe(snap_a.hold, &snap_a.queue);
                let g_a = garbage_rows(&stacker);
                let rem_g_a = 100usize.saturating_sub(ds_total) as u16;
                let b_a = FastBoard::from_basic_matrix(&snap_a.matrix, g_a);
                if b_a.rows < 20 {
                    let vis_a: Vec<u8> = snap_a.queue.iter().map(|&c| color_to_idx(c) as u8).collect();
                    let held_a: Option<u8> = snap_a.hold.map(|c| color_to_idx(c) as u8);
                    let in_h_a = held_a.or_else(|| vis_a.first().copied());
                    let av_a = avail_mask_5(held_a, &vis_a, 0);
                    let raw_bv = sm.evaluate_board_with_bag(&b_a, rem_g_a, in_h_a, av_a);
                    let base_v = ((raw_bv as f32) / 100.0).clamp(-50.0, 500.0);
                    let bag_head = bag_mask_at_head(&vis_a, after_vis_a);
                    let raw44 = pack_raw44(
                        &b_a,
                        rem_g_a,
                        held_a,
                        next5_at(&vis_a, 0),
                        bag_head,
                        lines_c.min(4) as u8,
                        ds_c.min(4) as u8,
                    );
                    let fl = (if is_val { 1 } else { 0 }) | (if do_explore { 2 } else { 0 });
                    traj.push(RlStateRec {
                        raw44,
                        base_v,
                        true_rem_pcs: 0.0,
                        game_id: seed as u32,
                        step: pieces as u16,
                        flags: fl,
                    });
                }
            }
        }
    }

    let topout = ds_total < 100;
    if topout {
        // Penalize topped-out games realistically based on remaining garbage + topout penalty
        let rem_end = 100usize.saturating_sub(ds_total) as f32;
        let final_est = (pieces as f32) + 2.05 * rem_end + 15.0;
        for rec in &mut traj {
            rec.true_rem_pcs = (final_est - rec.step as f32).max(1.0);
            rec.flags |= 4;
        }
    } else {
        for rec in &mut traj {
            rec.true_rem_pcs = (pieces.saturating_sub(rec.step as usize)) as f32;
        }
    }

    (pieces, topout, traj)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mut model_path = String::from("models/spatial_bc4.txt");
    let mut train_games: usize = 1000;
    let mut val_games: usize = 100;
    let mut train_seed_start: u64 = 5001;
    let mut val_seed_start: u64 = 301;
    let mut beam_w: usize = 32;
    let mut eps_pct: usize = 12;
    let mut threads: usize = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4);
    let mut out_path = String::from("/tmp/legal_rl_v1.bin");

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--model" => { i += 1; model_path = args[i].clone(); }
            "--train-games" => { i += 1; train_games = args[i].parse().unwrap(); }
            "--val-games" => { i += 1; val_games = args[i].parse().unwrap(); }
            "--train-seed-start" => { i += 1; train_seed_start = args[i].parse().unwrap(); }
            "--val-seed-start" => { i += 1; val_seed_start = args[i].parse().unwrap(); }
            "--beam" => { i += 1; beam_w = args[i].parse().unwrap(); }
            "--eps" => { i += 1; eps_pct = args[i].parse().unwrap(); }
            "--threads" => { i += 1; threads = args[i].parse().unwrap(); }
            "--out" => { i += 1; out_path = args[i].clone(); }
            other => {
                eprintln!("Unknown arg: {}", other);
                std::process::exit(1);
            }
        }
        i += 1;
    }

    let mut sm_raw = SpatialValueModel::load_from_file(&model_path).expect("load spatial model");
    sm_raw.prepare_fast();
    let sm = Arc::new(sm_raw);

    let mut jobs: Vec<(u64, bool)> = Vec::with_capacity(train_games + val_games);
    for k in 0..val_games {
        jobs.push((val_seed_start + k as u64, true));
    }
    for k in 0..train_games {
        jobs.push((train_seed_start + k as u64, false));
    }
    let total_jobs = jobs.len();
    println!(
        "=== export_legal_rl (BFRL0001) ===\n  model={} | train={} ({}..{}), val={} ({}..{}) | beam={} eps={}% threads={}\n  out={}",
        model_path, train_games, train_seed_start, train_seed_start + train_games as u64,
        val_games, val_seed_start, val_seed_start + val_games as u64,
        beam_w, eps_pct, threads, out_path
    );

    let t0 = Instant::now();
    let next_idx = AtomicUsize::new(0);
    let done_cnt = AtomicUsize::new(0);
    let results: Mutex<Vec<(u64, usize, bool, Vec<RlStateRec>)>> = Mutex::new(Vec::with_capacity(total_jobs));

    std::thread::scope(|s| {
        for _ in 0..threads.max(1) {
            let sm_c = sm.clone();
            let jobs_ref = &jobs;
            let next_ref = &next_idx;
            let done_ref = &done_cnt;
            let res_ref = &results;
            s.spawn(move || {
                loop {
                    let idx = next_ref.fetch_add(1, Ordering::Relaxed);
                    if idx >= total_jobs {
                        break;
                    }
                    let (seed, is_val) = jobs_ref[idx];
                    let (pcs, top, traj) = play_rl_game(seed, is_val, sm_c.clone(), beam_w, eps_pct);
                    res_ref.lock().unwrap().push((seed, pcs, top, traj));
                    let d = done_ref.fetch_add(1, Ordering::Relaxed) + 1;
                    if d % 100 == 0 || d == total_jobs {
                        let elapsed = t0.elapsed().as_secs_f64();
                        let rate = d as f64 / elapsed.max(0.001);
                        let eta = (total_jobs - d) as f64 / rate.max(1e-6);
                        println!(
                            "  [{:4}/{}] elapsed={:6.1}s rate={:.2} games/s ETA={:5.1}s",
                            d, total_jobs, elapsed, rate, eta
                        );
                    }
                }
            });
        }
    });

    let mut all = results.into_inner().unwrap();
    all.sort_unstable_by_key(|(seed, _, _, _)| *seed);

    let ok_games: Vec<usize> = all.iter().filter(|x| !x.2).map(|x| x.1).collect();
    let tops = all.len() - ok_games.len();
    let mean_ok = if !ok_games.is_empty() {
        ok_games.iter().sum::<usize>() as f64 / ok_games.len() as f64
    } else {
        0.0
    };
    let total_states: usize = all.iter().map(|x| x.3.len()).sum();
    let val_states: usize = all
        .iter()
        .flat_map(|x| x.3.iter())
        .filter(|r| (r.flags & 1) != 0)
        .count();

    if let Some(parent) = std::path::Path::new(&out_path).parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent).unwrap();
        }
    }
    let mut w = BufWriter::with_capacity(1 << 20, File::create(&out_path).expect("create out"));
    w.write_all(b"BFRL0001").unwrap();
    w.write_all(&(all.len() as u32).to_le_bytes()).unwrap();
    w.write_all(&(total_states as u32).to_le_bytes()).unwrap();
    for (_, _, _, traj) in &all {
        for rec in traj {
            rec.write_to(&mut w).unwrap();
        }
    }
    w.flush().unwrap();
    let sz = std::fs::metadata(&out_path).map(|m| m.len()).unwrap_or(0);
    println!(
        "DONE in {:.1}s | games={} (ok={}, top={}) mean_ok={:.2} | states={} (val={}) | size={:.2} MB",
        t0.elapsed().as_secs_f64(),
        all.len(),
        ok_games.len(),
        tops,
        mean_ok,
        total_states,
        val_states,
        sz as f64 / (1024.0 * 1024.0)
    );
}
