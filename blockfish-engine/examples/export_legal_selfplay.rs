use block_stacker::{Config as BSConfig, Ruleset, Stacker};
use blockfish::{
    ai::{
        fast::{color_to_idx, FastBoard, FastPlace, FastPlaceGen, FastShapeTable},
        spatial_val::SpatialValueModel,
        BagTracker, BeamConfig, FastAI,
    },
    Input, StackerExt as _,
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

fn spawn_random_cheese(b: &mut FastBoard, rem_g: u16, lines_cleared: u8, rng: &mut u64) -> bool {
    let target_g: usize = if lines_cleared > 0 { 4 } else { 9 };
    let g = b.garbage_rows as usize;
    let unspawned = (rem_g as usize).saturating_sub(g);
    let num = target_g.saturating_sub(g).min(unspawned);
    if num == 0 {
        return true;
    }
    let old_rows = b.rows as usize;
    if old_rows + num >= 32 {
        return false;
    }
    let mut prev_hole: usize = if g > 0 {
        let m = (!b.data[0]) & 0x3FF;
        if m != 0 { m.trailing_zeros() as usize } else { 10 }
    } else {
        10
    };
    for r in (0..old_rows).rev() {
        b.data[r + num] = b.data[r];
    }
    for k in 0..num {
        let r = next_rng(rng);
        let hole = if prev_hole < 10 {
            let mut h = ((r >> 33) as usize) % 9;
            if h >= prev_hole {
                h += 1;
            }
            h
        } else {
            ((r >> 33) as usize) % 10
        };
        prev_hole = hole;
        b.data[num - 1 - k] = 0xFC00 | (0x3FF & !(1u16 << hole));
    }
    b.rows = (old_rows + num) as u8;
    b.garbage_rows = (g + num) as u8;
    true
}

fn avail_mask_from_vis(held: Option<u8>, q: &[u8], nqc: usize) -> u8 {
    let mut m = 0u8;
    if let Some(h) = held {
        m |= 1u8 << h;
    }
    let end = (nqc + 6).min(q.len());
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

/// Fixed 48-byte afterstate record (BFDS0002): 44-byte BFDS0001 + base_v (f32).
#[derive(Clone, Copy)]
struct MoveRecV2 {
    rows: [u16; 16],
    board_rows: u8,
    garbage_rows: u8,
    rem_g: u8,
    held: u8,
    next5: [u8; 5],
    bag_mask: u8,
    lines_cleared: u8,
    ds_cleared: u8,
    base_v: f32,
}

impl MoveRecV2 {
    fn write_to<W: Write>(&self, w: &mut W) -> std::io::Result<()> {
        let mut buf = [0u8; 48];
        for r in 0..16 {
            let le = self.rows[r].to_le_bytes();
            buf[2 * r] = le[0];
            buf[2 * r + 1] = le[1];
        }
        buf[32] = self.board_rows;
        buf[33] = self.garbage_rows;
        buf[34] = self.rem_g;
        buf[35] = self.held;
        buf[36..41].copy_from_slice(&self.next5);
        buf[41] = self.bag_mask;
        buf[42] = self.lines_cleared;
        buf[43] = self.ds_cleared;
        buf[44..48].copy_from_slice(&self.base_v.to_le_bytes());
        w.write_all(&buf)
    }
}

struct DecisionRecV2 {
    seed: u32,
    step: u16,
    rem_pcs: u16,
    target_idx: u8,
    flags: u8,
    moves: Vec<MoveRecV2>,
}

impl DecisionRecV2 {
    fn write_to<W: Write>(&self, w: &mut W) -> std::io::Result<()> {
        let mut hdr = [0u8; 12];
        hdr[0..4].copy_from_slice(&self.seed.to_le_bytes());
        hdr[4..6].copy_from_slice(&self.step.to_le_bytes());
        hdr[6..8].copy_from_slice(&self.rem_pcs.to_le_bytes());
        hdr[8] = self.moves.len() as u8;
        hdr[9] = self.target_idx;
        hdr[10] = self.flags;
        hdr[11] = 0;
        w.write_all(&hdr)?;
        for m in &self.moves {
            m.write_to(w)?;
        }
        Ok(())
    }
}

fn make_teacher_cfg(sm: Arc<SpatialValueModel>, worlds: usize, horizon: usize, width: usize) -> BeamConfig {
    let mut c = BeamConfig::default();
    c.spatial_model = Some(sm);
    c.rollout_top_k = 0;
    c.beam_width = 80;
    c.max_per_root = 16;
    c.bag_det_extend = true;
    c.endgame_solver = true;
    c.simulate_cheese_spawn = true;
    c.spatial_rerank_only = true;
    c.spatial_no_pe_internal = true;
    c.root_pimc_prescreen_worlds = 0;
    c.root_pimc_roots = 32;
    c.root_pimc_fine_roots = 10;
    c.root_pimc_worlds = worlds;
    c.root_pimc_horizon = horizon;
    c.root_pimc_width = width;
    c.root_pimc_margin = 500;
    c.root_pimc_weight_pct = 100;
    c.root_pimc_steady_g4 = false;
    c.root_pimc_cvar_pct = 25;
    c
}

fn play_and_export_legal_game(
    seed: u64,
    is_val: bool,
    sm: Arc<SpatialValueModel>,
    worlds: usize,
    horizon: usize,
    width: usize,
) -> Option<(usize, Vec<DecisionRecV2>)> {
    let fst = FastShapeTable::srs();
    let cfg = make_teacher_cfg(sm.clone(), worlds, horizon, width);
    let mut ai = FastAI::new(cfg);
    let mut tracker = BagTracker::new();
    let mut pgen = FastPlaceGen::new();
    let mut tmp_places: Vec<FastPlace> = Vec::with_capacity(64);

    let mut game_cfg = BSConfig::default();
    game_cfg.garbage.total_lines = Some(100);
    game_cfg.prng_seed = Some(seed);
    let mut stacker = Stacker::new(Ruleset::guideline().into(), game_cfg);

    let mut ds_total = 0usize;
    let mut pieces = 0usize;
    let mut pending: Vec<DecisionRecV2> = Vec::with_capacity(220);

    while ds_total < 100 && pieces < 320 && stacker.current_piece_type().is_some() {
        let snap = stacker.snapshot()?;
        let after_vis_mask = tracker.observe(snap.hold, &snap.queue);
        let g = garbage_rows(&stacker);
        let rem_g = 100usize.saturating_sub(ds_total) as u16;
        let res = ai.best_move(&snap, Some(g), Some(rem_g))?;

        let b0 = FastBoard::from_basic_matrix(&snap.matrix, g);
        let vis_q: Vec<u8> = snap.queue.iter().map(|&c| color_to_idx(c) as u8).collect();
        let cur_held: Option<u8> = snap.hold.map(|c| color_to_idx(c) as u8);
        let (c_nh, c_h) = if cur_held.is_some() {
            (vis_q.get(0).copied(), cur_held)
        } else {
            (vis_q.get(0).copied(), vis_q.get(1).copied())
        };

        // Execute chosen move on a clone to identify exact resulting board & did_hold
        let did_hold_star = res.inputs.first() == Some(&Input::Hold);
        let mut st_after = stacker.clone();
        let mut inp_exec = res.inputs.clone();
        inp_exec.pop();
        st_after.run(inp_exec);
        let (lines_star, ds_star) = st_after.hard_drop();

        // Extract all legal candidate moves and match chosen move by afterstate board (pre-spawn) + did_hold
        pgen.generate_for_state(&fst, &b0, c_nh, c_h, &mut tmp_places);

        // Compute exact target board signature after placing on stacker before cheese spawn
        // Match via (did_hold, lines_cleared, ds_cleared) + matrix above garbage after spawn
        let mut target_fp_idx: Option<usize> = None;
        if let Some(snap_a) = st_after.snapshot() {
            let g_a = garbage_rows(&st_after);
            let b_a = FastBoard::from_basic_matrix(&snap_a.matrix, g_a);
            let g_rem_after = g.saturating_sub(ds_star as u16);
            let spawned = g_a.saturating_sub(g_rem_after) as usize;
            for (mi, &fp) in tmp_places.iter().enumerate() {
                if fp.did_hold != did_hold_star {
                    continue;
                }
                let mut b_test = b0;
                let o = &fst.shapes[fp.color_idx as usize].orients[fp.tf.2 as usize];
                let (ln, ds_d, _) = b_test.place_and_sift(o, fp.norm_i, fp.norm_j);
                if ln as usize != lines_star || ds_d as usize != ds_star {
                    continue;
                }
                if b_test.rows as usize + spawned != b_a.rows as usize {
                    continue;
                }
                let mut same = true;
                for r in 0..b_test.rows as usize {
                    if (b_test.data[r] & 0x3FF) != (b_a.data[r + spawned] & 0x3FF) {
                        same = false;
                        break;
                    }
                }
                if same {
                    target_fp_idx = Some(mi);
                    break;
                }
            }
        }

        if let Some(tmi) = target_fp_idx {
            let crn_base = seed
                .wrapping_mul(0x9E3779B97F4A7C15)
                .wrapping_add((pieces as u64).wrapping_mul(0xBF58476D1CE4E5B9));

            let mut moves = Vec::with_capacity(tmp_places.len());
            let mut matched_idx: Option<usize> = None;

            for (mi, &fp) in tmp_places.iter().enumerate() {
                let is_star = mi == tmi;
                let mut b = b0;
                let o = &fst.shapes[fp.color_idx as usize].orients[fp.tf.2 as usize];
                let (lines, ds_d, _) = b.place_and_sift(o, fp.norm_i, fp.norm_j);
                let nrem = rem_g.saturating_sub(ds_d as u16);
                if nrem == 0 && b.rows < 20 {
                    if is_star {
                        matched_idx = None;
                        break;
                    }
                    continue;
                }
                let mut rng_loc = crn_base;
                if !spawn_random_cheese(&mut b, nrem, lines, &mut rng_loc) || b.rows >= 20 {
                    if !is_star {
                        continue;
                    }
                }
                let (nh, nqc) = if cur_held.is_some() {
                    if fp.did_hold { (c_nh, 1usize) } else { (cur_held, 1usize) }
                } else if fp.did_hold {
                    (c_nh, 2usize)
                } else {
                    (None, 1usize)
                };
                let in_h = nh.or_else(|| vis_q.get(nqc).copied());
                let av = avail_mask_from_vis(nh, &vis_q, nqc);
                let raw_bv = sm.evaluate_board_with_bag(&b, nrem, in_h, av);
                let base_v = ((raw_bv as f32) / 100.0).clamp(-50.0, 500.0);

                let mut rows = [0u16; 16];
                for r in 0..16 {
                    rows[r] = b.data[r] & 0x3FF;
                }
                let rec = MoveRecV2 {
                    rows,
                    board_rows: b.rows,
                    garbage_rows: b.garbage_rows,
                    rem_g: nrem.min(255) as u8,
                    held: nh.unwrap_or(7),
                    next5: next5_at(&vis_q, nqc),
                    bag_mask: after_vis_mask,
                    lines_cleared: lines,
                    ds_cleared: ds_d,
                    base_v,
                };
                if is_star {
                    matched_idx = Some(moves.len());
                }
                if moves.len() < 255 {
                    moves.push(rec);
                }
            }

            if let Some(ti) = matched_idx {
                if moves.len() >= 2 && ti < moves.len() {
                    pending.push(DecisionRecV2 {
                        seed: seed as u32,
                        step: pieces as u16,
                        rem_pcs: 0, // filled in once game completes
                        target_idx: ti as u8,
                        flags: if is_val { 1 } else { 0 },
                        moves,
                    });
                }
            }
        }

        // Advance real stacker
        stacker = st_after;
        ds_total += ds_star;
        pieces += 1;
    }

    if ds_total < 100 {
        return None;
    }
    for dr in &mut pending {
        dr.rem_pcs = pieces.saturating_sub(dr.step as usize + 1) as u16;
    }
    Some((pieces, pending))
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mut model_path = String::from("models/spatial_bc4.txt");
    let mut train_games: usize = 80;
    let mut val_games: usize = 8;
    let mut train_seed_start: u64 = 2001;
    let mut val_seed_start: u64 = 201;
    let mut worlds: usize = 6;
    let mut horizon: usize = 7;
    let mut width: usize = 16;
    let mut threads: usize = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4);
    let mut out_path = String::from("/tmp/legal_sp_v1.bin");

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--model" => { i += 1; model_path = args[i].clone(); }
            "--train-games" => { i += 1; train_games = args[i].parse().unwrap(); }
            "--val-games" => { i += 1; val_games = args[i].parse().unwrap(); }
            "--train-seed-start" => { i += 1; train_seed_start = args[i].parse().unwrap(); }
            "--val-seed-start" => { i += 1; val_seed_start = args[i].parse().unwrap(); }
            "--worlds" => { i += 1; worlds = args[i].parse().unwrap(); }
            "--horizon" => { i += 1; horizon = args[i].parse().unwrap(); }
            "--width" => { i += 1; width = args[i].parse().unwrap(); }
            "--threads" => { i += 1; threads = args[i].parse().unwrap(); }
            "--out" => { i += 1; out_path = args[i].clone(); }
            other => {
                eprintln!("Unknown arg: {}", other);
                std::process::exit(1);
            }
        }
        i += 1;
    }

    let sm = Arc::new(SpatialValueModel::load_from_file(&model_path).expect("load spatial model"));
    let mut jobs: Vec<(u64, bool)> = Vec::with_capacity(train_games + val_games);
    for k in 0..val_games {
        jobs.push((val_seed_start + k as u64, true));
    }
    for k in 0..train_games {
        jobs.push((train_seed_start + k as u64, false));
    }

    let total_jobs = jobs.len();
    println!(
        "=== export_legal_selfplay (BFDS0002) ===\n  model={} | train={} ({}..{}), val={} ({}..{}) | w={} h={} b={} threads={}\n  out={}",
        model_path, train_games, train_seed_start, train_seed_start + train_games as u64,
        val_games, val_seed_start, val_seed_start + val_games as u64,
        worlds, horizon, width, threads, out_path
    );

    let t0 = Instant::now();
    let next_idx = AtomicUsize::new(0);
    let done_cnt = AtomicUsize::new(0);
    let results: Mutex<Vec<(u64, usize, Vec<DecisionRecV2>)>> = Mutex::new(Vec::with_capacity(total_jobs));

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
                    if let Some((pcs, decs)) = play_and_export_legal_game(seed, is_val, sm_c.clone(), worlds, horizon, width) {
                        res_ref.lock().unwrap().push((seed, pcs, decs));
                    }
                    let d = done_ref.fetch_add(1, Ordering::Relaxed) + 1;
                    if d % 8 == 0 || d == total_jobs {
                        let elapsed = t0.elapsed().as_secs_f64();
                        let rate = d as f64 / elapsed.max(0.001);
                        let eta = (total_jobs - d) as f64 / rate.max(1e-6);
                        println!(
                            "  [{:3}/{}] elapsed={:6.1}s rate={:.2} games/s ETA={:6.1}s",
                            d, total_jobs, elapsed, rate, eta
                        );
                    }
                }
            });
        }
    });

    let mut all = results.into_inner().unwrap();
    all.sort_unstable_by_key(|(seed, _, _)| *seed);
    let n_games = all.len();
    let sum_pcs: usize = all.iter().map(|(_, p, _)| *p).sum();
    let n_decisions: usize = all.iter().map(|(_, _, d)| d.len()).sum();
    let n_moves: usize = all.iter().flat_map(|(_, _, d)| d.iter()).map(|dr| dr.moves.len()).sum();
    let n_val_decs: usize = all
        .iter()
        .flat_map(|(_, _, d)| d.iter())
        .filter(|dr| (dr.flags & 1) != 0)
        .count();

    if let Some(parent) = std::path::Path::new(&out_path).parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent).unwrap();
        }
    }
    let f = File::create(&out_path).expect("create out file");
    let mut w = BufWriter::with_capacity(1 << 20, f);
    w.write_all(b"BFDS0002").unwrap();
    w.write_all(&(n_games as u32).to_le_bytes()).unwrap();
    w.write_all(&(n_decisions as u32).to_le_bytes()).unwrap();
    for (_, _, decs) in &all {
        for dr in decs {
            dr.write_to(&mut w).unwrap();
        }
    }
    w.flush().unwrap();
    let sz = std::fs::metadata(&out_path).map(|m| m.len()).unwrap_or(0);
    println!(
        "DONE in {:.1}s | games={} mean_pcs={:.2} | decisions={} (val={}) | moves={} ({:.1}/dec) | size={:.2} MB",
        t0.elapsed().as_secs_f64(),
        n_games,
        sum_pcs as f64 / n_games.max(1) as f64,
        n_decisions,
        n_val_decs,
        n_moves,
        n_moves as f64 / n_decisions.max(1) as f64,
        sz as f64 / (1024.0 * 1024.0)
    );
}
