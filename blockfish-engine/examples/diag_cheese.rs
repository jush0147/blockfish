use block_stacker::{Config as BSConfig, Ruleset, Stacker};
use blockfish::{
    ai::{
        cheese_val::{FEAT_DIM, MLP_HIDDEN},
        fast::{color_to_idx, FastBoard, FastPlaceGen, FastShapeTable},
        fast_eval::{eval_fast_with_bag, EvalWeights},
        BeamConfig, FastAI, LearnedValueModel, SpatialValueModel,
    },
    StackerExt as _,
};
use std::collections::HashSet;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::sync::Arc;
use std::time::Instant;

fn load_model(path: &str) -> LearnedValueModel {
    let f = File::open(path).expect("open model");
    let mut lines = BufReader::new(f).lines().map(|l| l.unwrap());
    let mut m = LearnedValueModel::default();
    let l1: Vec<f32> = lines
        .next()
        .unwrap()
        .split_whitespace()
        .map(|s| s.parse().unwrap())
        .collect();
    m.linear_bias = l1[0];
    m.mlp_weight = l1[1];
    let l2: Vec<f32> = lines
        .next()
        .unwrap()
        .split_whitespace()
        .map(|s| s.parse().unwrap())
        .collect();
    let n_f = (l2.len() / 3).min(FEAT_DIM);
    for j in 0..n_f {
        m.mean[j] = l2[j * 3];
        m.inv_std[j] = l2[j * 3 + 1];
        m.linear_w[j] = l2[j * 3 + 2];
    }
    m.b2 = lines.next().unwrap().trim().parse().unwrap();
    for h in 0..MLP_HIDDEN {
        let lh: Vec<f32> = lines
            .next()
            .unwrap()
            .split_whitespace()
            .map(|s| s.parse().unwrap())
            .collect();
        m.b1[h] = lh[0];
        m.w2[h] = lh[1];
        let n_w = lh.len().saturating_sub(2).min(FEAT_DIM);
        for j in 0..n_w {
            m.w1[h][j] = lh[2 + j];
        }
    }
    m.prepare_fast();
    m
}

trait DiagEvaluator: Sync {
    fn eval_b(&self, b: &FastBoard, rem_g: u16, in_h: Option<u8>, avail: u8) -> i64;
}

impl DiagEvaluator for LearnedValueModel {
    #[inline]
    fn eval_b(&self, b: &FastBoard, rem_g: u16, in_h: Option<u8>, avail: u8) -> i64 {
        self.evaluate_board_with_bag(b, rem_g, in_h, avail)
    }
}

impl DiagEvaluator for SpatialValueModel {
    #[inline]
    fn eval_b(&self, b: &FastBoard, rem_g: u16, in_h: Option<u8>, avail: u8) -> i64 {
        self.evaluate_board_with_bag(b, rem_g, in_h, avail)
    }
}

impl DiagEvaluator for EvalWeights {
    #[inline]
    fn eval_b(&self, b: &FastBoard, rem_g: u16, in_h: Option<u8>, avail: u8) -> i64 {
        eval_fast_with_bag(b, rem_g, in_h, avail, self)
    }
}

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

fn h1_info(stacker: &Stacker) -> (Option<u16>, usize, usize, usize) {
    let g = garbage_rows(stacker);
    let mut stack_h = [0usize; 10];
    let mut cells_above = 0usize;
    let mut h1_col = None;
    let mut h1_present = [false; 10];
    for ((r, c), _) in stacker.matrix() {
        if g > 0 && r == g - 1 {
            h1_present[c as usize] = true;
        }
        if r >= g {
            cells_above += 1;
            let sh = (r - g + 1) as usize;
            if sh > stack_h[c as usize] {
                stack_h[c as usize] = sh;
            }
        }
    }
    if g > 0 {
        h1_col = (0..10u16).find(|&c| !h1_present[c as usize]);
    }
    let blocks_over = h1_col.map_or(0, |c| stack_h[c as usize]);
    let chute_min = h1_col.map_or(0, |c| {
        let l = if c == 0 { 99 } else { stack_h[c as usize - 1] };
        let r = if c == 9 { 99 } else { stack_h[c as usize + 1] };
        l.min(r)
    });
    (h1_col, blocks_over, chute_min, cells_above)
}

fn extended_queue(stacker: &Stacker, len: usize) -> Vec<blockfish::Color> {
    use std::convert::TryInto;
    let mut sim = stacker.clone();
    let mut out = Vec::with_capacity(len);
    for &c in sim.next() {
        out.push(c.try_into().unwrap());
    }
    let mut ai_cfg = BeamConfig::default();
    ai_cfg.beam_width = 16;
    ai_cfg.max_per_root = 4;
    ai_cfg.rollout_top_k = 0;
    let mut ai = FastAI::new(ai_cfg);
    while out.len() < len {
        let snap = match sim.snapshot() {
            Some(s) => s,
            None => break,
        };
        if let Some(res) = ai.best_move(&snap, Some(0), Some(1)) {
            let mut inputs = res.inputs;
            inputs.pop();
            let had_hold = sim.held().is_some();
            let did_hold = inputs.first() == Some(&blockfish::Input::Hold);
            sim.run(inputs);
            sim.hard_drop();
            let nx = sim.next();
            if did_hold && !had_hold && nx.len() >= 2 {
                out.push(nx[nx.len() - 2].try_into().unwrap());
                out.push(nx[nx.len() - 1].try_into().unwrap());
            } else if let Some(&last) = nx.last() {
                out.push(last.try_into().unwrap());
            } else {
                break;
            }
        } else {
            break;
        }
    }
    out
}

#[derive(Clone, Copy, Default)]
struct GameStat {
    pcs: usize,
    nds: usize,
    zds: usize,
    ds1: usize,
    ds2: usize,
    ds3: usize,
    zds_h1_cov: usize,
    zds_g0: usize,
    zds_end: usize,
    zds_o: usize,
    zds_nar: usize,
    zds_wide: usize,
    ds_cov_next: usize,
    ds_cnt: usize,
    cells_sum: usize,
    r_end: usize,
    topout: bool,
}

fn run_legal_suite(label: &str, seeds: &[u64], cfg: &BeamConfig) {
    let t0 = Instant::now();
    let workers = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(2)
        .min(seeds.len().max(1));
    let next_idx = std::sync::atomic::AtomicUsize::new(0);
    let mut results: Vec<Option<GameStat>> = vec![None; seeds.len()];

    let play_one = |s: u64| -> GameStat {
        let mut game_cfg = BSConfig::default();
        game_cfg.garbage.total_lines = Some(100);
        game_cfg.prng_seed = Some(s);
        let mut stacker = Stacker::new(Ruleset::guideline().into(), game_cfg);
        let mut ai = FastAI::new(cfg.clone());
        let mut st = GameStat::default();
        let mut ds_total = 0usize;
        while ds_total < 100 && stacker.current_piece_type().is_some() {
            let snap = stacker.snapshot().unwrap();
            let g_rows = garbage_rows(&stacker);
            let rem_g = 100usize.saturating_sub(ds_total) as u16;
            let (_, blocks_over, chute_min, cells_above) = h1_info(&stacker);
            st.cells_sum += cells_above;
            let cur_c = snap.queue[0].as_char();
            let hold_c = snap.hold.or_else(|| snap.queue.get(1).copied()).map(|c| c.as_char());
            let res = match ai.best_move(&snap, Some(g_rows), Some(rem_g)) {
                Some(r) => r,
                None => { st.topout = true; break; }
            };
            let mut inputs = res.inputs;
            inputs.pop();
            let did_hold = inputs.first() == Some(&blockfish::Input::Hold);
            let placed_c = if did_hold { hold_c.unwrap_or(cur_c) } else { cur_c };
            stacker.run(inputs);
            let (lines, ds) = stacker.hard_drop();
            st.pcs += 1;
            ds_total += ds;
            st.nds += lines.saturating_sub(ds);
            if ds > 0 {
                st.ds_cnt += 1;
                if ds == 1 { st.ds1 += 1; }
                else if ds == 2 { st.ds2 += 1; }
                else { st.ds3 += 1; }
                let (_, new_bo, _, _) = h1_info(&stacker);
                if new_bo > 0 { st.ds_cov_next += 1; }
            } else {
                st.zds += 1;
                if rem_g <= 7 && rem_g == g_rows { st.zds_end += 1; }
                else if g_rows == 0 { st.zds_g0 += 1; }
                else if blocks_over > 0 { st.zds_h1_cov += 1; }
                else {
                    let is_o = placed_c == 'O';
                    let has_i = cur_c == 'I' || hold_c == Some('I');
                    if is_o { st.zds_o += 1; }
                    else if chute_min >= 1 && !has_i { st.zds_nar += 1; }
                    else { st.zds_wide += 1; }
                }
            }
        }
        if ds_total < 100 {
            st.topout = true;
            st.pcs = 400;
        }
        st.r_end = stacker.matrix().count();
        st
    };

    std::thread::scope(|sc| {
        let mut handles = Vec::new();
        for _ in 0..workers {
            handles.push(sc.spawn(|| {
                let mut local = Vec::new();
                loop {
                    let i = next_idx.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    if i >= seeds.len() {
                        break;
                    }
                    local.push((i, play_one(seeds[i])));
                }
                local
            }));
        }
        for h in handles {
            for (i, st) in h.join().unwrap() {
                results[i] = Some(st);
            }
        }
    });
    let res: Vec<GameStat> = results.into_iter().map(|x| x.unwrap()).collect();
    print_summary(label, &res, t0.elapsed().as_secs_f64());
}

fn print_summary(label: &str, res: &[GameStat], dt: f64) {
    let n = res.len() as f64;
    let pcs_vec: Vec<usize> = res.iter().map(|x| x.pcs).collect();
    let tot_pcs: usize = pcs_vec.iter().sum();
    let tot_ds: usize = res.iter().map(|x| x.ds_cnt).sum();
    let tot_cov: usize = res.iter().map(|x| x.ds_cov_next).sum();
    let tot_cells: usize = res.iter().map(|x| x.cells_sum).sum();
    let tops = res.iter().filter(|x| x.topout).count();
    let min_p = *pcs_vec.iter().min().unwrap_or(&0);
    let max_p = *pcs_vec.iter().max().unwrap_or(&0);
    println!(
        "{:<24} mean={:6.2} [{}..{}] top={} nds={:5.2} 0ds={:5.2} (cov={:.1} O={:.1} nar={:.1} wide={:.1} g0={:.1} end={:.1}) ds(1/2/3+)={:.1}/{:.1}/{:.1} cov_nxt={:.1}% avg_c={:.1} r_end={:.1} dt={:.1}s",
        label,
        tot_pcs as f64 / n,
        min_p,
        max_p,
        tops,
        res.iter().map(|x| x.nds).sum::<usize>() as f64 / n,
        res.iter().map(|x| x.zds).sum::<usize>() as f64 / n,
        res.iter().map(|x| x.zds_h1_cov).sum::<usize>() as f64 / n,
        res.iter().map(|x| x.zds_o).sum::<usize>() as f64 / n,
        res.iter().map(|x| x.zds_nar).sum::<usize>() as f64 / n,
        res.iter().map(|x| x.zds_wide).sum::<usize>() as f64 / n,
        res.iter().map(|x| x.zds_g0).sum::<usize>() as f64 / n,
        res.iter().map(|x| x.zds_end).sum::<usize>() as f64 / n,
        res.iter().map(|x| x.ds1).sum::<usize>() as f64 / n,
        res.iter().map(|x| x.ds2).sum::<usize>() as f64 / n,
        res.iter().map(|x| x.ds3).sum::<usize>() as f64 / n,
        100.0 * tot_cov as f64 / tot_ds.max(1) as f64,
        tot_cells as f64 / tot_pcs.max(1) as f64,
        res.iter().map(|x| x.r_end).sum::<usize>() as f64 / n,
        dt,
    );
    if res.len() <= 10 {
        println!("  per_seed={:?}", pcs_vec);
    }
    let _ = std::io::Write::flush(&mut std::io::stdout());
}

struct ExactSeedData {
    seed: u64,
    holes: Vec<u16>,
    pieces: Vec<u8>,
}

fn collect_exact_seeds(seeds: &[u64]) -> Vec<ExactSeedData> {
    use std::convert::TryInto;
    let hole_at_row = |st: &Stacker, target_r: u16| -> u16 {
        let mut present = [false; 10];
        for ((r, c), _) in st.matrix() {
            if r == target_r {
                present[c as usize] = true;
            }
        }
        (0..10u16).find(|&c| !present[c as usize]).unwrap()
    };
    seeds.iter().map(|&s| {
        let mut game_cfg = BSConfig::default();
        game_cfg.garbage.total_lines = Some(100);
        game_cfg.prng_seed = Some(s);
        let mut stacker = Stacker::new(Ruleset::guideline().into(), game_cfg);
        let mut holes = Vec::with_capacity(100);
        let g0 = garbage_rows(&stacker);
        for r in (0..g0).rev() {
            holes.push(hole_at_row(&stacker, r));
        }
        let mut pieces = Vec::with_capacity(320);
        pieces.push(color_to_idx(stacker.current_piece_type().unwrap().try_into().unwrap()) as u8);
        for &ch in stacker.next() {
            pieces.push(color_to_idx(ch.try_into().unwrap()) as u8);
        }
        let mut c = BeamConfig::default();
        c.beam_width = 48;
        c.max_per_root = 12;
        c.rollout_top_k = 0;
        let mut ai = FastAI::new(c);
        let mut ds_total = 0usize;
        while ds_total < 100 && stacker.current_piece_type().is_some() {
            let snap = stacker.snapshot().unwrap();
            let g_before = garbage_rows(&stacker);
            let rem_g = 100usize.saturating_sub(ds_total) as u16;
            let res = ai.best_move(&snap, Some(g_before), Some(rem_g)).unwrap();
            let had_hold = stacker.held().is_some();
            let did_hold = res.inputs.first() == Some(&blockfish::Input::Hold);
            let mut inputs = res.inputs;
            inputs.pop();
            stacker.run(inputs);
            let (_, ds) = stacker.hard_drop();
            ds_total += ds;
            let g_after = garbage_rows(&stacker);
            let new_g = g_after.saturating_sub(g_before.saturating_sub(ds as u16));
            for r in (0..new_g).rev() {
                holes.push(hole_at_row(&stacker, r));
            }
            let nx = stacker.next();
            if did_hold && !had_hold && nx.len() >= 2 {
                pieces.push(color_to_idx(nx[nx.len() - 2].try_into().unwrap()) as u8);
                pieces.push(color_to_idx(nx[nx.len() - 1].try_into().unwrap()) as u8);
            } else if !nx.is_empty() {
                pieces.push(color_to_idx(nx[nx.len() - 1].try_into().unwrap()) as u8);
            }
        }
        assert_eq!(holes.len(), 100);
        let skip_n = stacker.snapshot().map_or(0, |sn| sn.queue.len());
        if stacker.current_piece_type().is_some() {
            for &col in extended_queue(&stacker, 220).iter().skip(skip_n) {
                pieces.push(color_to_idx(col) as u8);
            }
        }
        ExactSeedData { seed: s, holes, pieces }
    }).collect()
}

fn run_oracle_suite<M: DiagEvaluator>(label: &str, exact_data: &[ExactSeedData], horizon: usize, beam_w: usize, exact_cheese: bool, mdl: &M) {
    #[derive(Clone, Copy)]
    struct ONode {
        board: FastBoard,
        held: Option<u8>,
        q_cursor: usize,
        rem_g: u16,
        first_idx: u16,
        score: i64,
    }
    #[derive(Clone, Copy, Eq, PartialEq, Hash)]
    struct OKey {
        board: FastBoard,
        held: Option<u8>,
        rem_g: u16,
    }
    type FxBuild = std::hash::BuildHasherDefault<blockfish::ai::beam::FxHasher>;

    let t0 = Instant::now();
    let fst = FastShapeTable::srs();

    let spawn_exact_cheese = |b: &mut FastBoard, rem_g: u16, lines_cleared: u8, holes: &[u16]| -> bool {
        let target_g: usize = if lines_cleared > 0 { 4 } else { 9 };
        let g = b.garbage_rows as usize;
        let unspawned = (rem_g as usize).saturating_sub(g);
        let num = target_g.saturating_sub(g).min(unspawned);
        if num == 0 { return true; }
        let old_rows = b.rows as usize;
        if old_rows + num >= 32 { return false; }
        let spawned_so_far = 100usize - unspawned;
        for r in (0..old_rows).rev() {
            b.data[r + num] = b.data[r];
        }
        for k in 0..num {
            let hole = holes[spawned_so_far + k];
            let dst_r = num - 1 - k;
            b.data[dst_r] = 0xFC00 | (0x3FF & !(1u16 << hole));
        }
        b.rows = (old_rows + num) as u8;
        b.garbage_rows = (g + num) as u8;
        true
    };

    let run_one = |ed: &ExactSeedData| -> GameStat {
        let mut game_cfg = BSConfig::default();
        game_cfg.garbage.total_lines = Some(100);
        game_cfg.prng_seed = Some(ed.seed);
        let mut stacker = Stacker::new(Ruleset::guideline().into(), game_cfg);
        let mut pgen = FastPlaceGen::new();
        let mut ai_in = FastAI::new(BeamConfig::default());
        let mut root_places = Vec::with_capacity(64);
        let mut tmp_places = Vec::with_capacity(64);
        let mut cur_beam: Vec<ONode> = Vec::with_capacity(beam_w);
        let mut children: Vec<ONode> = Vec::with_capacity(beam_w * 32);
        let mut seen: HashSet<OKey, FxBuild> = HashSet::with_capacity_and_hasher(beam_w * 4, FxBuild::default());

        let mut st = GameStat::default();
        let mut ds_total = 0usize;
        let mut q_pos = 0usize;

        while ds_total < 100 && stacker.current_piece_type().is_some() {
            let snap = stacker.snapshot().unwrap();
            let g_rows = garbage_rows(&stacker);
            let root_board = FastBoard::from_basic_matrix(&snap.matrix, g_rows);
            let rem_g0 = 100usize.saturating_sub(ds_total) as u16;
            let hold0 = snap.hold.map(|cc| color_to_idx(cc) as u8);
            let q_start = q_pos.min(ed.pieces.len());
            let q_end = (q_start + horizon).min(ed.pieces.len());
            let q = &ed.pieces[q_start..q_end];
            let q_len = q.len();
            if q_len == 0 { st.topout = true; st.pcs = 400; break; }
            let cur0 = if q_len > 0 { Some(q[0]) } else { None };
            let alt0 = if hold0.is_some() { hold0 } else if q_len > 1 { Some(q[1]) } else { None };

            let (_, blocks_over, chute_min, cells_above) = h1_info(&stacker);
            st.cells_sum += cells_above;
            let cur_ch = snap.queue[0].as_char();
            let hold_ch = snap.hold.or_else(|| snap.queue.get(1).copied()).map(|cc| cc.as_char());

            pgen.generate_for_state(&fst, &root_board, cur0, alt0, &mut root_places);
            if root_places.is_empty() { st.topout = true; st.pcs = 400; break; }

            let is_end = rem_g0 <= 7 && rem_g0 == g_rows;
            let eff_w = if is_end { beam_w.max(360) } else { beam_w };
            let eff_mpr = if is_end { 96 } else { (eff_w / 4).max(24) };
            let target_d = q_len.saturating_sub(if hold0.is_some() { 0 } else { 1 }).max(1);

            cur_beam.clear();
            seen.clear();
            let mut best_goal: Option<(usize, u16, i64)> = None;

            for (idx, &fp) in root_places.iter().enumerate() {
                let mut b = root_board;
                let o = &fst.shapes[fp.color_idx as usize].orients[fp.tf.2 as usize];
                let (lines, ds, _) = b.place_and_sift(o, fp.norm_i, fp.norm_j);
                let rem_g = rem_g0.saturating_sub(ds as u16);
                if exact_cheese && rem_g > 0 && !spawn_exact_cheese(&mut b, rem_g, lines, &ed.holes) {
                    continue;
                }
                let (held, qc) = if hold0.is_some() {
                    if fp.did_hold { (cur0, 1.min(q_len)) } else { (hold0, 1.min(q_len)) }
                } else if fp.did_hold {
                    (cur0, 2.min(q_len))
                } else {
                    (None, 1.min(q_len))
                };
                if rem_g == 0 && b.rows < 20 {
                    let gr = -30_000_000 + 100 + b.rows as i64;
                    if best_goal.map_or(true, |x| gr < x.2) { best_goal = Some((1, idx as u16, gr)); }
                }
                let in_h = held.or_else(|| q.get(qc).copied());
                let sc = mdl.eval_b(&b, rem_g, in_h, 0x7F);
                let key = OKey { board: b, held, rem_g };
                if seen.insert(key) {
                    cur_beam.push(ONode { board: b, held, q_cursor: qc, rem_g, first_idx: idx as u16, score: sc });
                }
            }

            for depth in 2..=target_d {
                children.clear();
                for node in &cur_beam {
                    if node.rem_g == 0 { continue; }
                    let qc = node.q_cursor;
                    let (c_nh, c_h) = if node.held.is_some() {
                        (q.get(qc).copied(), node.held)
                    } else {
                        (q.get(qc).copied(), q.get(qc + 1).copied())
                    };
                    pgen.generate_for_state(&fst, &node.board, c_nh, c_h, &mut tmp_places);
                    for &fp in &tmp_places {
                        let mut b = node.board;
                        let o = &fst.shapes[fp.color_idx as usize].orients[fp.tf.2 as usize];
                        let (lines, ds, _) = b.place_and_sift(o, fp.norm_i, fp.norm_j);
                        let rem_g = node.rem_g.saturating_sub(ds as u16);
                        if exact_cheese && rem_g > 0 && !spawn_exact_cheese(&mut b, rem_g, lines, &ed.holes) {
                            continue;
                        }
                        let (held, nqc) = if node.held.is_some() {
                            if fp.did_hold { (c_nh, (qc + 1).min(q_len)) } else { (node.held, (qc + 1).min(q_len)) }
                        } else if fp.did_hold {
                            (c_nh, (qc + 2).min(q_len))
                        } else {
                            (None, (qc + 1).min(q_len))
                        };
                        if rem_g == 0 && b.rows < 20 {
                            let gr = -30_000_000 + (depth as i64) * 100 + b.rows as i64;
                            if best_goal.map_or(true, |x| gr < x.2) { best_goal = Some((depth, node.first_idx, gr)); }
                        }
                        let in_h = held.or_else(|| q.get(nqc).copied());
                        let sc = mdl.eval_b(&b, rem_g, in_h, 0x7F);
                        children.push(ONode { board: b, held, q_cursor: nqc, rem_g, first_idx: node.first_idx, score: sc });
                    }
                }
                if children.is_empty() { break; }
                children.sort_unstable_by_key(|n| (n.score, n.first_idx));
                cur_beam.clear();
                seen.clear();
                let mut rc = vec![0usize; root_places.len()];
                for &ch in &children {
                    if cur_beam.len() >= eff_w { break; }
                    if rc[ch.first_idx as usize] >= eff_mpr { continue; }
                    let key = OKey { board: ch.board, held: ch.held, rem_g: ch.rem_g };
                    if seen.insert(key) {
                        rc[ch.first_idx as usize] += 1;
                        cur_beam.push(ch);
                    }
                }
                if cur_beam.len() < eff_w {
                    for &ch in &children {
                        if cur_beam.len() >= eff_w { break; }
                        let key = OKey { board: ch.board, held: ch.held, rem_g: ch.rem_g };
                        if seen.insert(key) { cur_beam.push(ch); }
                    }
                }
            }

            let chosen_idx = if let Some((_, ri, _)) = best_goal {
                ri as usize
            } else if let Some(best) = cur_beam.first() {
                best.first_idx as usize
            } else {
                0
            };
            let fp = root_places[chosen_idx];
            let mut inputs = ai_in.inputs_for_place(&snap.matrix, fp).unwrap();
            inputs.pop();
            if fp.did_hold {
                if hold0.is_none() { q_pos += 2; } else { q_pos += 1; }
            } else {
                q_pos += 1;
            }
            stacker.run(inputs);
            let (lines, ds) = stacker.hard_drop();
            st.pcs += 1;
            ds_total += ds;
            st.nds += lines.saturating_sub(ds);
            if ds > 0 {
                st.ds_cnt += 1;
                if ds == 1 { st.ds1 += 1; }
                else if ds == 2 { st.ds2 += 1; }
                else { st.ds3 += 1; }
                let (_, new_bo, _, _) = h1_info(&stacker);
                if new_bo > 0 { st.ds_cov_next += 1; }
            } else {
                st.zds += 1;
                if rem_g0 <= 7 && rem_g0 == g_rows { st.zds_end += 1; }
                else if g_rows == 0 { st.zds_g0 += 1; }
                else if blocks_over > 0 { st.zds_h1_cov += 1; }
                else {
                    let is_o = fp.color_idx == 1;
                    let has_i = cur_ch == 'I' || hold_ch == Some('I');
                    if is_o { st.zds_o += 1; }
                    else if chute_min >= 1 && !has_i { st.zds_nar += 1; }
                    else { st.zds_wide += 1; }
                }
            }
        }
        if ds_total < 100 { st.topout = true; st.pcs = 400; }
        st.r_end = stacker.matrix().count();
        st
    };

    let workers = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(2)
        .min(exact_data.len().max(1));
    let next_idx = std::sync::atomic::AtomicUsize::new(0);
    let mut results: Vec<Option<GameStat>> = vec![None; exact_data.len()];
    std::thread::scope(|sc| {
        let mut handles = Vec::new();
        for _ in 0..workers {
            handles.push(sc.spawn(|| {
                let mut local = Vec::new();
                loop {
                    let i = next_idx.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    if i >= exact_data.len() {
                        break;
                    }
                    local.push((i, run_one(&exact_data[i])));
                }
                local
            }));
        }
        for h in handles {
            for (i, st) in h.join().unwrap() {
                results[i] = Some(st);
            }
        }
    });
    let res: Vec<GameStat> = results.into_iter().map(|x| x.unwrap()).collect();
    print_summary(label, &res, t0.elapsed().as_secs_f64());
}

fn print_board_ascii(b: &FastBoard) {
    let top = (b.rows as usize).min(14);
    for r in (0..top).rev() {
        let bits = b.data[r] & 0x3FF;
        let is_g = r < b.garbage_rows as usize;
        let mut line = String::with_capacity(10);
        for c in 0..10 {
            if (bits & (1 << c)) != 0 {
                line.push(if is_g { 'G' } else { '#' });
            } else {
                line.push('.');
            }
        }
        println!("    {:2}: {}", r, line);
    }
}

fn trace_oracle_seed101(mdl: &LearnedValueModel, anneal_top3: bool) {
    use blockfish::ai::fast::idx_to_color;
    let ed = &collect_exact_seeds(&[101])[0];
    let mut game_cfg = BSConfig::default();
    game_cfg.garbage.total_lines = Some(100);
    game_cfg.prng_seed = Some(ed.seed);
    let mut stacker = Stacker::new(Ruleset::guideline().into(), game_cfg);
    let fst = FastShapeTable::srs();
    let mut pgen = FastPlaceGen::new();
    let mut ai_in = FastAI::new(BeamConfig::default());
    let mut root_places = Vec::with_capacity(64);
    let mut tmp_places = Vec::with_capacity(64);
    let mut cur_beam = Vec::new();
    let mut children = Vec::new();
    type FxBuild = std::hash::BuildHasherDefault<blockfish::ai::beam::FxHasher>;
    let mut seen: HashSet<(FastBoard, Option<u8>, u16), FxBuild> = HashSet::with_hasher(FxBuild::default());

    let spawn_exact_cheese = |b: &mut FastBoard, rem_g: u16, lines_cleared: u8, holes: &[u16]| -> bool {
        let target_g: usize = if lines_cleared > 0 { 4 } else { 9 };
        let g = b.garbage_rows as usize;
        let unspawned = (rem_g as usize).saturating_sub(g);
        let num = target_g.saturating_sub(g).min(unspawned);
        if num == 0 { return true; }
        let old_rows = b.rows as usize;
        if old_rows + num >= 32 { return false; }
        let spawned_so_far = 100usize - unspawned;
        for r in (0..old_rows).rev() { b.data[r + num] = b.data[r]; }
        for k in 0..num {
            let hole = holes[spawned_so_far + k];
            b.data[num - 1 - k] = 0xFC00 | (0x3FF & !(1u16 << hole));
        }
        b.rows = (old_rows + num) as u8;
        b.garbage_rows = (g + num) as u8;
        true
    };

    #[derive(Clone, Copy)]
    struct TNode { board: FastBoard, held: Option<u8>, q_cursor: usize, rem_g: u16, first_idx: u16, score: i64 }

    let mut ds_total = 0usize;
    let mut q_pos = 0usize;
    let mut pcs = 0usize;
    let mut printed_wide = 0usize;
    let mut printed_covnxt = 0usize;
    let mut wide_with_ds_avail = 0usize;
    let mut wide_no_ds_avail = 0usize;

    while ds_total < 100 && stacker.current_piece_type().is_some() {
        let snap = stacker.snapshot().unwrap();
        let g_rows = garbage_rows(&stacker);
        let root_board = FastBoard::from_basic_matrix(&snap.matrix, g_rows);
        let rem_g0 = 100usize.saturating_sub(ds_total) as u16;
        let hold0 = snap.hold.map(|cc| color_to_idx(cc) as u8);
        let q_start = q_pos.min(ed.pieces.len());
        let q_end = (q_start + 20).min(ed.pieces.len());
        let q = &ed.pieces[q_start..q_end];
        let q_len = q.len();
        let cur0 = Some(q[0]);
        let alt0 = hold0.or_else(|| q.get(1).copied());

        let (h1_col, blocks_over, chute_min, _) = h1_info(&stacker);
        pgen.generate_for_state(&fst, &root_board, cur0, alt0, &mut root_places);
        let is_end = rem_g0 <= 7 && rem_g0 == g_rows;
        let eff_w = if is_end { 400 } else { 240 };
        let eff_mpr = if is_end { 96 } else { 60 };
        let target_d = q_len.saturating_sub(if hold0.is_some() { 0 } else { 1 }).max(1);
        cur_beam.clear();
        seen.clear();
        let mut best_goal: Option<(usize, u16, i64)> = None;
        let mut any_root_ds = false;

        for (idx, &fp) in root_places.iter().enumerate() {
            let mut b = root_board;
            let o = &fst.shapes[fp.color_idx as usize].orients[fp.tf.2 as usize];
            let (lines, ds, _) = b.place_and_sift(o, fp.norm_i, fp.norm_j);
            if ds > 0 { any_root_ds = true; }
            let rem_g = rem_g0.saturating_sub(ds as u16);
            if rem_g > 0 && !spawn_exact_cheese(&mut b, rem_g, lines, &ed.holes) { continue; }
            let (held, qc) = if hold0.is_some() {
                if fp.did_hold { (cur0, 1.min(q_len)) } else { (hold0, 1.min(q_len)) }
            } else if fp.did_hold { (cur0, 2.min(q_len)) } else { (None, 1.min(q_len)) };
            if rem_g == 0 && b.rows < 20 {
                let gr = -30_000_000 + 100 + b.rows as i64;
                if best_goal.map_or(true, |x| gr < x.2) { best_goal = Some((1, idx as u16, gr)); }
            }
            let in_h = held.or_else(|| q.get(qc).copied());
            let mut sc = mdl.evaluate_board_with_bag(&b, rem_g, in_h, 0x7F);
            if anneal_top3 && !is_end && target_d > 1 {
                sc += (rem_g0.saturating_sub(rem_g) as i64) * 35;
            }
            if seen.insert((b, held, rem_g)) {
                cur_beam.push(TNode { board: b, held, q_cursor: qc, rem_g, first_idx: idx as u16, score: sc });
            }
        }
        for depth in 2..=target_d {
            children.clear();
            let mid_pen = if anneal_top3 && !is_end && depth < target_d {
                (35 * ((target_d - depth) as i64)) / (target_d as i64)
            } else { 0 };
            for node in &cur_beam {
                if node.rem_g == 0 { continue; }
                let qc = node.q_cursor;
                let (c_nh, c_h) = if node.held.is_some() { (q.get(qc).copied(), node.held) } else { (q.get(qc).copied(), q.get(qc + 1).copied()) };
                pgen.generate_for_state(&fst, &node.board, c_nh, c_h, &mut tmp_places);
                for &fp in &tmp_places {
                    let mut b = node.board;
                    let o = &fst.shapes[fp.color_idx as usize].orients[fp.tf.2 as usize];
                    let (lines, ds, _) = b.place_and_sift(o, fp.norm_i, fp.norm_j);
                    let rem_g = node.rem_g.saturating_sub(ds as u16);
                    if rem_g > 0 && !spawn_exact_cheese(&mut b, rem_g, lines, &ed.holes) { continue; }
                    let (held, nqc) = if node.held.is_some() {
                        if fp.did_hold { (c_nh, (qc + 1).min(q_len)) } else { (node.held, (qc + 1).min(q_len)) }
                    } else if fp.did_hold { (c_nh, (qc + 2).min(q_len)) } else { (None, (qc + 1).min(q_len)) };
                    if rem_g == 0 && b.rows < 20 {
                        let gr = -30_000_000 + (depth as i64) * 100 + b.rows as i64;
                        if best_goal.map_or(true, |x| gr < x.2) { best_goal = Some((depth, node.first_idx, gr)); }
                    }
                    let in_h = held.or_else(|| q.get(nqc).copied());
                    let mut sc = mdl.evaluate_board_with_bag(&b, rem_g, in_h, 0x7F);
                    if mid_pen > 0 {
                        sc += (rem_g0.saturating_sub(rem_g) as i64) * mid_pen;
                    }
                    children.push(TNode { board: b, held, q_cursor: nqc, rem_g, first_idx: node.first_idx, score: sc });
                }
            }
            if children.is_empty() { break; }
            children.sort_unstable_by_key(|n| (n.score, n.first_idx));
            cur_beam.clear();
            seen.clear();
            let mut rc = vec![0usize; root_places.len()];
            for &ch in &children {
                if cur_beam.len() >= eff_w { break; }
                if rc[ch.first_idx as usize] >= eff_mpr { continue; }
                if seen.insert((ch.board, ch.held, ch.rem_g)) { rc[ch.first_idx as usize] += 1; cur_beam.push(ch); }
            }
        }
        let chosen_idx = if let Some((_, ri, _)) = best_goal {
            ri as usize
        } else if anneal_top3 && !is_end && !cur_beam.is_empty() {
            let mut sums = vec![0i64; root_places.len()];
            let mut cnts = vec![0usize; root_places.len()];
            let mut best_sc = vec![i64::MAX; root_places.len()];
            for n in &cur_beam {
                let ri = n.first_idx as usize;
                if cnts[ri] < 3 {
                    if cnts[ri] == 0 { best_sc[ri] = n.score; }
                    sums[ri] += n.score;
                    cnts[ri] += 1;
                }
            }
            let mut best_ri = cur_beam[0].first_idx as usize;
            let mut best_agg = i64::MAX;
            for ri in 0..root_places.len() {
                if cnts[ri] == 0 { continue; }
                let padded = sums[ri] + ((3 - cnts[ri]) as i64) * (best_sc[ri] + 35);
                let agg = padded / 3;
                if agg < best_agg { best_agg = agg; best_ri = ri; }
            }
            best_ri
        } else {
            cur_beam.first().map_or(0, |n| n.first_idx as usize)
        };
        let fp = root_places[chosen_idx];
        let mut root_best_leaf: Vec<(u16, i64, u16)> = Vec::new();
        for n in &cur_beam {
            if !root_best_leaf.iter().any(|x| x.0 == n.first_idx) {
                root_best_leaf.push((n.first_idx, n.score, n.rem_g));
            }
            if root_best_leaf.len() >= 6 { break; }
        }

        let mut inputs = ai_in.inputs_for_place(&snap.matrix, fp).unwrap();
        inputs.pop();
        if fp.did_hold { if hold0.is_none() { q_pos += 2; } else { q_pos += 1; } } else { q_pos += 1; }
        stacker.run(inputs);
        let (lines, ds) = stacker.hard_drop();
        pcs += 1;
        ds_total += ds;
        let (_, new_bo, _, _) = h1_info(&stacker);
        let cur_ch = snap.queue[0].as_char();
        let hold_ch = snap.hold.or_else(|| snap.queue.get(1).copied()).map(|cc| cc.as_char());
        let has_i = cur_ch == 'I' || hold_ch == Some('I');
        let is_o = fp.color_idx == 1;
        let q_str: String = q.iter().take(8).map(|&x| idx_to_color(x as usize).as_char()).collect();
        let chosen_ch = idx_to_color(fp.color_idx as usize).as_char();
        let mut tag = None;
        if ds > 0 && new_bo > 0 {
            if printed_covnxt < 4 { printed_covnxt += 1; tag = Some("COV_NXT"); }
        } else if ds == 0 && blocks_over == 0 && g_rows > 0 && !(rem_g0 <= 7 && rem_g0 == g_rows) && !is_o && !(chute_min >= 1 && !has_i) {
            if any_root_ds { wide_with_ds_avail += 1; } else { wide_no_ds_avail += 1; }
            if printed_wide < 7 { printed_wide += 1; tag = Some("0DS_WIDE"); }
        }
        if ds == 0 {
            println!("  ZDS pc={:3} rg={:3} g={} bo={} ch={} ds_avail={} | {}({},{},r={:?},h={}) l={} | cur={} hold={:?} q={}",
                pcs, rem_g0, g_rows, blocks_over, chute_min, any_root_ds, chosen_ch, fp.norm_i, fp.norm_j, fp.tf.2, fp.did_hold, lines, cur_ch, snap.hold.map(|c| c.as_char()), q_str);
        }
        if let Some(tg) = tag {
            println!("\n[{}] pc={} rem_g={} g={} h1={:?} bo={} chute={} ds_avail={} hold={:?} q={}",
                tg, pcs, rem_g0, g_rows, h1_col, blocks_over, chute_min, any_root_ds, snap.hold.map(|c| c.as_char()), q_str);
            print_board_ascii(&root_board);
            println!("  chosen: {} (i={},j={},r={:?},hold={}) -> lines={}, ds={}, new_bo={}",
                chosen_ch, fp.norm_i, fp.norm_j, fp.tf.2, fp.did_hold, lines, ds, new_bo);
            for (ridx, sc, l_rem) in &root_best_leaf {
                let rfp = root_places[*ridx as usize];
                let mut rb = root_board;
                let ro = &fst.shapes[rfp.color_idx as usize].orients[rfp.tf.2 as usize];
                let (rl, rds, _) = rb.place_and_sift(ro, rfp.norm_i, rfp.norm_j);
                println!("    top_root: {} (i={},j={},r={:?},hold={}) 1ply(l={},ds={}) leaf_sc={} leaf_rem={}",
                    idx_to_color(rfp.color_idx as usize).as_char(), rfp.norm_i, rfp.norm_j, rfp.tf.2, rfp.did_hold, rl, rds, sc, l_rem);
            }
        }
    }
    println!("\nTrace finished: {} pieces, {} ds | wide_ds_avail={} wide_no_ds_avail={}", pcs, ds_total, wide_with_ds_avail, wide_no_ds_avail);
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mode = args.get(1).map(|s| s.as_str()).unwrap_or("gap5");

    let m_h2 = load_model(concat!(env!("CARGO_MANIFEST_DIR"), "/../models/crn_r3_h2stk03.txt"));
    let mut m_h2_nog0 = m_h2.clone();
    m_h2_nog0.linear_w[2] = 0.0;
    m_h2_nog0.prepare_fast();

    let mut cfg_legal = BeamConfig::default();
    cfg_legal.beam_width = 120;
    cfg_legal.max_per_root = 24;
    cfg_legal.rollout_top_k = 12;
    cfg_legal.endgame_wide = true;
    cfg_legal.value_model = Some(Arc::new(m_h2_nog0.clone()));

    let seeds5: Vec<u64> = (101..=105).collect();
    let seeds20: Vec<u64> = (201..=220).collect();

    if mode == "gap5" {
        println!("=== Legal Breakdown (seeds 101..=105) ===");
        run_legal_suite("LEGAL_h2nog0_p5_b120", &seeds5, &cfg_legal);

        let mut cfg_noroll = cfg_legal.clone();
        cfg_noroll.rollout_top_k = 0;
        run_legal_suite("LEGAL_h2nog0_p5_noroll", &seeds5, &cfg_noroll);

        let mut cfg_b240 = cfg_legal.clone();
        cfg_b240.beam_width = 240;
        cfg_b240.max_per_root = 48;
        cfg_b240.rollout_top_k = 24;
        run_legal_suite("LEGAL_h2nog0_p5_b240", &seeds5, &cfg_b240);

        println!("\n=== Oracle Gap Decomposition (seeds 101..=105) ===");
        let ed = collect_exact_seeds(&seeds5);
        run_oracle_suite("ORACLE_p6_nosp_b120", &ed, 6, 120, false, &m_h2_nog0);
        run_oracle_suite("ORACLE_p6_EXCH_b120", &ed, 6, 120, true, &m_h2_nog0);
        run_oracle_suite("ORACLE_p14_nosp_b240", &ed, 14, 240, false, &m_h2_nog0);
        run_oracle_suite("ORACLE_p14_EXCH_b240", &ed, 14, 240, true, &m_h2_nog0);
        run_oracle_suite("ORACLE_p20_EXCH_b240", &ed, 20, 240, true, &m_h2);
    } else if mode == "oracle5" {
        println!("=== Oracle Gap Decomposition (seeds 101..=105) ===");
        let ed = collect_exact_seeds(&seeds5);
        run_oracle_suite("ORACLE_p6_nosp_b120", &ed, 6, 120, false, &m_h2_nog0);
        run_oracle_suite("ORACLE_p6_EXCH_b120", &ed, 6, 120, true, &m_h2_nog0);
        run_oracle_suite("ORACLE_p14_nosp_b240", &ed, 14, 240, false, &m_h2_nog0);
        run_oracle_suite("ORACLE_p14_EXCH_b240", &ed, 14, 240, true, &m_h2_nog0);
        run_oracle_suite("ORACLE_p20_EXCH_b240", &ed, 20, 240, true, &m_h2);
    } else if mode == "probe5" {
        println!("=== Fast Oracle Probe (seeds 101..=105) ===");
        let ed = collect_exact_seeds(&seeds5);
        run_oracle_suite("ORACLE_p6_EXCH_b120", &ed, 6, 120, true, &m_h2_nog0);
        run_oracle_suite("ORACLE_p12_EXCH_b120", &ed, 12, 120, true, &m_h2_nog0);
    } else if mode == "trace101" {
        let ann = args.get(2).map(|s| s.as_str()) == Some("anneal");
        trace_oracle_seed101(&m_h2_nog0, ann);
    } else if mode == "ablation5" {
        println!("=== Oracle Ceiling Ablations (seeds 101..=105) ===");
        let ed = collect_exact_seeds(&seeds5);
        run_oracle_suite("ORACLE_p20_b240_h2nog0", &ed, 20, 240, true, &m_h2_nog0);
        run_oracle_suite("ORACLE_p20_b600_h2nog0", &ed, 20, 600, true, &m_h2_nog0);

        // Clean surgical patch of crn_r3_h2stk03 flaws:
        // [0] rem_g: 2.12 -> 1.65 (match true ~1.65 pieces/line instead of greedy 2.12)
        // [63] hand_O: +0.523 -> 0.0 (stop dumping O blindly)
        // [88] h2_only_row_g_blocked: -0.127 -> +0.20 (stop rewarding blocking h2)
        // [36] h2_blocks: 0 -> +0.22/unit, [39] h2_open: 0 -> -0.15
        let mut m_clean = m_h2_nog0.clone();
        m_clean.linear_w[0] = 1.65 / m_clean.inv_std[0];
        m_clean.linear_w[63] = 0.0;
        m_clean.linear_w[88] = 0.25 / m_clean.inv_std[88];
        m_clean.linear_w[36] = 0.20 / m_clean.inv_std[36];
        m_clean.linear_w[39] = -0.15 / m_clean.inv_std[39];
        m_clean.prepare_fast();
        run_oracle_suite("ORACLE_p20_b240_clean", &ed, 20, 240, true, &m_clean);

        let m_r4 = load_model(concat!(env!("CARGO_MANIFEST_DIR"), "/../models/crn_r4_gbox.txt"));
        run_oracle_suite("ORACLE_p20_b240_r4gbox", &ed, 20, 240, true, &m_r4);
    } else if mode == "oracle_eval5" {
        println!("=== Fast Oracle Evaluator Comparison p14_b160 (seeds 101..=105) ===");
        let ed = collect_exact_seeds(&seeds5);
        run_oracle_suite("OR_p14_b160_h2nog0", &ed, 14, 160, true, &m_h2_nog0);

        let mut m_clean = m_h2_nog0.clone();
        m_clean.linear_w[0] = 1.65 / m_clean.inv_std[0];
        m_clean.linear_w[63] = 0.0;
        m_clean.linear_w[88] = 0.25 / m_clean.inv_std[88];
        m_clean.linear_w[36] = 0.20 / m_clean.inv_std[36];
        m_clean.linear_w[39] = -0.15 / m_clean.inv_std[39];
        m_clean.prepare_fast();
        run_oracle_suite("OR_p14_b160_clean", &ed, 14, 160, true, &m_clean);

        // Strong conservation + clean h2/O/skim calibration
        let mut m_cons = m_clean.clone();
        m_cons.linear_w[0] = 1.60 / m_cons.inv_std[0];    // rem_g ~ 1.60 pieces/line
        m_cons.linear_w[9] = 0.14 / m_cons.inv_std[9];    // stack_cells ~ 0.14/cell
        m_cons.linear_w[11] = 0.06 / m_cons.inv_std[11];  // stack_cells_above1
        m_cons.linear_w[36] = 0.30 / m_cons.inv_std[36];  // h2_blocks
        m_cons.linear_w[39] = -0.20 / m_cons.inv_std[39]; // h2_open
        m_cons.linear_w[57] = -0.25 / m_cons.inv_std[57]; // near_full_9
        m_cons.linear_w[92] = -0.15 / m_cons.inv_std[92]; // clean_skim_rows
        m_cons.prepare_fast();
        run_oracle_suite("OR_p14_b160_cons", &ed, 14, 160, true, &m_cons);

        let m_r4 = load_model(concat!(env!("CARGO_MANIFEST_DIR"), "/../models/crn_r4_gbox.txt"));
        run_oracle_suite("OR_p14_b160_r4gbox", &ed, 14, 160, true, &m_r4);
    } else if mode == "spatial_oracle5" {
        let model_path = args.get(2).map(|s| s.as_str()).unwrap_or("models/spatial_v1.txt");
        let sm = SpatialValueModel::load_from_file(model_path).expect("load spatial model");
        println!("=== Spatial Oracle Suite ({}) (seeds 101..=105) ===", model_path);
        let ed = collect_exact_seeds(&seeds5);
        run_oracle_suite("OR_p6_EXCH_b120_sp", &ed, 6, 120, true, &sm);
        run_oracle_suite("OR_p14_EXCH_b160_sp", &ed, 14, 160, true, &sm);
        run_oracle_suite("OR_p20_EXCH_b240_sp", &ed, 20, 240, true, &sm);
    } else if mode == "spatial_probe" {
        let model_path = args.get(2).map(|s| s.as_str()).unwrap_or("models/spatial_pilot20_v2.txt");
        let sm = SpatialValueModel::load_from_file(model_path).expect("load spatial model");
        let mut sm_lin = sm.clone();
        sm_lin.b3 = 0.0;
        sm_lin.w3 = [0.0; blockfish::ai::spatial_val::SPATIAL_H2];
        sm_lin.prepare_fast();
        println!("=== Spatial Probe ({}) (seeds 101..=105) ===", model_path);
        let ed = collect_exact_seeds(&seeds5);
        run_oracle_suite("OR_p6_b120_h2nog0", &ed, 6, 120, true, &m_h2_nog0);
        run_oracle_suite("OR_p6_b120_sp_lin", &ed, 6, 120, true, &sm_lin);
        run_oracle_suite("OR_p6_b120_sp_full", &ed, 6, 120, true, &sm);

        // Quick 1-seed step comparison on seed 101 OR_p6
        let ed1 = &ed[0];
        let mut game_cfg = BSConfig::default();
        game_cfg.garbage.total_lines = Some(100);
        game_cfg.prng_seed = Some(ed1.seed);
        let mut stacker = Stacker::new(Ruleset::guideline().into(), game_cfg);
        let fst = FastShapeTable::srs();
        let mut pgen = FastPlaceGen::new();
        let mut ai_in = FastAI::new(BeamConfig::default());
        let mut root_places = Vec::with_capacity(64);
        let mut ds_total = 0usize;
        let mut q_pos = 0usize;
        let mut act = [0u16; blockfish::ai::spatial_val::SPARSE_DIM];
        let mut den = [0.0f32; blockfish::ai::spatial_val::DENSE_DIM];
        println!("\n=== 1-ply breakdown on first 12 moves of seed 101 (sp_full vs h2nog0) ===");
        for step in 0..15usize {
            if ds_total >= 100 || stacker.current_piece_type().is_none() { break; }
            let snap = stacker.snapshot().unwrap();
            let g_rows = garbage_rows(&stacker);
            let root_board = FastBoard::from_basic_matrix(&snap.matrix, g_rows);
            let rem_g0 = 100usize.saturating_sub(ds_total) as u16;
            let hold0 = snap.hold.map(|cc| color_to_idx(cc) as u8);
            let q = &ed1.pieces[q_pos..(q_pos + 6).min(ed1.pieces.len())];
            let cur0 = Some(q[0]);
            let alt0 = hold0.or_else(|| q.get(1).copied());
            pgen.generate_for_state(&fst, &root_board, cur0, alt0, &mut root_places);
            let mut best_sp = (i64::MAX, 0usize, 0.0f32, 0.0f32);
            let mut best_h2 = (i64::MAX, 0usize);
            for (idx, &fp) in root_places.iter().enumerate() {
                let mut b = root_board;
                let o = &fst.shapes[fp.color_idx as usize].orients[fp.tf.2 as usize];
                let (lines, ds, _) = b.place_and_sift(o, fp.norm_i, fp.norm_j);
                let rem_g = rem_g0.saturating_sub(ds as u16);
                if rem_g > 0 {
                    let target_g: usize = if lines > 0 { 4 } else { 9 };
                    let g = b.garbage_rows as usize;
                    let unsp = (rem_g as usize).saturating_sub(g);
                    let num = target_g.saturating_sub(g).min(unsp);
                    if num > 0 && b.rows as usize + num < 32 {
                        let sp_so_far = 100usize - unsp;
                        for r in (0..b.rows as usize).rev() { b.data[r + num] = b.data[r]; }
                        for k in 0..num {
                            b.data[num - 1 - k] = 0xFC00 | (0x3FF & !(1u16 << ed1.holes[sp_so_far + k]));
                        }
                        b.rows += num as u8;
                        b.garbage_rows += num as u8;
                    }
                }
                let held = if hold0.is_some() { if fp.did_hold { cur0 } else { hold0 } } else if fp.did_hold { cur0 } else { None };
                let qc = if hold0.is_none() && fp.did_hold { 2 } else { 1 };
                let in_h = held.or_else(|| q.get(qc).copied());
                let n_act = blockfish::ai::spatial_val::extract_spatial_features(&b, rem_g, in_h, 0x7F, &mut act, &mut den);
                let p_full = sm.predict_extracted(&act[..n_act], &den);
                let p_lin = sm_lin.predict_extracted(&act[..n_act], &den);
                let sc_sp = sm.evaluate_board_with_bag(&b, rem_g, in_h, 0x7F);
                let sc_h2 = m_h2_nog0.evaluate_board_with_bag(&b, rem_g, in_h, 0x7F);
                if sc_sp < best_sp.0 { best_sp = (sc_sp, idx, p_lin, p_full - p_lin); }
                if sc_h2 < best_h2.0 { best_h2 = (sc_h2, idx); }
            }
            let fp = root_places[best_sp.1];
            let fp_h2 = root_places[best_h2.1];
            let (_, bo, _, _) = h1_info(&stacker);
            let mut inputs = ai_in.inputs_for_place(&snap.matrix, fp).unwrap();
            inputs.pop();
            if fp.did_hold { if hold0.is_none() { q_pos += 2; } else { q_pos += 1; } } else { q_pos += 1; }
            stacker.run(inputs);
            let (lines, ds) = stacker.hard_drop();
            ds_total += ds;
            let (_, new_bo, _, cells_ab) = h1_info(&stacker);
            println!("step={:2} bo={}->{:<2} cells={:2} ds={} l={} | sp(lin={:6.2} mlp={:+6.2}) same_as_h2={}",
                step + 1, bo, new_bo, cells_ab, ds, lines, best_sp.2, best_sp.3, best_sp.1 == best_h2.1);
            let _ = fp_h2;
        }
    } else if mode == "legal20" {
        println!("=== Legal Held-out Dev20 (seeds 201..=220) ===");
        run_legal_suite("LEGAL_h2nog0_p5_b120", &seeds20, &cfg_legal);
    } else if mode == "or_ceil2_h14" {
        let seeds2: Vec<u64> = vec![101, 102];
        let ed2 = collect_exact_seeds(&seeds2);
        println!("=== Oracle H=14 Ceiling Probe (seeds 101..=102) ===");
        run_oracle_suite("OR_p14_b160_h2nog0", &ed2, 14, 160, true, &m_h2_nog0);
        run_oracle_adv("OR_p14_b320_prog_strat", &ed2, 14, 320, true, true, None, &m_h2_nog0);

        let mut m_rg165 = m_h2_nog0.clone();
        m_rg165.linear_w[0] = 1.65 / m_rg165.inv_std[0];
        m_rg165.prepare_fast();
        run_oracle_adv("OR_p14_b240_rg165_keepO", &ed2, 14, 240, true, true, None, &m_rg165);

        let mut m_rg255 = m_h2_nog0.clone();
        m_rg255.linear_w[0] = 2.55 / m_rg255.inv_std[0];
        m_rg255.prepare_fast();
        run_oracle_adv("OR_p14_b240_rg255", &ed2, 14, 240, true, true, None, &m_rg255);

        let ew_def = EvalWeights::default();
        run_oracle_adv("OR_p14_b240_ew_default", &ed2, 14, 240, true, true, None, &ew_def);
    } else if mode == "or_ceil2_clean" {
        let seeds2: Vec<u64> = vec![101, 102];
        let ed2 = collect_exact_seeds(&seeds2);
        println!("=== Oracle Clean Evaluator & Width Probe (seeds 101..=102, NO prog_strat) ===");
        // Pure clean physics calibration on top of m_h2_nog0:
        // Lower rem_g from 2.12 -> 1.60 (true marginal cost of 1 garbage line)
        // Raise stack_cells 0.03 -> 0.12, h2_blocks 0 -> 0.25, h2_open 0 -> -0.18,
        // covered_holes 0.13 -> 0.45, h1_blocks 0.026 -> 0.22, near_full_9 0 -> -0.18
        let mut m_clean160 = m_h2_nog0.clone();
        m_clean160.linear_w[0] = 1.60 / m_clean160.inv_std[0];   // rem_g
        m_clean160.linear_w[9] = 0.12 / m_clean160.inv_std[9];   // stack_cells
        m_clean160.linear_w[11] = 0.05 / m_clean160.inv_std[11]; // stack_cells_above1
        m_clean160.linear_w[20] = 0.45 / m_clean160.inv_std[20]; // covered_holes
        m_clean160.linear_w[24] = 0.22 / m_clean160.inv_std[24]; // h1_blocks
        m_clean160.linear_w[36] = 0.25 / m_clean160.inv_std[36]; // h2_blocks
        m_clean160.linear_w[39] = -0.18 / m_clean160.inv_std[39];// h2_open
        m_clean160.linear_w[57] = -0.18 / m_clean160.inv_std[57];// near_full_9
        m_clean160.linear_w[88] = 0.20 / m_clean160.inv_std[88]; // h2_only_row_g_blocked
        m_clean160.prepare_fast();

        let mut m_clean145 = m_clean160.clone();
        m_clean145.linear_w[0] = 1.45 / m_clean145.inv_std[0];
        m_clean145.prepare_fast();

        run_oracle_adv("OR_p14_b240_h2nog0", &ed2, 14, 240, false, false, None, &m_h2_nog0);
        run_oracle_adv("OR_p14_b240_clean160", &ed2, 14, 240, false, false, None, &m_clean160);
        run_oracle_adv("OR_p14_b240_clean145", &ed2, 14, 240, false, false, None, &m_clean145);
        run_oracle_adv("OR_p20_b240_clean160", &ed2, 20, 240, false, false, None, &m_clean160);
    } else if mode == "or_ceil2_probe" {
        let seeds2: Vec<u64> = vec![101, 102];
        let ed2 = collect_exact_seeds(&seeds2);
        println!("=== Oracle Pure rem_g & prog_mpr Probe (seeds 101..=102) ===");
        let mut m_rg170 = m_h2_nog0.clone();
        m_rg170.linear_w[0] = 1.70 / m_rg170.inv_std[0];
        m_rg170.prepare_fast();

        let mut m_rg260 = m_h2_nog0.clone();
        m_rg260.linear_w[0] = 2.60 / m_rg260.inv_std[0];
        m_rg260.prepare_fast();

        let mut m_rg320 = m_h2_nog0.clone();
        m_rg320.linear_w[0] = 3.20 / m_rg320.inv_std[0];
        m_rg320.prepare_fast();

        run_oracle_adv("OR_p14_b240_rg170_only", &ed2, 14, 240, false, false, None, &m_rg170);
        run_oracle_adv("OR_p14_b240_rg260_only", &ed2, 14, 240, false, false, None, &m_rg260);
        run_oracle_adv("OR_p14_b240_rg320_only", &ed2, 14, 240, false, false, None, &m_rg320);
        run_oracle_adv("OR_p14_b240_prog_only", &ed2, 14, 240, true, false, None, &m_h2_nog0);
    } else if mode == "or_ceil2_h30" {
        let seeds2: Vec<u64> = vec![101, 102];
        let ed2 = collect_exact_seeds(&seeds2);
        println!("=== Oracle H=20 mpr24 & H=30 Probe (seeds 101..=102) ===");
        run_oracle_adv("OR_p20_b360_mpr24", &ed2, 20, 360, false, false, Some(24), &m_h2_nog0);
        run_oracle_adv("OR_p30_b240_mpr40", &ed2, 30, 240, false, false, Some(40), &m_h2_nog0);
    } else if mode == "or_ceil2_deep" {
        let seeds2: Vec<u64> = vec![101, 102];
        let ed2 = collect_exact_seeds(&seeds2);
        println!("=== Oracle Deep Ceiling Probe (seeds 101..=102, NO prog_strat) ===");
        run_oracle_adv("OR_p20_b500_h2nog0", &ed2, 20, 500, false, false, None, &m_h2_nog0);
    } else if mode == "or_ceil_gh" {
        let ed5 = collect_exact_seeds(&seeds5);
        println!("=== Oracle Ceiling Suite (seeds 101..=105) ===");
        run_oracle_adv("OR_p20_b240_h2nog0", &ed5, 20, 240, false, false, None, &m_h2_nog0);
        run_oracle_adv("OR_p20_b600_h2nog0", &ed5, 20, 600, false, false, None, &m_h2_nog0);
        run_oracle_adv("OR_p30_b400_h2nog0", &ed5, 30, 400, false, false, Some(64), &m_h2_nog0);
    } else if mode == "or_diag_decomp" {
        let seeds2: Vec<u64> = vec![101, 102];
        let ed2 = collect_exact_seeds(&seeds2);
        println!("=== Causal Decomposition of Oracle 176 Ceiling (seeds 101..=102) ===");
        println!("  (ref 1_base_p14_b240 = 178.50 [179, 178])");
        run_oracle_decomp("2_pareto_p14_b240", &ed2, 14, 240, true, false, false, &m_h2_nog0);
        run_oracle_decomp("3_7bag_roll_p14_b240", &ed2, 14, 240, true, true, false, &m_h2_nog0);
        run_oracle_decomp("4_ub_roll_p14_b240", &ed2, 14, 240, true, false, true, &m_h2_nog0);
    } else if mode == "or_agg_probe" {
        let seeds2: Vec<u64> = vec![101, 102];
        let ed2 = collect_exact_seeds(&seeds2);
        println!("=== Oracle H=20 Root Aggregation & Annealing Probe (seeds 101..=102) ===");
        run_oracle_agg("OR_p20_b240_top1", &ed2, 20, 240, 1, false, 0, &m_h2_nog0);
        run_oracle_agg("OR_p20_b240_top3mean", &ed2, 20, 240, 3, false, 0, &m_h2_nog0);
        run_oracle_agg("OR_p20_b240_anneal_top3", &ed2, 20, 240, 3, true, 0, &m_h2_nog0);
    } else if mode == "or_deep_roll" {
        let which = args.get(2).map(|s| s.as_str()).unwrap_or("true36");
        let seeds2: Vec<u64> = vec![101, 102];
        let ed2 = collect_exact_seeds(&seeds2);
        let mut m_no_o = m_h2_nog0.clone();
        m_no_o.linear_w[63] = 0.0;
        m_no_o.prepare_fast();

        let mut m_term_true = m_no_o.clone();
        m_term_true.linear_w[0] = 1.68 / m_term_true.inv_std[0];
        m_term_true.linear_w[9] = 0.12 / m_term_true.inv_std[9];
        m_term_true.linear_w[11] = 0.04 / m_term_true.inv_std[11];
        m_term_true.linear_w[20] = 0.40 / m_term_true.inv_std[20];
        m_term_true.linear_w[24] = 0.15 / m_term_true.inv_std[24];
        m_term_true.linear_w[36] = 0.25 / m_term_true.inv_std[36];
        m_term_true.linear_w[39] = -0.15 / m_term_true.inv_std[39];
        m_term_true.linear_w[57] = -0.15 / m_term_true.inv_std[57];
        m_term_true.linear_w[88] = 0.20 / m_term_true.inv_std[88];
        m_term_true.prepare_fast();

        if which == "true36" {
            run_oracle_deep_roll("OR_p14_roll36_true_step", &ed2, 14, 240, 40, 1, 36, 8, &m_term_true, &m_term_true);
        } else if which == "mix36" {
            run_oracle_deep_roll("OR_p14_roll36_mix", &ed2, 14, 240, 40, 1, 36, 8, &m_no_o, &m_term_true);
        } else if which == "true20" {
            run_oracle_deep_roll("OR_p14_roll20_true", &ed2, 14, 240, 40, 1, 20, 12, &m_term_true, &m_term_true);
        } else if which == "ann_p20" {
            run_oracle_deep_roll("OR_p20_ann35_mterm", &ed2, 20, 240, 35, 3, 0, 8, &m_term_true, &m_term_true);
        } else if which == "h20_pen_h2" {
            let mut m_h2p = m_h2_nog0.clone();
            m_h2p.linear_w[20] = 0.25 / m_h2p.inv_std[20];  // covered_holes: 0.13 -> 0.25
            m_h2p.linear_w[36] = 0.25 / m_h2p.inv_std[36];  // h2_blocks: 0 -> 0.25
            m_h2p.linear_w[39] = -0.15 / m_h2p.inv_std[39]; // h2_open: 0 -> -0.15
            m_h2p.linear_w[88] = 0.20 / m_h2p.inv_std[88];  // h2_only_row_g_blocked: -0.13 -> +0.20
            m_h2p.prepare_fast();
            run_oracle_deep_roll("OR_p20_b240_ann35_h2pen", &ed2, 20, 240, 35, 3, 0, 8, &m_h2p, &m_h2p);
        } else if which == "h30_ann" {
            run_oracle_deep_roll("OR_p30_b360_ann50", &ed2, 30, 360, 50, 3, 0, 8, &m_h2_nog0, &m_h2_nog0);
        }
    } else if mode == "spatial_or_p20" {
        let model_path = args.get(2).map(|s| s.as_str()).unwrap_or("models/spatial_r1.txt");
        let sm = SpatialValueModel::load_from_file(model_path).expect("load spatial model");
        let seeds2: Vec<u64> = vec![101, 102];
        let ed2 = collect_exact_seeds(&seeds2);
        println!("=== Spatial OR_p20 ({}) (seeds 101..=102) ===", model_path);
        run_oracle_agg("OR_p20_b240_sp_top1", &ed2, 20, 240, 1, false, 0, &sm);
        run_oracle_agg("OR_p20_b240_sp_ann_top3", &ed2, 20, 240, 3, true, 0, &sm);
    } else if mode == "or_break_probe" {
        let seeds2: Vec<u64> = vec![101, 102];
        let ed2 = collect_exact_seeds(&seeds2);
        println!("=== Oracle Ceiling Break Probe (seeds 101..=102) ===");
        let mut m_no_o = m_h2_nog0.clone();
        m_no_o.linear_w[63] = 0.0; // remove hand_O bias
        m_no_o.prepare_fast();
        run_oracle_agg("1_p20_b240_noO", &ed2, 20, 240, 1, false, 0, &m_no_o);
        run_oracle_agg("2_p20_b240_noO_ann_top3", &ed2, 20, 240, 3, true, 0, &m_no_o);
        run_oracle_deep_roll("3_p14_b240_roll24_bw12", &ed2, 14, 240, 0, 1, 24, 12, &m_h2_nog0, &m_h2_nog0);
        run_oracle_deep_roll("4_p20_b240_roll18_bw10", &ed2, 20, 240, 0, 1, 18, 10, &m_h2_nog0, &m_h2_nog0);
    } else if mode == "or_break_probe2" {
        let which = args.get(2).map(|s| s.as_str()).unwrap_or("all");
        let seeds2: Vec<u64> = vec![101, 102];
        let ed2 = collect_exact_seeds(&seeds2);
        println!("=== Oracle Trajectory & Clean-Sign Probe (seeds 101..=102, which={}) ===", which);
        let mut m_no_o = m_h2_nog0.clone();
        m_no_o.linear_w[63] = 0.0;
        m_no_o.prepare_fast();

        let mut m_fix = m_no_o.clone();
        m_fix.linear_w[36] = 0.08 / m_fix.inv_std[36];  // h2_blocks: 0 -> +0.08
        m_fix.linear_w[39] = -0.08 / m_fix.inv_std[39]; // h2_open: 0 -> -0.08
        m_fix.linear_w[57] = -0.10 / m_fix.inv_std[57]; // near_full_9: 0 -> -0.10
        m_fix.linear_w[88] = 0.16 / m_fix.inv_std[88];  // h2_only_row_g_blocked: -0.127 -> +0.16
        m_fix.prepare_fast();

        if which == "all" || which == "1" {
            run_oracle_agg("1_p20_b240_noO_early12", &ed2, 20, 240, 1, false, 12, &m_no_o);
        }
        if which == "all" || which == "2" {
            run_oracle_agg("2_p20_b240_noO_h2fix", &ed2, 20, 240, 1, false, 0, &m_fix);
        }
        if which == "all" || which == "3" {
            run_oracle_agg("3_p20_b240_h2fix_early12", &ed2, 20, 240, 1, false, 12, &m_fix);
        }
        if which == "all" || which == "4" {
            run_oracle_agg("4_p20_b480_noO", &ed2, 20, 480, 1, false, 0, &m_no_o);
        }
    } else if mode == "whole_game" {
        let which = args.get(2).map(|s| s.as_str()).unwrap_or("all");
        let n_seeds: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(2);
        let seeds: Vec<u64> = (101..101 + n_seeds as u64).collect();
        let ed = collect_exact_seeds(&seeds);
        println!("=== Whole-Game Exact Oracle Beam (seeds 101..={}, which={}) ===", 100 + n_seeds, which);
        let mut m_clean_in_bucket = m_h2_nog0.clone();
        // Inside same rem_g bucket, rank purely by clean/playable structure!
        m_clean_in_bucket.linear_w[0] = 0.0; // rem_g handled by bucketing
        m_clean_in_bucket.linear_w[9] = 0.09 / m_clean_in_bucket.inv_std[9];   // stack_cells
        m_clean_in_bucket.linear_w[11] = 0.04 / m_clean_in_bucket.inv_std[11]; // stack_cells_above1
        m_clean_in_bucket.linear_w[20] = 0.38 / m_clean_in_bucket.inv_std[20]; // covered_holes
        m_clean_in_bucket.linear_w[24] = 0.25 / m_clean_in_bucket.inv_std[24]; // h1_blocks
        m_clean_in_bucket.linear_w[36] = 0.20 / m_clean_in_bucket.inv_std[36]; // h2_blocks
        m_clean_in_bucket.linear_w[39] = -0.18 / m_clean_in_bucket.inv_std[39];// h2_open
        m_clean_in_bucket.linear_w[57] = -0.25 / m_clean_in_bucket.inv_std[57];// near_full_9
        m_clean_in_bucket.linear_w[59] = -0.25 / m_clean_in_bucket.inv_std[59];// row_g_9_aligned
        m_clean_in_bucket.linear_w[63] = 0.0; // no artificial O bias
        m_clean_in_bucket.linear_w[88] = 0.22 / m_clean_in_bucket.inv_std[88]; // h2_only_row_g_blocked
        m_clean_in_bucket.linear_w[92] = -0.20 / m_clean_in_bucket.inv_std[92];// clean_skim_rows
        m_clean_in_bucket.prepare_fast();

        if which == "all" || which == "1" {
            run_whole_game_beam("WG_b1000_h2nog0", &ed, 1000, 0, &m_h2_nog0);
        }
        if which == "all" || which == "2" {
            run_whole_game_beam("WG_b1400_buck_h2", &ed, 1400, 200, &m_h2_nog0);
        }
        if which == "all" || which == "3" {
            run_whole_game_beam("WG_b1400_buck_clean", &ed, 1400, 200, &m_clean_in_bucket);
        }
        if which == "b4000" {
            run_whole_game_beam("WG_b4000_buck_clean", &ed, 4000, 500, &m_clean_in_bucket);
        }
    } else if mode == "whole_game2" {
        let which = args.get(2).map(|s| s.as_str()).unwrap_or("all");
        let n_seeds: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(2);
        let seeds: Vec<u64> = (101..101 + n_seeds as u64).collect();
        let ed = collect_exact_seeds(&seeds);
        println!("=== Whole-Game Conservation & Wide-Bucket Probe (seeds 101..={}, which={}) ===", 100 + n_seeds, which);

        // Conservation-law clean model: zero/negative penalty on low clean stack_cells
        // (lower K_so_far at fixed rem_g!), penalize only upper_cells/stack_over2_sq/holes/h12_blocks
        let mut m_cons_global = m_h2_nog0.clone();
        m_cons_global.linear_w[0] = 1.65 / m_cons_global.inv_std[0];    // rem_g
        m_cons_global.linear_w[7] = 0.18 / m_cons_global.inv_std[7];    // pe_net
        m_cons_global.linear_w[9] = 0.0;                                // stack_cells: NO blind penalty!
        m_cons_global.linear_w[11] = 0.02 / m_cons_global.inv_std[11];  // stack_cells_above1
        m_cons_global.linear_w[13] = 0.04 / m_cons_global.inv_std[13];  // upper_cells
        m_cons_global.linear_w[18] = 0.03 / m_cons_global.inv_std[18];  // stack_over2_sq
        m_cons_global.linear_w[20] = 0.45 / m_cons_global.inv_std[20];  // covered_holes
        m_cons_global.linear_w[24] = 0.25 / m_cons_global.inv_std[24];  // h1_blocks
        m_cons_global.linear_w[36] = 0.22 / m_cons_global.inv_std[36];  // h2_blocks
        m_cons_global.linear_w[39] = -0.18 / m_cons_global.inv_std[39]; // h2_open
        m_cons_global.linear_w[57] = -0.20 / m_cons_global.inv_std[57]; // near_full_9
        m_cons_global.linear_w[59] = -0.25 / m_cons_global.inv_std[59]; // row_g_9_aligned
        m_cons_global.linear_w[63] = 0.0;                               // no artificial O bias
        m_cons_global.linear_w[88] = 0.20 / m_cons_global.inv_std[88];  // h2_only_row_g_blocked
        m_cons_global.linear_w[92] = -0.18 / m_cons_global.inv_std[92]; // clean_skim_rows
        m_cons_global.prepare_fast();

        let mut m_cons_buck = m_cons_global.clone();
        m_cons_buck.linear_w[0] = 0.0;                                  // rem_g handled by bucket
        m_cons_buck.linear_w[9] = -0.06 / m_cons_buck.inv_std[9];       // prefer keeping clean stack cells (lower K_so_far!)
        m_cons_buck.linear_w[10] = -0.06 / m_cons_buck.inv_std[10];     // cells on row g
        m_cons_buck.linear_w[13] = 0.05 / m_cons_buck.inv_std[13];      // penalize high upper_cells
        m_cons_buck.prepare_fast();

        let mut m_h2fix_global = m_h2_nog0.clone();
        m_h2fix_global.linear_w[63] = 0.0;
        m_h2fix_global.linear_w[36] = 0.14 / m_h2fix_global.inv_std[36];
        m_h2fix_global.linear_w[39] = -0.12 / m_h2fix_global.inv_std[39];
        m_h2fix_global.linear_w[88] = 0.18 / m_h2fix_global.inv_std[88];
        m_h2fix_global.prepare_fast();

        if which == "all" || which == "1" {
            run_whole_game_beam("WG_b1000_h2fix", &ed, 1000, 0, &m_h2fix_global);
        }
        if which == "all" || which == "2" {
            run_whole_game_beam("WG_b1000_cons_glob", &ed, 1000, 0, &m_cons_global);
        }
        if which == "all" || which == "3" {
            run_whole_game_beam("WG_b2000_buck16_cons", &ed, 2000, 125, &m_cons_buck);
        }
        if which == "b2000_h2" {
            run_whole_game_beam("WG_b2000_h2nog0", &ed, 2000, 0, &m_h2_nog0);
        }
        if which == "b4000_h2" {
            run_whole_game_beam("WG_b4000_h2nog0", &ed, 4000, 0, &m_h2_nog0);
        }
        if which == "b4000_cons" {
            run_whole_game_beam("WG_b4000_cons_glob", &ed, 4000, 0, &m_cons_global);
        }
        if which == "b4000_h2fix" {
            run_whole_game_beam("WG_b4000_h2fix", &ed, 4000, 0, &m_h2fix_global);
        }
    } else if mode == "cons_legal5" {
        let mut m_cons = m_h2_nog0.clone();
        m_cons.linear_w[0] = 1.65 / m_cons.inv_std[0];
        m_cons.linear_w[7] = 0.18 / m_cons.inv_std[7];
        m_cons.linear_w[9] = 0.0;
        m_cons.linear_w[11] = 0.02 / m_cons.inv_std[11];
        m_cons.linear_w[13] = 0.04 / m_cons.inv_std[13];
        m_cons.linear_w[18] = 0.03 / m_cons.inv_std[18];
        m_cons.linear_w[20] = 0.45 / m_cons.inv_std[20];
        m_cons.linear_w[24] = 0.25 / m_cons.inv_std[24];
        m_cons.linear_w[36] = 0.22 / m_cons.inv_std[36];
        m_cons.linear_w[39] = -0.18 / m_cons.inv_std[39];
        m_cons.linear_w[57] = -0.20 / m_cons.inv_std[57];
        m_cons.linear_w[59] = -0.25 / m_cons.inv_std[59];
        m_cons.linear_w[63] = 0.0;
        m_cons.linear_w[88] = 0.20 / m_cons.inv_std[88];
        m_cons.linear_w[92] = -0.18 / m_cons.inv_std[92];
        m_cons.prepare_fast();

        let which = args.get(2).map(|s| s.as_str()).unwrap_or("all");
        if which == "all" || which == "legal" {
            let mut c_cons = cfg_legal.clone();
            c_cons.value_model = Some(Arc::new(m_cons.clone()));
            run_legal_suite("LEGAL_cons_p5_b120", &seeds5, &c_cons);

            let mut c_up = c_cons.clone();
            c_up.bag_det_extend = true;
            c_up.endgame_solver = true;
            c_up.simulate_cheese_spawn = true;
            run_legal_suite("LEGAL_cons_bdet_end_sim", &seeds5, &c_up);
        }
        if which == "all" || which == "oracle" {
            let ed5 = collect_exact_seeds(&seeds5);
            run_oracle_agg("OR_p6_b120_cons", &ed5, 6, 120, 1, false, 0, &m_cons);
            run_oracle_agg("OR_p14_b240_cons", &ed5, 14, 240, 1, false, 0, &m_cons);
        }
    }
}

fn run_whole_game_beam<M: DiagEvaluator>(
    label: &str,
    exact_data: &[ExactSeedData],
    beam_w: usize,
    bucket_cap: usize,
    mdl: &M,
) {
    use blockfish::ai::fast::FastPlace;
    type FxBuild = std::hash::BuildHasherDefault<blockfish::ai::beam::FxHasher>;
    #[derive(Clone, Copy)]
    struct WNode {
        board: FastBoard,
        held: Option<u8>,
        q_cursor: u16,
        rem_g: u16,
        score: i64,
        parent: u32,
        fp_idx: u32,
    }
    #[derive(Clone, Copy, Eq, PartialEq, Hash)]
    struct WKey {
        board: FastBoard,
        held: Option<u8>,
        rem_g: u16,
    }

    let t0 = Instant::now();
    let fst = FastShapeTable::srs();
    let spawn_exact_cheese = |b: &mut FastBoard, rem_g: u16, lines_cleared: u8, holes: &[u16]| -> bool {
        let target_g: usize = if lines_cleared > 0 { 4 } else { 9 };
        let g = b.garbage_rows as usize;
        let unspawned = (rem_g as usize).saturating_sub(g);
        let num = target_g.saturating_sub(g).min(unspawned);
        if num == 0 { return true; }
        let old_rows = b.rows as usize;
        if old_rows + num >= 32 { return false; }
        let spawned_so_far = 100usize - unspawned;
        if spawned_so_far + num > holes.len() { return false; }
        for r in (0..old_rows).rev() { b.data[r + num] = b.data[r]; }
        for k in 0..num {
            let hole = holes[spawned_so_far + k];
            b.data[num - 1 - k] = 0xFC00 | (0x3FF & !(1u16 << hole));
        }
        b.rows = (old_rows + num) as u8;
        b.garbage_rows = (g + num) as u8;
        true
    };

    let solve_one = |ed: &ExactSeedData| -> GameStat {
        let mut game_cfg = BSConfig::default();
        game_cfg.garbage.total_lines = Some(100);
        game_cfg.prng_seed = Some(ed.seed);
        let mut stacker = Stacker::new(Ruleset::guideline().into(), game_cfg);
        let snap0 = stacker.snapshot().unwrap();
        let g0 = garbage_rows(&stacker);
        let b0 = FastBoard::from_basic_matrix(&snap0.matrix, g0);
        let mut pgen = FastPlaceGen::new();
        let mut tmp_places: Vec<FastPlace> = Vec::with_capacity(64);
        let mut cur_beam: Vec<WNode> = Vec::with_capacity(beam_w);
        let mut next_cands: Vec<WNode> = Vec::with_capacity(beam_w * 32);
        let mut seen: HashSet<WKey, FxBuild> = HashSet::with_capacity_and_hasher(beam_w * 4, FxBuild::default());
        let mut hist_parents: Vec<Vec<(u32, FastPlace)>> = Vec::with_capacity(240);

        cur_beam.push(WNode {
            board: b0,
            held: None,
            q_cursor: 0,
            rem_g: 100,
            score: 0,
            parent: u32::MAX,
            fp_idx: 0,
        });

        let mut win_moves: Option<Vec<FastPlace>> = None;
        let q_len = ed.pieces.len();

        for step in 1..=260usize {
            next_cands.clear();
            let mut step_places: Vec<FastPlace> = Vec::with_capacity(cur_beam.len() * 24);
            let mut best_win: Option<(u32, FastPlace, u8)> = None;

            for (ni, &node) in cur_beam.iter().enumerate() {
                if node.rem_g == 0 { continue; }
                let qc = node.q_cursor as usize;
                let (c_nh, c_h) = if node.held.is_some() {
                    (ed.pieces.get(qc).copied(), node.held)
                } else {
                    (ed.pieces.get(qc).copied(), ed.pieces.get(qc + 1).copied())
                };
                if c_nh.is_none() && c_h.is_none() { continue; }
                pgen.generate_for_state(&fst, &node.board, c_nh, c_h, &mut tmp_places);
                for &fp in &tmp_places {
                    let mut b = node.board;
                    let o = &fst.shapes[fp.color_idx as usize].orients[fp.tf.2 as usize];
                    let (lines, ds, _) = b.place_and_sift(o, fp.norm_i, fp.norm_j);
                    let rem_g = node.rem_g.saturating_sub(ds as u16);
                    if rem_g == 0 && b.rows < 20 {
                        if best_win.map_or(true, |w| b.rows < w.2) {
                            best_win = Some((ni as u32, fp, b.rows));
                        }
                        continue;
                    }
                    if !spawn_exact_cheese(&mut b, rem_g, lines, &ed.holes) { continue; }
                    let (held, nqc) = if node.held.is_some() {
                        if fp.did_hold { (c_nh, (qc + 1).min(q_len)) } else { (node.held, (qc + 1).min(q_len)) }
                    } else if fp.did_hold {
                        (c_nh, (qc + 2).min(q_len))
                    } else {
                        (None, (qc + 1).min(q_len))
                    };
                    let in_h = held.or_else(|| ed.pieces.get(nqc).copied());
                    let sc = mdl.eval_b(&b, rem_g, in_h, 0x7F);
                    let fpi = step_places.len() as u32;
                    step_places.push(fp);
                    next_cands.push(WNode {
                        board: b,
                        held,
                        q_cursor: nqc as u16,
                        rem_g,
                        score: sc,
                        parent: ni as u32,
                        fp_idx: fpi,
                    });
                }
            }

            if let Some((mut p_idx, last_fp, _)) = best_win {
                let mut rev = vec![last_fp];
                let mut s = step - 1;
                while s > 0 && p_idx != u32::MAX {
                    let (prev_p, fp) = hist_parents[s - 1][p_idx as usize];
                    rev.push(fp);
                    p_idx = prev_p;
                    s -= 1;
                }
                rev.reverse();
                win_moves = Some(rev);
                break;
            }
            if next_cands.is_empty() { break; }

            cur_beam.clear();
            seen.clear();
            if bucket_cap > 0 {
                let min_rg = next_cands.iter().map(|n| n.rem_g).min().unwrap_or(0);
                next_cands.sort_unstable_by_key(|n| (n.rem_g, n.score));
                let mut buck_cnt = [0usize; 20];
                for &ch in &next_cands {
                    let bi = ch.rem_g.saturating_sub(min_rg) as usize;
                    if bi < 20 && buck_cnt[bi] < bucket_cap && cur_beam.len() < beam_w {
                        if seen.insert(WKey { board: ch.board, held: ch.held, rem_g: ch.rem_g }) {
                            buck_cnt[bi] += 1;
                            cur_beam.push(ch);
                        }
                    }
                }
                if cur_beam.len() < beam_w {
                    next_cands.sort_unstable_by_key(|n| (n.score, n.rem_g));
                    for &ch in &next_cands {
                        if cur_beam.len() >= beam_w { break; }
                        if seen.insert(WKey { board: ch.board, held: ch.held, rem_g: ch.rem_g }) {
                            cur_beam.push(ch);
                        }
                    }
                }
            } else {
                next_cands.sort_unstable_by_key(|n| n.score);
                for &ch in &next_cands {
                    if cur_beam.len() >= beam_w { break; }
                    if seen.insert(WKey { board: ch.board, held: ch.held, rem_g: ch.rem_g }) {
                        cur_beam.push(ch);
                    }
                }
            }

            let mut step_hist = Vec::with_capacity(cur_beam.len());
            for n in &cur_beam {
                step_hist.push((n.parent, step_places[n.fp_idx as usize]));
            }
            hist_parents.push(step_hist);
        }

        let mut st = GameStat::default();
        if let Some(moves) = win_moves {
            let mut ai_in = FastAI::new(BeamConfig::default());
            let mut ds_total = 0usize;
            for fp in moves {
                let snap = stacker.snapshot().unwrap();
                let g_rows = garbage_rows(&stacker);
                let rem_g0 = 100usize.saturating_sub(ds_total) as u16;
                let (_, blocks_over, chute_min, cells_above) = h1_info(&stacker);
                st.cells_sum += cells_above;
                let cur_ch = snap.queue[0].as_char();
                let hold_ch = snap.hold.or_else(|| snap.queue.get(1).copied()).map(|cc| cc.as_char());
                let mut inputs = match ai_in.inputs_for_place(&snap.matrix, fp) {
                    Some(inp) => inp,
                    None => { st.topout = true; st.pcs = 400; return st; }
                };
                inputs.pop();
                stacker.run(inputs);
                let (lines, ds) = stacker.hard_drop();
                st.pcs += 1;
                ds_total += ds;
                st.nds += lines.saturating_sub(ds);
                if ds > 0 {
                    st.ds_cnt += 1;
                    if ds == 1 { st.ds1 += 1; } else if ds == 2 { st.ds2 += 1; } else { st.ds3 += 1; }
                    let (_, new_bo, _, _) = h1_info(&stacker);
                    if new_bo > 0 { st.ds_cov_next += 1; }
                } else {
                    st.zds += 1;
                    if rem_g0 <= 7 && rem_g0 == g_rows { st.zds_end += 1; }
                    else if g_rows == 0 { st.zds_g0 += 1; }
                    else if blocks_over > 0 { st.zds_h1_cov += 1; }
                    else {
                        let is_o = fp.color_idx == 1;
                        let has_i = cur_ch == 'I' || hold_ch == Some('I');
                        if is_o { st.zds_o += 1; }
                        else if chute_min >= 1 && !has_i { st.zds_nar += 1; }
                        else { st.zds_wide += 1; }
                    }
                }
            }
            if ds_total < 100 { st.topout = true; st.pcs = 400; }
            st.r_end = stacker.matrix().count();
        } else {
            st.topout = true;
            st.pcs = 400;
        }
        st
    };

    let workers = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(2).min(exact_data.len().max(1));
    let next_idx = std::sync::atomic::AtomicUsize::new(0);
    let mut results: Vec<Option<GameStat>> = vec![None; exact_data.len()];
    std::thread::scope(|sc| {
        let mut handles = Vec::new();
        for _ in 0..workers {
            handles.push(sc.spawn(|| {
                let mut local = Vec::new();
                loop {
                    let i = next_idx.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    if i >= exact_data.len() { break; }
                    local.push((i, solve_one(&exact_data[i])));
                }
                local
            }));
        }
        for h in handles {
            for (i, st) in h.join().unwrap() { results[i] = Some(st); }
        }
    });
    let res: Vec<GameStat> = results.into_iter().map(|x| x.unwrap()).collect();
    print_summary(label, &res, t0.elapsed().as_secs_f64());
}

fn run_oracle_agg<M: DiagEvaluator>(
    label: &str,
    exact_data: &[ExactSeedData],
    horizon: usize,
    beam_w: usize,
    top_k_mean: usize,
    anneal_rg: bool,
    early_w: i64,
    mdl: &M,
) {
    type FxBuild = std::hash::BuildHasherDefault<blockfish::ai::beam::FxHasher>;
    #[derive(Clone, Copy)]
    struct ONode { board: FastBoard, held: Option<u8>, q_cursor: usize, rem_g: u16, first_idx: u16, cum_rg: i64, score: i64 }
    #[derive(Clone, Copy, Eq, PartialEq, Hash)]
    struct OKey { board: FastBoard, held: Option<u8>, rem_g: u16 }

    let t0 = Instant::now();
    let fst = FastShapeTable::srs();
    let spawn_exact_cheese = |b: &mut FastBoard, rem_g: u16, lines_cleared: u8, holes: &[u16]| -> bool {
        let target_g: usize = if lines_cleared > 0 { 4 } else { 9 };
        let g = b.garbage_rows as usize;
        let unspawned = (rem_g as usize).saturating_sub(g);
        let num = target_g.saturating_sub(g).min(unspawned);
        if num == 0 { return true; }
        let old_rows = b.rows as usize;
        if old_rows + num >= 32 { return false; }
        let spawned_so_far = 100usize - unspawned;
        for r in (0..old_rows).rev() { b.data[r + num] = b.data[r]; }
        for k in 0..num {
            let hole = holes[spawned_so_far + k];
            b.data[num - 1 - k] = 0xFC00 | (0x3FF & !(1u16 << hole));
        }
        b.rows = (old_rows + num) as u8;
        b.garbage_rows = (g + num) as u8;
        true
    };

    let run_one = |ed: &ExactSeedData| -> GameStat {
        let mut game_cfg = BSConfig::default();
        game_cfg.garbage.total_lines = Some(100);
        game_cfg.prng_seed = Some(ed.seed);
        let mut stacker = Stacker::new(Ruleset::guideline().into(), game_cfg);
        let mut pgen = FastPlaceGen::new();
        let mut ai_in = FastAI::new(BeamConfig::default());
        let mut root_places = Vec::with_capacity(64);
        let mut tmp_places = Vec::with_capacity(64);
        let mut cur_beam: Vec<ONode> = Vec::with_capacity(beam_w);
        let mut children: Vec<ONode> = Vec::with_capacity(beam_w * 32);
        let mut seen: HashSet<OKey, FxBuild> = HashSet::with_capacity_and_hasher(beam_w * 4, FxBuild::default());

        let mut st = GameStat::default();
        let mut ds_total = 0usize;
        let mut q_pos = 0usize;

        while ds_total < 100 && stacker.current_piece_type().is_some() {
            let snap = stacker.snapshot().unwrap();
            let g_rows = garbage_rows(&stacker);
            let root_board = FastBoard::from_basic_matrix(&snap.matrix, g_rows);
            let rem_g0 = 100usize.saturating_sub(ds_total) as u16;
            let hold0 = snap.hold.map(|cc| color_to_idx(cc) as u8);
            let q_start = q_pos.min(ed.pieces.len());
            let q_end = (q_start + horizon).min(ed.pieces.len());
            let q = &ed.pieces[q_start..q_end];
            let q_len = q.len();
            if q_len == 0 { st.topout = true; st.pcs = 400; break; }
            let cur0 = Some(q[0]);
            let alt0 = if hold0.is_some() { hold0 } else if q_len > 1 { Some(q[1]) } else { None };
            let (_, blocks_over, chute_min, cells_above) = h1_info(&stacker);
            st.cells_sum += cells_above;
            let cur_ch = snap.queue[0].as_char();
            let hold_ch = snap.hold.or_else(|| snap.queue.get(1).copied()).map(|cc| cc.as_char());

            pgen.generate_for_state(&fst, &root_board, cur0, alt0, &mut root_places);
            if root_places.is_empty() { st.topout = true; st.pcs = 400; break; }

            let is_end = rem_g0 <= 7 && rem_g0 == g_rows;
            let eff_w = if is_end { beam_w.max(400) } else { beam_w };
            let eff_mpr = if is_end { 96 } else { (eff_w / 4).max(24) };
            let target_d = q_len.saturating_sub(if hold0.is_some() { 0 } else { 1 }).max(1);

            cur_beam.clear();
            seen.clear();
            let mut best_goal: Option<(usize, u16, i64)> = None;

            for (idx, &fp) in root_places.iter().enumerate() {
                let mut b = root_board;
                let o = &fst.shapes[fp.color_idx as usize].orients[fp.tf.2 as usize];
                let (lines, ds, _) = b.place_and_sift(o, fp.norm_i, fp.norm_j);
                let rem_g = rem_g0.saturating_sub(ds as u16);
                if rem_g > 0 && !spawn_exact_cheese(&mut b, rem_g, lines, &ed.holes) { continue; }
                let (held, qc) = if hold0.is_some() {
                    if fp.did_hold { (cur0, 1.min(q_len)) } else { (hold0, 1.min(q_len)) }
                } else if fp.did_hold { (cur0, 2.min(q_len)) } else { (None, 1.min(q_len)) };
                if rem_g == 0 && b.rows < 20 {
                    let gr = -30_000_000 + 100 + b.rows as i64;
                    if best_goal.map_or(true, |x| gr < x.2) { best_goal = Some((1, idx as u16, gr)); }
                }
                let in_h = held.or_else(|| q.get(qc).copied());
                let mut sc = mdl.eval_b(&b, rem_g, in_h, 0x7F);
                let cleared1 = rem_g0.saturating_sub(rem_g) as i64;
                if anneal_rg && !is_end && target_d > 1 {
                    sc += cleared1 * 35; // reduce greedy rg incentive in early beam depths
                }
                if early_w > 0 && !is_end && target_d > 1 {
                    sc -= (early_w * cleared1) / (target_d as i64);
                }
                if seen.insert(OKey { board: b, held, rem_g }) {
                    cur_beam.push(ONode { board: b, held, q_cursor: qc, rem_g, first_idx: idx as u16, cum_rg: cleared1, score: sc });
                }
            }

            for depth in 2..=target_d {
                children.clear();
                let mid_pen = if anneal_rg && !is_end && depth < target_d {
                    let rem_frac = (target_d - depth) as i64;
                    (35 * rem_frac) / (target_d as i64)
                } else { 0 };
                for node in &cur_beam {
                    if node.rem_g == 0 { continue; }
                    let qc = node.q_cursor;
                    let (c_nh, c_h) = if node.held.is_some() { (q.get(qc).copied(), node.held) } else { (q.get(qc).copied(), q.get(qc + 1).copied()) };
                    pgen.generate_for_state(&fst, &node.board, c_nh, c_h, &mut tmp_places);
                    for &fp in &tmp_places {
                        let mut b = node.board;
                        let o = &fst.shapes[fp.color_idx as usize].orients[fp.tf.2 as usize];
                        let (lines, ds, _) = b.place_and_sift(o, fp.norm_i, fp.norm_j);
                        let rem_g = node.rem_g.saturating_sub(ds as u16);
                        if rem_g > 0 && !spawn_exact_cheese(&mut b, rem_g, lines, &ed.holes) { continue; }
                        let (held, nqc) = if node.held.is_some() {
                            if fp.did_hold { (c_nh, (qc + 1).min(q_len)) } else { (node.held, (qc + 1).min(q_len)) }
                        } else if fp.did_hold { (c_nh, (qc + 2).min(q_len)) } else { (None, (qc + 1).min(q_len)) };
                        if rem_g == 0 && b.rows < 20 {
                            let gr = -30_000_000 + (depth as i64) * 100 + b.rows as i64;
                            if best_goal.map_or(true, |x| gr < x.2) { best_goal = Some((depth, node.first_idx, gr)); }
                        }
                        let in_h = held.or_else(|| q.get(nqc).copied());
                        let mut sc = mdl.eval_b(&b, rem_g, in_h, 0x7F);
                        let cleared = rem_g0.saturating_sub(rem_g) as i64;
                        let new_cum = node.cum_rg + cleared;
                        if mid_pen > 0 {
                            sc += cleared * mid_pen;
                        }
                        if early_w > 0 && !is_end {
                            sc -= (early_w * new_cum) / (target_d as i64);
                        }
                        children.push(ONode { board: b, held, q_cursor: nqc, rem_g, first_idx: node.first_idx, cum_rg: new_cum, score: sc });
                    }
                }
                if children.is_empty() { break; }
                children.sort_unstable_by_key(|n| (n.score, n.first_idx));
                cur_beam.clear();
                seen.clear();
                let mut rc = vec![0usize; root_places.len()];
                for &ch in &children {
                    if cur_beam.len() >= eff_w { break; }
                    if rc[ch.first_idx as usize] >= eff_mpr { continue; }
                    if seen.insert(OKey { board: ch.board, held: ch.held, rem_g: ch.rem_g }) {
                        rc[ch.first_idx as usize] += 1;
                        cur_beam.push(ch);
                    }
                }
            }

            let chosen_idx = if let Some((_, ri, _)) = best_goal {
                ri as usize
            } else if top_k_mean > 1 && !is_end && !cur_beam.is_empty() {
                let mut sums = vec![0i64; root_places.len()];
                let mut cnts = vec![0usize; root_places.len()];
                let mut best_sc = vec![i64::MAX; root_places.len()];
                for n in &cur_beam {
                    let ri = n.first_idx as usize;
                    if cnts[ri] < top_k_mean {
                        if cnts[ri] == 0 { best_sc[ri] = n.score; }
                        sums[ri] += n.score;
                        cnts[ri] += 1;
                    }
                }
                let mut best_ri = cur_beam[0].first_idx as usize;
                let mut best_agg = i64::MAX;
                for ri in 0..root_places.len() {
                    if cnts[ri] == 0 { continue; }
                    // Pad missing leaves with best+35 so a single fluke without backups gets slight penalty
                    let padded = sums[ri] + ((top_k_mean - cnts[ri]) as i64) * (best_sc[ri] + 35);
                    let agg = padded / (top_k_mean as i64);
                    if agg < best_agg {
                        best_agg = agg;
                        best_ri = ri;
                    }
                }
                best_ri
            } else if let Some(best) = cur_beam.first() {
                best.first_idx as usize
            } else { 0 };

            let fp = root_places[chosen_idx];
            let mut inputs = ai_in.inputs_for_place(&snap.matrix, fp).unwrap();
            inputs.pop();
            if fp.did_hold { if hold0.is_none() { q_pos += 2; } else { q_pos += 1; } } else { q_pos += 1; }
            stacker.run(inputs);
            let (lines, ds) = stacker.hard_drop();
            st.pcs += 1;
            ds_total += ds;
            st.nds += lines.saturating_sub(ds);
            if ds > 0 {
                st.ds_cnt += 1;
                if ds == 1 { st.ds1 += 1; } else if ds == 2 { st.ds2 += 1; } else { st.ds3 += 1; }
                let (_, new_bo, _, _) = h1_info(&stacker);
                if new_bo > 0 { st.ds_cov_next += 1; }
            } else {
                st.zds += 1;
                if rem_g0 <= 7 && rem_g0 == g_rows { st.zds_end += 1; }
                else if g_rows == 0 { st.zds_g0 += 1; }
                else if blocks_over > 0 { st.zds_h1_cov += 1; }
                else {
                    let is_o = fp.color_idx == 1;
                    let has_i = cur_ch == 'I' || hold_ch == Some('I');
                    if is_o { st.zds_o += 1; }
                    else if chute_min >= 1 && !has_i { st.zds_nar += 1; }
                    else { st.zds_wide += 1; }
                }
            }
        }
        if ds_total < 100 { st.topout = true; st.pcs = 400; }
        st.r_end = stacker.matrix().count();
        st
    };

    let workers = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(2).min(exact_data.len().max(1));
    let next_idx = std::sync::atomic::AtomicUsize::new(0);
    let mut results: Vec<Option<GameStat>> = vec![None; exact_data.len()];
    std::thread::scope(|sc| {
        let mut handles = Vec::new();
        for _ in 0..workers {
            handles.push(sc.spawn(|| {
                let mut local = Vec::new();
                loop {
                    let i = next_idx.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    if i >= exact_data.len() { break; }
                    local.push((i, run_one(&exact_data[i])));
                }
                local
            }));
        }
        for h in handles {
            for (i, st) in h.join().unwrap() { results[i] = Some(st); }
        }
    });
    let res: Vec<GameStat> = results.into_iter().map(|x| x.unwrap()).collect();
    print_summary(label, &res, t0.elapsed().as_secs_f64());
}

fn run_oracle_decomp(
    label: &str,
    exact_data: &[ExactSeedData],
    horizon: usize,
    beam_w: usize,
    pareto_clean: bool,
    roll_7bag_leaf: bool,
    roll_exact_ub_leaf: bool,
    mdl: &LearnedValueModel,
) {
    type FxBuild = std::hash::BuildHasherDefault<blockfish::ai::beam::FxHasher>;
    #[derive(Clone, Copy)]
    struct ONode {
        board: FastBoard,
        held: Option<u8>,
        q_cursor: usize,
        rem_g: u16,
        first_idx: u16,
        score: i64,
        clean_score: i64,
    }
    #[derive(Clone, Copy, Eq, PartialEq, Hash)]
    struct OKey {
        board: FastBoard,
        held: Option<u8>,
        rem_g: u16,
    }

    let t0 = Instant::now();
    let fst = FastShapeTable::srs();
    let spawn_exact_cheese = |b: &mut FastBoard, rem_g: u16, lines_cleared: u8, holes: &[u16]| -> bool {
        let target_g: usize = if lines_cleared > 0 { 4 } else { 9 };
        let g = b.garbage_rows as usize;
        let unspawned = (rem_g as usize).saturating_sub(g);
        let num = target_g.saturating_sub(g).min(unspawned);
        if num == 0 { return true; }
        let old_rows = b.rows as usize;
        if old_rows + num >= 32 { return false; }
        let spawned_so_far = 100usize - unspawned;
        for r in (0..old_rows).rev() {
            b.data[r + num] = b.data[r];
        }
        for k in 0..num {
            let hole = holes[spawned_so_far + k];
            let dst_r = num - 1 - k;
            b.data[dst_r] = 0xFC00 | (0x3FF & !(1u16 << hole));
        }
        b.rows = (old_rows + num) as u8;
        b.garbage_rows = (g + num) as u8;
        true
    };

    // Clean-preserving score: lower rem_g weight (1.65 vs 2.12), plus direct stack/hole cleanliness
    let mut m_clean = mdl.clone();
    m_clean.linear_w[0] = 1.65 / m_clean.inv_std[0];
    m_clean.linear_w[9] = 0.12 / m_clean.inv_std[9];
    m_clean.linear_w[20] = 0.40 / m_clean.inv_std[20];
    m_clean.linear_w[36] = 0.22 / m_clean.inv_std[36];
    m_clean.linear_w[39] = -0.15 / m_clean.inv_std[39];
    m_clean.prepare_fast();

    let leaf_roll_score = |pgen: &mut FastPlaceGen,
                           tmp_places: &mut Vec<blockfish::ai::fast::FastPlace>,
                           b0: FastBoard,
                           h0: Option<u8>,
                           rem_g0: u16,
                           q_tail: &[u8],
                           holes: &[u16],
                           exact_q: bool,
                           seed_base: u64| -> i64 {
        if rem_g0 == 0 {
            return -20_000_000 + (b0.rows as i64) * 5;
        }
        let roll_len = 12usize;
        let roll_bw = 6usize;
        let w_cnt = if exact_q { 1usize } else { 3usize };
        let mut sum_sc: i64 = 0;

        #[derive(Clone, Copy)]
        struct RNode {
            board: FastBoard,
            held: Option<u8>,
            qc: usize,
            rem_g: u16,
            score: i64,
        }
        let mut r_beam: Vec<RNode> = Vec::with_capacity(roll_bw);
        let mut r_next: Vec<RNode> = Vec::with_capacity(roll_bw * 32);
        let mut r_seen: HashSet<OKey, FxBuild> =
            HashSet::with_capacity_and_hasher(roll_bw * 8, FxBuild::default());

        for w in 0..w_cnt {
            let mut wq = [0u8; 18];
            let wq_len;
            if exact_q {
                wq_len = q_tail.len().min(roll_len + 1);
                wq[..wq_len].copy_from_slice(&q_tail[..wq_len]);
            } else {
                wq_len = roll_len + 1;
                let mut bag: u8 = 0x7F;
                let mut rng = seed_base ^ ((w as u64) + 1).wrapping_mul(0x9E3779B97F4A7C15);
                for slot in wq.iter_mut().take(wq_len) {
                    if bag == 0 {
                        bag = 0x7F;
                    }
                    let cnt = bag.count_ones();
                    rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                    let mut pick = ((rng >> 32) % cnt as u64) as u32;
                    let mut m = bag;
                    while pick > 0 {
                        m &= m - 1;
                        pick -= 1;
                    }
                    let p = m.trailing_zeros() as u8;
                    bag &= !(1u8 << p);
                    *slot = p;
                }
            }
            if wq_len == 0 {
                sum_sc += mdl.evaluate_board_with_bag(&b0, rem_g0, h0, 0x7F);
                continue;
            }
            r_beam.clear();
            r_beam.push(RNode {
                board: b0,
                held: h0,
                qc: 0,
                rem_g: rem_g0,
                score: mdl.evaluate_board_with_bag(&b0, rem_g0, h0, 0x7F),
            });
            let mut best_done: Option<i64> = None;
            let mut steps_ran = 0usize;
            for step in 1..=roll_len {
                r_next.clear();
                for &rn in &r_beam {
                    if rn.rem_g == 0 {
                        continue;
                    }
                    let qc = rn.qc;
                    let (c_nh, c_h) = if rn.held.is_some() {
                        (wq[..wq_len].get(qc).copied(), rn.held)
                    } else {
                        (wq[..wq_len].get(qc).copied(), wq[..wq_len].get(qc + 1).copied())
                    };
                    if c_nh.is_none() && c_h.is_none() {
                        continue;
                    }
                    pgen.generate_for_state(&fst, &rn.board, c_nh, c_h, tmp_places);
                    for &fp in tmp_places.iter() {
                        let mut b = rn.board;
                        let o = &fst.shapes[fp.color_idx as usize].orients[fp.tf.2 as usize];
                        let (lines, ds, _) = b.place_and_sift(o, fp.norm_i, fp.norm_j);
                        let rem_g = rn.rem_g.saturating_sub(ds as u16);
                        if rem_g > 0 && !spawn_exact_cheese(&mut b, rem_g, lines, holes) {
                            continue;
                        }
                        let (held, nqc) = if rn.held.is_some() {
                            if fp.did_hold { (c_nh, (qc + 1).min(wq_len)) } else { (rn.held, (qc + 1).min(wq_len)) }
                        } else if fp.did_hold {
                            (c_nh, (qc + 2).min(wq_len))
                        } else {
                            (None, (qc + 1).min(wq_len))
                        };
                        if rem_g == 0 && b.rows < 20 {
                            let d_sc = -20_000_000 + (step as i64) * 100 + (b.rows as i64) * 5;
                            if best_done.map_or(true, |x| d_sc < x) {
                                best_done = Some(d_sc);
                            }
                            continue;
                        }
                        let in_h = held.or_else(|| wq[..wq_len].get(nqc).copied());
                        let sc = mdl.evaluate_board_with_bag(&b, rem_g, in_h, 0x7F);
                        r_next.push(RNode { board: b, held, qc: nqc, rem_g, score: sc });
                    }
                }
                if r_next.is_empty() {
                    break;
                }
                steps_ran = step;
                if best_done.is_some() {
                    break;
                }
                r_next.sort_unstable_by_key(|x| x.score);
                r_beam.clear();
                r_seen.clear();
                for &rn in &r_next {
                    if r_beam.len() >= roll_bw {
                        break;
                    }
                    if r_seen.insert(OKey { board: rn.board, held: rn.held, rem_g: rn.rem_g }) {
                        r_beam.push(rn);
                    }
                }
            }
            let w_sc = if let Some(d_sc) = best_done {
                d_sc
            } else if let Some(top) = r_beam.first() {
                (steps_ran as i64) * 100 + top.score
            } else {
                200_000
            };
            sum_sc += w_sc;
        }
        sum_sc / (w_cnt as i64)
    };

    let run_one = |ed: &ExactSeedData| -> GameStat {
        let mut game_cfg = BSConfig::default();
        game_cfg.garbage.total_lines = Some(100);
        game_cfg.prng_seed = Some(ed.seed);
        let mut stacker = Stacker::new(Ruleset::guideline().into(), game_cfg);
        let mut pgen = FastPlaceGen::new();
        let mut rgen = FastPlaceGen::new();
        let mut ai_in = FastAI::new(BeamConfig::default());
        let mut root_places = Vec::with_capacity(64);
        let mut tmp_places = Vec::with_capacity(64);
        let mut roll_tmp_places = Vec::with_capacity(64);
        let mut cur_beam: Vec<ONode> = Vec::with_capacity(beam_w);
        let mut children: Vec<ONode> = Vec::with_capacity(beam_w * 32);
        let mut seen: HashSet<OKey, FxBuild> = HashSet::with_capacity_and_hasher(beam_w * 4, FxBuild::default());

        let mut st = GameStat::default();
        let mut ds_total = 0usize;
        let mut q_pos = 0usize;

        while ds_total < 100 && stacker.current_piece_type().is_some() {
            let snap = stacker.snapshot().unwrap();
            let g_rows = garbage_rows(&stacker);
            let root_board = FastBoard::from_basic_matrix(&snap.matrix, g_rows);
            let rem_g0 = 100usize.saturating_sub(ds_total) as u16;
            let hold0 = snap.hold.map(|cc| color_to_idx(cc) as u8);
            let q_start = q_pos.min(ed.pieces.len());
            let q_end = (q_start + horizon).min(ed.pieces.len());
            let q = &ed.pieces[q_start..q_end];
            let q_len = q.len();
            if q_len == 0 { st.topout = true; st.pcs = 400; break; }
            let cur0 = Some(q[0]);
            let alt0 = if hold0.is_some() { hold0 } else if q_len > 1 { Some(q[1]) } else { None };

            let (_, blocks_over, chute_min, cells_above) = h1_info(&stacker);
            st.cells_sum += cells_above;
            let cur_ch = snap.queue[0].as_char();
            let hold_ch = snap.hold.or_else(|| snap.queue.get(1).copied()).map(|cc| cc.as_char());

            pgen.generate_for_state(&fst, &root_board, cur0, alt0, &mut root_places);
            if root_places.is_empty() { st.topout = true; st.pcs = 400; break; }

            let is_end = rem_g0 <= 7 && rem_g0 == g_rows;
            let eff_w = if is_end { beam_w.max(400) } else { beam_w };
            let eff_mpr = if is_end { 96 } else { (eff_w / 4).max(24) };
            let target_d = q_len.saturating_sub(if hold0.is_some() { 0 } else { 1 }).max(1);

            cur_beam.clear();
            seen.clear();
            let mut best_goal: Option<(usize, u16, i64)> = None;

            for (idx, &fp) in root_places.iter().enumerate() {
                let mut b = root_board;
                let o = &fst.shapes[fp.color_idx as usize].orients[fp.tf.2 as usize];
                let (lines, ds, _) = b.place_and_sift(o, fp.norm_i, fp.norm_j);
                let rem_g = rem_g0.saturating_sub(ds as u16);
                if rem_g > 0 && !spawn_exact_cheese(&mut b, rem_g, lines, &ed.holes) { continue; }
                let (held, qc) = if hold0.is_some() {
                    if fp.did_hold { (cur0, 1.min(q_len)) } else { (hold0, 1.min(q_len)) }
                } else if fp.did_hold { (cur0, 2.min(q_len)) } else { (None, 1.min(q_len)) };
                if rem_g == 0 && b.rows < 20 {
                    let gr = -30_000_000 + 100 + b.rows as i64;
                    if best_goal.map_or(true, |x| gr < x.2) { best_goal = Some((1, idx as u16, gr)); }
                }
                let in_h = held.or_else(|| q.get(qc).copied());
                let sc = mdl.evaluate_board_with_bag(&b, rem_g, in_h, 0x7F);
                let csc = m_clean.evaluate_board_with_bag(&b, rem_g, in_h, 0x7F);
                if seen.insert(OKey { board: b, held, rem_g }) {
                    cur_beam.push(ONode { board: b, held, q_cursor: qc, rem_g, first_idx: idx as u16, score: sc, clean_score: csc });
                }
            }

            for depth in 2..=target_d {
                children.clear();
                for node in &cur_beam {
                    if node.rem_g == 0 { continue; }
                    let qc = node.q_cursor;
                    let (c_nh, c_h) = if node.held.is_some() {
                        (q.get(qc).copied(), node.held)
                    } else {
                        (q.get(qc).copied(), q.get(qc + 1).copied())
                    };
                    pgen.generate_for_state(&fst, &node.board, c_nh, c_h, &mut tmp_places);
                    for &fp in &tmp_places {
                        let mut b = node.board;
                        let o = &fst.shapes[fp.color_idx as usize].orients[fp.tf.2 as usize];
                        let (lines, ds, _) = b.place_and_sift(o, fp.norm_i, fp.norm_j);
                        let rem_g = node.rem_g.saturating_sub(ds as u16);
                        if rem_g > 0 && !spawn_exact_cheese(&mut b, rem_g, lines, &ed.holes) { continue; }
                        let (held, nqc) = if node.held.is_some() {
                            if fp.did_hold { (c_nh, (qc + 1).min(q_len)) } else { (node.held, (qc + 1).min(q_len)) }
                        } else if fp.did_hold { (c_nh, (qc + 2).min(q_len)) } else { (None, (qc + 1).min(q_len)) };
                        if rem_g == 0 && b.rows < 20 {
                            let gr = -30_000_000 + (depth as i64) * 100 + b.rows as i64;
                            if best_goal.map_or(true, |x| gr < x.2) { best_goal = Some((depth, node.first_idx, gr)); }
                        }
                        let in_h = held.or_else(|| q.get(nqc).copied());
                        let sc = mdl.evaluate_board_with_bag(&b, rem_g, in_h, 0x7F);
                        let csc = m_clean.evaluate_board_with_bag(&b, rem_g, in_h, 0x7F);
                        children.push(ONode { board: b, held, q_cursor: nqc, rem_g, first_idx: node.first_idx, score: sc, clean_score: csc });
                    }
                }
                if children.is_empty() { break; }
                children.sort_unstable_by_key(|n| (n.score, n.first_idx));
                cur_beam.clear();
                seen.clear();
                let mut rc = vec![0usize; root_places.len()];

                // Primary 65% by score (100% if !pareto_clean)
                let primary_cap = if pareto_clean && !is_end { (eff_w * 2) / 3 } else { eff_w };
                for &ch in &children {
                    if cur_beam.len() >= primary_cap { break; }
                    if rc[ch.first_idx as usize] >= eff_mpr { continue; }
                    let key = OKey { board: ch.board, held: ch.held, rem_g: ch.rem_g };
                    if seen.insert(key) {
                        rc[ch.first_idx as usize] += 1;
                        cur_beam.push(ch);
                    }
                }
                if pareto_clean && !is_end && cur_beam.len() < eff_w {
                    let mut clean_order: Vec<usize> = (0..children.len()).collect();
                    clean_order.sort_unstable_by_key(|&i| (children[i].clean_score, children[i].first_idx));
                    for idx in clean_order {
                        if cur_beam.len() >= eff_w { break; }
                        let ch = children[idx];
                        if rc[ch.first_idx as usize] >= eff_mpr { continue; }
                        let key = OKey { board: ch.board, held: ch.held, rem_g: ch.rem_g };
                        if seen.insert(key) {
                            rc[ch.first_idx as usize] += 1;
                            cur_beam.push(ch);
                        }
                    }
                }
                if cur_beam.len() < eff_w {
                    for &ch in &children {
                        if cur_beam.len() >= eff_w { break; }
                        let key = OKey { board: ch.board, held: ch.held, rem_g: ch.rem_g };
                        if seen.insert(key) { cur_beam.push(ch); }
                    }
                }
                if pareto_clean && !is_end {
                    cur_beam.sort_unstable_by_key(|n| (n.score, n.first_idx));
                }
            }

            let chosen_idx = if let Some((_, ri, _)) = best_goal {
                ri as usize
            } else if (roll_7bag_leaf || roll_exact_ub_leaf) && !is_end && !cur_beam.is_empty() {
                // Evaluate top-16 diverse leaves by rollout continuation!
                let mut per_root_best: Vec<usize> = Vec::with_capacity(16);
                let mut used_roots: Vec<u16> = Vec::with_capacity(16);
                for (li, n) in cur_beam.iter().enumerate() {
                    if !used_roots.contains(&n.first_idx) {
                        used_roots.push(n.first_idx);
                        per_root_best.push(li);
                    }
                    if per_root_best.len() >= 6 { break; }
                }
                let mut clean_cands: Vec<usize> = (0..cur_beam.len()).collect();
                clean_cands.sort_unstable_by_key(|&li| cur_beam[li].clean_score);
                for li in clean_cands {
                    if per_root_best.len() >= 14 { break; }
                    if !per_root_best.contains(&li) {
                        per_root_best.push(li);
                    }
                }
                let mut best_ri = cur_beam[0].first_idx as usize;
                let mut best_sc = i64::MAX;
                let seed_b = ed.seed ^ ((st.pcs as u64 + 1).wrapping_mul(0x9E3779B97F4A7C15));
                for li in per_root_best {
                    let n = cur_beam[li];
                    let q_tail = &ed.pieces[(q_start + n.q_cursor).min(ed.pieces.len())..];
                    let r_sc = leaf_roll_score(
                        &mut rgen,
                        &mut roll_tmp_places,
                        n.board,
                        n.held,
                        n.rem_g,
                        q_tail,
                        &ed.holes,
                        roll_exact_ub_leaf,
                        seed_b,
                    );
                    if r_sc < best_sc {
                        best_sc = r_sc;
                        best_ri = n.first_idx as usize;
                    }
                }
                best_ri
            } else if let Some(best) = cur_beam.first() {
                best.first_idx as usize
            } else {
                0
            };

            let fp = root_places[chosen_idx];
            let mut inputs = ai_in.inputs_for_place(&snap.matrix, fp).unwrap();
            inputs.pop();
            if fp.did_hold {
                if hold0.is_none() { q_pos += 2; } else { q_pos += 1; }
            } else {
                q_pos += 1;
            }
            stacker.run(inputs);
            let (lines, ds) = stacker.hard_drop();
            st.pcs += 1;
            ds_total += ds;
            st.nds += lines.saturating_sub(ds);
            if ds > 0 {
                st.ds_cnt += 1;
                if ds == 1 { st.ds1 += 1; }
                else if ds == 2 { st.ds2 += 1; }
                else { st.ds3 += 1; }
                let (_, new_bo, _, _) = h1_info(&stacker);
                if new_bo > 0 { st.ds_cov_next += 1; }
            } else {
                st.zds += 1;
                if rem_g0 <= 7 && rem_g0 == g_rows { st.zds_end += 1; }
                else if g_rows == 0 { st.zds_g0 += 1; }
                else if blocks_over > 0 { st.zds_h1_cov += 1; }
                else {
                    let is_o = fp.color_idx == 1;
                    let has_i = cur_ch == 'I' || hold_ch == Some('I');
                    if is_o { st.zds_o += 1; }
                    else if chute_min >= 1 && !has_i { st.zds_nar += 1; }
                    else { st.zds_wide += 1; }
                }
            }
        }
        if ds_total < 100 { st.topout = true; st.pcs = 400; }
        st.r_end = stacker.matrix().count();
        st
    };

    let workers = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(2)
        .min(exact_data.len().max(1));
    let next_idx = std::sync::atomic::AtomicUsize::new(0);
    let mut results: Vec<Option<GameStat>> = vec![None; exact_data.len()];
    std::thread::scope(|sc| {
        let mut handles = Vec::new();
        for _ in 0..workers {
            handles.push(sc.spawn(|| {
                let mut local = Vec::new();
                loop {
                    let i = next_idx.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    if i >= exact_data.len() { break; }
                    local.push((i, run_one(&exact_data[i])));
                }
                local
            }));
        }
        for h in handles {
            for (i, st) in h.join().unwrap() {
                results[i] = Some(st);
            }
        }
    });
    let res: Vec<GameStat> = results.into_iter().map(|x| x.unwrap()).collect();
    print_summary(label, &res, t0.elapsed().as_secs_f64());
}

fn run_oracle_adv<M: DiagEvaluator>(
    label: &str,
    exact_data: &[ExactSeedData],
    horizon: usize,
    beam_w: usize,
    prog_mpr: bool,
    stratify: bool,
    mpr_override: Option<usize>,
    mdl: &M,
) {
    type FxBuild = std::hash::BuildHasherDefault<blockfish::ai::beam::FxHasher>;
    #[derive(Clone, Copy)]
    struct ONode { board: FastBoard, held: Option<u8>, q_cursor: usize, rem_g: u16, first_idx: u16, score: i64 }
    #[derive(Clone, Copy, Eq, PartialEq, Hash)]
    struct OKey { board: FastBoard, held: Option<u8>, rem_g: u16 }

    let t0 = Instant::now();
    let fst = FastShapeTable::srs();
    let spawn_exact_cheese = |b: &mut FastBoard, rem_g: u16, lines_cleared: u8, holes: &[u16]| -> bool {
        let target_g: usize = if lines_cleared > 0 { 4 } else { 9 };
        let g = b.garbage_rows as usize;
        let unspawned = (rem_g as usize).saturating_sub(g);
        let num = target_g.saturating_sub(g).min(unspawned);
        if num == 0 { return true; }
        let old_rows = b.rows as usize;
        if old_rows + num >= 32 { return false; }
        let spawned_so_far = 100usize - unspawned;
        for r in (0..old_rows).rev() {
            b.data[r + num] = b.data[r];
        }
        for k in 0..num {
            let hole = holes[spawned_so_far + k];
            let dst_r = num - 1 - k;
            b.data[dst_r] = 0xFC00 | (0x3FF & !(1u16 << hole));
        }
        b.rows = (old_rows + num) as u8;
        b.garbage_rows = (g + num) as u8;
        true
    };

    let run_one = |ed: &ExactSeedData| -> GameStat {
        let mut game_cfg = BSConfig::default();
        game_cfg.garbage.total_lines = Some(100);
        game_cfg.prng_seed = Some(ed.seed);
        let mut stacker = Stacker::new(Ruleset::guideline().into(), game_cfg);
        let mut pgen = FastPlaceGen::new();
        let mut ai_in = FastAI::new(BeamConfig::default());
        let mut root_places = Vec::with_capacity(64);
        let mut tmp_places = Vec::with_capacity(64);
        let mut cur_beam: Vec<ONode> = Vec::with_capacity(beam_w);
        let mut children: Vec<ONode> = Vec::with_capacity(beam_w * 32);
        let mut seen: HashSet<OKey, FxBuild> = HashSet::with_capacity_and_hasher(beam_w * 4, FxBuild::default());

        let mut st = GameStat::default();
        let mut ds_total = 0usize;
        let mut q_pos = 0usize;

        while ds_total < 100 && stacker.current_piece_type().is_some() {
            let snap = stacker.snapshot().unwrap();
            let g_rows = garbage_rows(&stacker);
            let root_board = FastBoard::from_basic_matrix(&snap.matrix, g_rows);
            let rem_g0 = 100usize.saturating_sub(ds_total) as u16;
            let hold0 = snap.hold.map(|cc| color_to_idx(cc) as u8);
            let q_start = q_pos.min(ed.pieces.len());
            let q_end = (q_start + horizon).min(ed.pieces.len());
            let q = &ed.pieces[q_start..q_end];
            let q_len = q.len();
            if q_len == 0 { st.topout = true; st.pcs = 400; break; }
            let cur0 = Some(q[0]);
            let alt0 = if hold0.is_some() { hold0 } else if q_len > 1 { Some(q[1]) } else { None };

            let (_, blocks_over, chute_min, cells_above) = h1_info(&stacker);
            st.cells_sum += cells_above;
            let cur_ch = snap.queue[0].as_char();
            let hold_ch = snap.hold.or_else(|| snap.queue.get(1).copied()).map(|cc| cc.as_char());

            pgen.generate_for_state(&fst, &root_board, cur0, alt0, &mut root_places);
            if root_places.is_empty() { st.topout = true; st.pcs = 400; break; }

            let is_end = rem_g0 <= 7 && rem_g0 == g_rows;
            let eff_w = if is_end { beam_w.max(400) } else { beam_w };
            let eff_mpr = if is_end { 96 } else { mpr_override.unwrap_or((eff_w / 4).max(24)) };
            let target_d = q_len.saturating_sub(if hold0.is_some() { 0 } else { 1 }).max(1);

            cur_beam.clear();
            seen.clear();
            let mut best_goal: Option<(usize, u16, i64)> = None;

            for (idx, &fp) in root_places.iter().enumerate() {
                let mut b = root_board;
                let o = &fst.shapes[fp.color_idx as usize].orients[fp.tf.2 as usize];
                let (lines, ds, _) = b.place_and_sift(o, fp.norm_i, fp.norm_j);
                let rem_g = rem_g0.saturating_sub(ds as u16);
                if rem_g > 0 && !spawn_exact_cheese(&mut b, rem_g, lines, &ed.holes) { continue; }
                let (held, qc) = if hold0.is_some() {
                    if fp.did_hold { (cur0, 1.min(q_len)) } else { (hold0, 1.min(q_len)) }
                } else if fp.did_hold { (cur0, 2.min(q_len)) } else { (None, 1.min(q_len)) };
                if rem_g == 0 && b.rows < 20 {
                    let gr = -30_000_000 + 100 + b.rows as i64;
                    if best_goal.map_or(true, |x| gr < x.2) { best_goal = Some((1, idx as u16, gr)); }
                }
                let in_h = held.or_else(|| q.get(qc).copied());
                let sc = mdl.eval_b(&b, rem_g, in_h, 0x7F);
                let key = OKey { board: b, held, rem_g };
                if seen.insert(key) {
                    cur_beam.push(ONode { board: b, held, q_cursor: qc, rem_g, first_idx: idx as u16, score: sc });
                }
            }

            for depth in 2..=target_d {
                children.clear();
                for node in &cur_beam {
                    if node.rem_g == 0 { continue; }
                    let qc = node.q_cursor;
                    let (c_nh, c_h) = if node.held.is_some() {
                        (q.get(qc).copied(), node.held)
                    } else {
                        (q.get(qc).copied(), q.get(qc + 1).copied())
                    };
                    pgen.generate_for_state(&fst, &node.board, c_nh, c_h, &mut tmp_places);
                    for &fp in &tmp_places {
                        let mut b = node.board;
                        let o = &fst.shapes[fp.color_idx as usize].orients[fp.tf.2 as usize];
                        let (lines, ds, _) = b.place_and_sift(o, fp.norm_i, fp.norm_j);
                        let rem_g = node.rem_g.saturating_sub(ds as u16);
                        if rem_g > 0 && !spawn_exact_cheese(&mut b, rem_g, lines, &ed.holes) { continue; }
                        let (held, nqc) = if node.held.is_some() {
                            if fp.did_hold { (c_nh, (qc + 1).min(q_len)) } else { (node.held, (qc + 1).min(q_len)) }
                        } else if fp.did_hold { (c_nh, (qc + 2).min(q_len)) } else { (None, (qc + 1).min(q_len)) };
                        if rem_g == 0 && b.rows < 20 {
                            let gr = -30_000_000 + (depth as i64) * 100 + b.rows as i64;
                            if best_goal.map_or(true, |x| gr < x.2) { best_goal = Some((depth, node.first_idx, gr)); }
                        }
                        let in_h = held.or_else(|| q.get(nqc).copied());
                        let sc = mdl.eval_b(&b, rem_g, in_h, 0x7F);
                        children.push(ONode { board: b, held, q_cursor: nqc, rem_g, first_idx: node.first_idx, score: sc });
                    }
                }
                if children.is_empty() { break; }
                children.sort_unstable_by_key(|n| (n.score, n.first_idx));
                let step_mpr = if prog_mpr && !is_end {
                    ((eff_mpr * depth) / target_d.max(1)).max(4).min(eff_mpr)
                } else {
                    eff_mpr
                };
                cur_beam.clear();
                seen.clear();
                let mut rc = vec![0usize; root_places.len()];
                if stratify && !is_end {
                    let mut g_counts = [0usize; 24];
                    let bucket_cap = (eff_w / 4).max(16);
                    let strat_cap = (eff_w * 3) / 4;
                    for &ch in &children {
                        if cur_beam.len() >= strat_cap { break; }
                        let bi = rem_g0.saturating_sub(ch.rem_g).min(23) as usize;
                        if g_counts[bi] >= bucket_cap || rc[ch.first_idx as usize] >= step_mpr { continue; }
                        let key = OKey { board: ch.board, held: ch.held, rem_g: ch.rem_g };
                        if seen.insert(key) {
                            g_counts[bi] += 1;
                            rc[ch.first_idx as usize] += 1;
                            cur_beam.push(ch);
                        }
                    }
                }
                for &ch in &children {
                    if cur_beam.len() >= eff_w { break; }
                    if rc[ch.first_idx as usize] >= step_mpr { continue; }
                    let key = OKey { board: ch.board, held: ch.held, rem_g: ch.rem_g };
                    if seen.insert(key) {
                        rc[ch.first_idx as usize] += 1;
                        cur_beam.push(ch);
                    }
                }
                if cur_beam.len() < eff_w {
                    for &ch in &children {
                        if cur_beam.len() >= eff_w { break; }
                        let key = OKey { board: ch.board, held: ch.held, rem_g: ch.rem_g };
                        if seen.insert(key) { cur_beam.push(ch); }
                    }
                }
                if stratify && !is_end {
                    cur_beam.sort_unstable_by_key(|n| (n.score, n.first_idx));
                }
            }

            let chosen_idx = if let Some((_, ri, _)) = best_goal {
                ri as usize
            } else if let Some(best) = cur_beam.first() {
                best.first_idx as usize
            } else {
                0
            };
            let fp = root_places[chosen_idx];
            let mut inputs = ai_in.inputs_for_place(&snap.matrix, fp).unwrap();
            inputs.pop();
            if fp.did_hold {
                if hold0.is_none() { q_pos += 2; } else { q_pos += 1; }
            } else {
                q_pos += 1;
            }
            stacker.run(inputs);
            let (lines, ds) = stacker.hard_drop();
            st.pcs += 1;
            ds_total += ds;
            st.nds += lines.saturating_sub(ds);
            if ds > 0 {
                st.ds_cnt += 1;
                if ds == 1 { st.ds1 += 1; }
                else if ds == 2 { st.ds2 += 1; }
                else { st.ds3 += 1; }
                let (_, new_bo, _, _) = h1_info(&stacker);
                if new_bo > 0 { st.ds_cov_next += 1; }
            } else {
                st.zds += 1;
                if rem_g0 <= 7 && rem_g0 == g_rows { st.zds_end += 1; }
                else if g_rows == 0 { st.zds_g0 += 1; }
                else if blocks_over > 0 { st.zds_h1_cov += 1; }
                else {
                    let is_o = fp.color_idx == 1;
                    let has_i = cur_ch == 'I' || hold_ch == Some('I');
                    if is_o { st.zds_o += 1; }
                    else if chute_min >= 1 && !has_i { st.zds_nar += 1; }
                    else { st.zds_wide += 1; }
                }
            }
        }
        if ds_total < 100 { st.topout = true; st.pcs = 400; }
        st.r_end = stacker.matrix().count();
        st
    };

    let workers = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(2)
        .min(exact_data.len().max(1));
    let next_idx = std::sync::atomic::AtomicUsize::new(0);
    let mut results: Vec<Option<GameStat>> = vec![None; exact_data.len()];
    std::thread::scope(|sc| {
        let mut handles = Vec::new();
        for _ in 0..workers {
            handles.push(sc.spawn(|| {
                let mut local = Vec::new();
                loop {
                    let i = next_idx.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    if i >= exact_data.len() { break; }
                    local.push((i, run_one(&exact_data[i])));
                }
                local
            }));
        }
        for h in handles {
            for (i, st) in h.join().unwrap() {
                results[i] = Some(st);
            }
        }
    });
    let res: Vec<GameStat> = results.into_iter().map(|x| x.unwrap()).collect();
    print_summary(label, &res, t0.elapsed().as_secs_f64());
}

#[allow(clippy::too_many_arguments)]
fn run_oracle_deep_roll(
    label: &str,
    exact_data: &[ExactSeedData],
    horizon: usize,
    beam_w: usize,
    anneal_pen: i64,
    top_k_mean: usize,
    roll_len: usize,
    roll_bw: usize,
    mdl: &LearnedValueModel,
    term_mdl: &LearnedValueModel,
) {
    type FxBuild = std::hash::BuildHasherDefault<blockfish::ai::beam::FxHasher>;
    #[derive(Clone, Copy)]
    struct ONode { board: FastBoard, held: Option<u8>, q_cursor: usize, rem_g: u16, first_idx: u16, score: i64 }
    #[derive(Clone, Copy, Eq, PartialEq, Hash)]
    struct OKey { board: FastBoard, held: Option<u8>, rem_g: u16 }

    let t0 = Instant::now();
    let fst = FastShapeTable::srs();
    let spawn_exact_cheese = |b: &mut FastBoard, rem_g: u16, lines_cleared: u8, holes: &[u16]| -> bool {
        let target_g: usize = if lines_cleared > 0 { 4 } else { 9 };
        let g = b.garbage_rows as usize;
        let unspawned = (rem_g as usize).saturating_sub(g);
        let num = target_g.saturating_sub(g).min(unspawned);
        if num == 0 { return true; }
        let old_rows = b.rows as usize;
        if old_rows + num >= 32 { return false; }
        let spawned_so_far = 100usize - unspawned;
        if spawned_so_far + num > holes.len() { return false; }
        for r in (0..old_rows).rev() { b.data[r + num] = b.data[r]; }
        for k in 0..num {
            let hole = holes[spawned_so_far + k];
            b.data[num - 1 - k] = 0xFC00 | (0x3FF & !(1u16 << hole));
        }
        b.rows = (old_rows + num) as u8;
        b.garbage_rows = (g + num) as u8;
        true
    };

    let eval_roll_leaf = |pgen: &mut FastPlaceGen,
                          tmp_places: &mut Vec<blockfish::ai::fast::FastPlace>,
                          b0: FastBoard,
                          h0: Option<u8>,
                          rem_g0: u16,
                          q_tail: &[u8],
                          holes: &[u16]| -> i64 {
        if rem_g0 == 0 {
            return if b0.rows < 20 { -20_000_000 + (b0.rows as i64) * 5 } else { 5_000_000 };
        }
        #[derive(Clone, Copy)]
        struct RNode { board: FastBoard, held: Option<u8>, qc: usize, rem_g: u16, score: i64 }
        let mut r_beam: Vec<RNode> = Vec::with_capacity(roll_bw);
        let mut r_next: Vec<RNode> = Vec::with_capacity(roll_bw * 32);
        let mut r_seen: HashSet<OKey, FxBuild> = HashSet::with_capacity_and_hasher(roll_bw * 8, FxBuild::default());
        let ql = q_tail.len();
        let eff_len = roll_len.min(ql);
        if eff_len == 0 {
            let inh = h0;
            return term_mdl.evaluate_board_with_bag(&b0, rem_g0, inh, 0x7F);
        }
        r_beam.push(RNode {
            board: b0,
            held: h0,
            qc: 0,
            rem_g: rem_g0,
            score: mdl.evaluate_board_with_bag(&b0, rem_g0, h0.or_else(|| q_tail.first().copied()), 0x7F),
        });
        let mut best_done = i64::MAX;
        let mut last_step = 0usize;
        for step in 1..=eff_len {
            r_next.clear();
            let is_last = step == eff_len;
            for &rn in &r_beam {
                if rn.rem_g == 0 { continue; }
                let qc = rn.qc;
                let (c_nh, c_h) = if rn.held.is_some() {
                    (q_tail.get(qc).copied(), rn.held)
                } else {
                    (q_tail.get(qc).copied(), q_tail.get(qc + 1).copied())
                };
                if c_nh.is_none() && c_h.is_none() { continue; }
                pgen.generate_for_state(&fst, &rn.board, c_nh, c_h, tmp_places);
                for &fp in tmp_places.iter() {
                    let mut b = rn.board;
                    let o = &fst.shapes[fp.color_idx as usize].orients[fp.tf.2 as usize];
                    let (lines, ds, _) = b.place_and_sift(o, fp.norm_i, fp.norm_j);
                    let rem_g = rn.rem_g.saturating_sub(ds as u16);
                    if rem_g > 0 && !spawn_exact_cheese(&mut b, rem_g, lines, holes) { continue; }
                    let (held, nqc) = if rn.held.is_some() {
                        if fp.did_hold { (c_nh, (qc + 1).min(ql)) } else { (rn.held, (qc + 1).min(ql)) }
                    } else if fp.did_hold { (c_nh, (qc + 2).min(ql)) } else { (None, (qc + 1).min(ql)) };
                    if rem_g == 0 && b.rows < 20 {
                        let d_sc = -20_000_000 + (step as i64) * 100 + (b.rows as i64) * 5;
                        if d_sc < best_done { best_done = d_sc; }
                        continue;
                    }
                    let inh = held.or_else(|| q_tail.get(nqc).copied());
                    let sc = if is_last {
                        term_mdl.evaluate_board_with_bag(&b, rem_g, inh, 0x7F)
                    } else {
                        mdl.evaluate_board_with_bag(&b, rem_g, inh, 0x7F)
                    };
                    r_next.push(RNode { board: b, held, qc: nqc, rem_g, score: sc });
                }
            }
            if best_done != i64::MAX { return best_done; }
            if r_next.is_empty() { return 5_000_000; }
            last_step = step;
            r_next.sort_unstable_by_key(|n| n.score);
            r_beam.clear();
            r_seen.clear();
            for &rn in &r_next {
                if r_beam.len() >= roll_bw { break; }
                if r_seen.insert(OKey { board: rn.board, held: rn.held, rem_g: rn.rem_g }) {
                    r_beam.push(rn);
                }
            }
        }
        if let Some(top) = r_beam.first() {
            (last_step as i64) * 100 + top.score
        } else {
            5_000_000
        }
    };

    let run_one = |ed: &ExactSeedData| -> GameStat {
        let mut game_cfg = BSConfig::default();
        game_cfg.garbage.total_lines = Some(100);
        game_cfg.prng_seed = Some(ed.seed);
        let mut stacker = Stacker::new(Ruleset::guideline().into(), game_cfg);
        let mut pgen = FastPlaceGen::new();
        let mut rgen = FastPlaceGen::new();
        let mut ai_in = FastAI::new(BeamConfig::default());
        let mut root_places = Vec::with_capacity(64);
        let mut tmp_places = Vec::with_capacity(64);
        let mut r_tmp_places = Vec::with_capacity(64);
        let mut cur_beam: Vec<ONode> = Vec::with_capacity(beam_w);
        let mut children: Vec<ONode> = Vec::with_capacity(beam_w * 32);
        let mut seen: HashSet<OKey, FxBuild> = HashSet::with_capacity_and_hasher(beam_w * 4, FxBuild::default());

        let mut st = GameStat::default();
        let mut ds_total = 0usize;
        let mut q_pos = 0usize;

        while ds_total < 100 && stacker.current_piece_type().is_some() {
            let snap = stacker.snapshot().unwrap();
            let g_rows = garbage_rows(&stacker);
            let root_board = FastBoard::from_basic_matrix(&snap.matrix, g_rows);
            let rem_g0 = 100usize.saturating_sub(ds_total) as u16;
            let hold0 = snap.hold.map(|cc| color_to_idx(cc) as u8);
            let q_start = q_pos.min(ed.pieces.len());
            let q_end = (q_start + horizon).min(ed.pieces.len());
            let q = &ed.pieces[q_start..q_end];
            let q_len = q.len();
            if q_len == 0 { st.topout = true; st.pcs = 400; break; }
            let cur0 = Some(q[0]);
            let alt0 = if hold0.is_some() { hold0 } else if q_len > 1 { Some(q[1]) } else { None };
            let (_, blocks_over, chute_min, cells_above) = h1_info(&stacker);
            st.cells_sum += cells_above;
            let cur_ch = snap.queue[0].as_char();
            let hold_ch = snap.hold.or_else(|| snap.queue.get(1).copied()).map(|cc| cc.as_char());

            pgen.generate_for_state(&fst, &root_board, cur0, alt0, &mut root_places);
            if root_places.is_empty() { st.topout = true; st.pcs = 400; break; }

            let is_end = rem_g0 <= 7 && rem_g0 == g_rows;
            let eff_w = if is_end { beam_w.max(400) } else { beam_w };
            let eff_mpr = if is_end { 96 } else { (eff_w / 4).max(24) };
            let target_d = q_len.saturating_sub(if hold0.is_some() { 0 } else { 1 }).max(1);

            cur_beam.clear();
            seen.clear();
            let mut best_goal: Option<(usize, u16, i64)> = None;
            let mut root_ds = vec![false; root_places.len()];

            for (idx, &fp) in root_places.iter().enumerate() {
                let mut b = root_board;
                let o = &fst.shapes[fp.color_idx as usize].orients[fp.tf.2 as usize];
                let (lines, ds, _) = b.place_and_sift(o, fp.norm_i, fp.norm_j);
                root_ds[idx] = ds > 0;
                let rem_g = rem_g0.saturating_sub(ds as u16);
                if rem_g > 0 && !spawn_exact_cheese(&mut b, rem_g, lines, &ed.holes) { continue; }
                let (held, qc) = if hold0.is_some() {
                    if fp.did_hold { (cur0, 1.min(q_len)) } else { (hold0, 1.min(q_len)) }
                } else if fp.did_hold { (cur0, 2.min(q_len)) } else { (None, 1.min(q_len)) };
                if rem_g == 0 && b.rows < 20 {
                    let gr = -30_000_000 + 100 + b.rows as i64;
                    if best_goal.map_or(true, |x| gr < x.2) { best_goal = Some((1, idx as u16, gr)); }
                }
                let in_h = held.or_else(|| q.get(qc).copied());
                let mut sc = mdl.evaluate_board_with_bag(&b, rem_g, in_h, 0x7F);
                if anneal_pen > 0 && !is_end && target_d > 1 {
                    sc += (rem_g0.saturating_sub(rem_g) as i64) * anneal_pen;
                }
                if seen.insert(OKey { board: b, held, rem_g }) {
                    cur_beam.push(ONode { board: b, held, q_cursor: qc, rem_g, first_idx: idx as u16, score: sc });
                }
            }

            for depth in 2..=target_d {
                children.clear();
                let mid_pen = if anneal_pen > 0 && !is_end && depth < target_d {
                    (anneal_pen * ((target_d - depth) as i64)) / (target_d as i64)
                } else { 0 };
                for node in &cur_beam {
                    if node.rem_g == 0 { continue; }
                    let qc = node.q_cursor;
                    let (c_nh, c_h) = if node.held.is_some() { (q.get(qc).copied(), node.held) } else { (q.get(qc).copied(), q.get(qc + 1).copied()) };
                    pgen.generate_for_state(&fst, &node.board, c_nh, c_h, &mut tmp_places);
                    for &fp in &tmp_places {
                        let mut b = node.board;
                        let o = &fst.shapes[fp.color_idx as usize].orients[fp.tf.2 as usize];
                        let (lines, ds, _) = b.place_and_sift(o, fp.norm_i, fp.norm_j);
                        let rem_g = node.rem_g.saturating_sub(ds as u16);
                        if rem_g > 0 && !spawn_exact_cheese(&mut b, rem_g, lines, &ed.holes) { continue; }
                        let (held, nqc) = if node.held.is_some() {
                            if fp.did_hold { (c_nh, (qc + 1).min(q_len)) } else { (node.held, (qc + 1).min(q_len)) }
                        } else if fp.did_hold { (c_nh, (qc + 2).min(q_len)) } else { (None, (qc + 1).min(q_len)) };
                        if rem_g == 0 && b.rows < 20 {
                            let gr = -30_000_000 + (depth as i64) * 100 + b.rows as i64;
                            if best_goal.map_or(true, |x| gr < x.2) { best_goal = Some((depth, node.first_idx, gr)); }
                        }
                        let in_h = held.or_else(|| q.get(nqc).copied());
                        let mut sc = mdl.evaluate_board_with_bag(&b, rem_g, in_h, 0x7F);
                        if mid_pen > 0 {
                            sc += (rem_g0.saturating_sub(rem_g) as i64) * mid_pen;
                        }
                        children.push(ONode { board: b, held, q_cursor: nqc, rem_g, first_idx: node.first_idx, score: sc });
                    }
                }
                if children.is_empty() { break; }
                children.sort_unstable_by_key(|n| (n.score, n.first_idx));
                cur_beam.clear();
                seen.clear();
                let mut rc = vec![0usize; root_places.len()];
                for &ch in &children {
                    if cur_beam.len() >= eff_w { break; }
                    if rc[ch.first_idx as usize] >= eff_mpr { continue; }
                    if seen.insert(OKey { board: ch.board, held: ch.held, rem_g: ch.rem_g }) {
                        rc[ch.first_idx as usize] += 1;
                        cur_beam.push(ch);
                    }
                }
            }

            let chosen_idx = if let Some((_, ri, _)) = best_goal {
                ri as usize
            } else if roll_len > 0 && !is_end && !cur_beam.is_empty() {
                let mut cand_leaves: Vec<usize> = Vec::with_capacity(10);
                let mut used_roots: Vec<u16> = Vec::with_capacity(10);
                let mut has_0ds = false;
                let mut has_ds1 = false;
                for (li, n) in cur_beam.iter().enumerate() {
                    if !used_roots.contains(&n.first_idx) {
                        used_roots.push(n.first_idx);
                        cand_leaves.push(li);
                        if root_ds[n.first_idx as usize] { has_ds1 = true; } else { has_0ds = true; }
                    }
                    if cand_leaves.len() >= 8 { break; }
                }
                if !(has_0ds && has_ds1) {
                    for (li, n) in cur_beam.iter().enumerate() {
                        if used_roots.contains(&n.first_idx) { continue; }
                        let is_d = root_ds[n.first_idx as usize];
                        if (!has_0ds && !is_d) || (!has_ds1 && is_d) {
                            used_roots.push(n.first_idx);
                            cand_leaves.push(li);
                            if is_d { has_ds1 = true; } else { has_0ds = true; }
                            if has_0ds && has_ds1 { break; }
                        }
                    }
                }
                let mut best_ri = cur_beam[0].first_idx as usize;
                let mut best_rsc = i64::MAX;
                for li in cand_leaves {
                    let n = cur_beam[li];
                    let qt = &ed.pieces[(q_start + n.q_cursor).min(ed.pieces.len())..];
                    let r_sc = eval_roll_leaf(&mut rgen, &mut r_tmp_places, n.board, n.held, n.rem_g, qt, &ed.holes);
                    if r_sc < best_rsc {
                        best_rsc = r_sc;
                        best_ri = n.first_idx as usize;
                    }
                }
                best_ri
            } else if top_k_mean > 1 && !is_end && !cur_beam.is_empty() {
                let mut sums = vec![0i64; root_places.len()];
                let mut cnts = vec![0usize; root_places.len()];
                let mut best_sc = vec![i64::MAX; root_places.len()];
                for n in &cur_beam {
                    let ri = n.first_idx as usize;
                    if cnts[ri] < top_k_mean {
                        if cnts[ri] == 0 { best_sc[ri] = n.score; }
                        sums[ri] += n.score;
                        cnts[ri] += 1;
                    }
                }
                let mut best_ri = cur_beam[0].first_idx as usize;
                let mut best_agg = i64::MAX;
                for ri in 0..root_places.len() {
                    if cnts[ri] == 0 { continue; }
                    let padded = sums[ri] + ((top_k_mean - cnts[ri]) as i64) * (best_sc[ri] + 35);
                    let agg = padded / (top_k_mean as i64);
                    if agg < best_agg { best_agg = agg; best_ri = ri; }
                }
                best_ri
            } else if let Some(best) = cur_beam.first() {
                best.first_idx as usize
            } else { 0 };

            let fp = root_places[chosen_idx];
            let mut inputs = ai_in.inputs_for_place(&snap.matrix, fp).unwrap();
            inputs.pop();
            if fp.did_hold { if hold0.is_none() { q_pos += 2; } else { q_pos += 1; } } else { q_pos += 1; }
            stacker.run(inputs);
            let (lines, ds) = stacker.hard_drop();
            st.pcs += 1;
            ds_total += ds;
            st.nds += lines.saturating_sub(ds);
            if ds > 0 {
                st.ds_cnt += 1;
                if ds == 1 { st.ds1 += 1; } else if ds == 2 { st.ds2 += 1; } else { st.ds3 += 1; }
                let (_, new_bo, _, _) = h1_info(&stacker);
                if new_bo > 0 { st.ds_cov_next += 1; }
            } else {
                st.zds += 1;
                if rem_g0 <= 7 && rem_g0 == g_rows { st.zds_end += 1; }
                else if g_rows == 0 { st.zds_g0 += 1; }
                else if blocks_over > 0 { st.zds_h1_cov += 1; }
                else {
                    let is_o = fp.color_idx == 1;
                    let has_i = cur_ch == 'I' || hold_ch == Some('I');
                    if is_o { st.zds_o += 1; }
                    else if chute_min >= 1 && !has_i { st.zds_nar += 1; }
                    else { st.zds_wide += 1; }
                }
            }
        }
        if ds_total < 100 { st.topout = true; st.pcs = 400; }
        st.r_end = stacker.matrix().count();
        st
    };

    let workers = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(2).min(exact_data.len().max(1));
    let next_idx = std::sync::atomic::AtomicUsize::new(0);
    let mut results: Vec<Option<GameStat>> = vec![None; exact_data.len()];
    std::thread::scope(|sc| {
        let mut handles = Vec::new();
        for _ in 0..workers {
            handles.push(sc.spawn(|| {
                let mut local = Vec::new();
                loop {
                    let i = next_idx.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    if i >= exact_data.len() { break; }
                    local.push((i, run_one(&exact_data[i])));
                }
                local
            }));
        }
        for h in handles {
            for (i, st) in h.join().unwrap() { results[i] = Some(st); }
        }
    });
    let res: Vec<GameStat> = results.into_iter().map(|x| x.unwrap()).collect();
    print_summary(label, &res, t0.elapsed().as_secs_f64());
}

