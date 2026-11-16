use block_stacker::{Config as BSConfig, Ruleset, Stacker};
use blockfish::{
    ai::{
        cheese_val::{FEAT_DIM, MLP_HIDDEN},
        fast::{color_to_idx, FastBoard, FastPlaceGen, FastShapeTable},
        BeamConfig, FastAI, LearnedValueModel,
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
    let mid = seeds.len() / 2;
    let (s1, s2) = seeds.split_at(mid);
    let run_slice = |sl: &[u64]| -> Vec<GameStat> {
        sl.iter().map(|&s| {
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
                        let has_o = cur_c == 'O' || hold_c == Some('O');
                        let has_i = cur_c == 'I' || hold_c == Some('I');
                        if has_o { st.zds_o += 1; }
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
        }).collect()
    };

    let res: Vec<GameStat> = std::thread::scope(|sc| {
        let h = sc.spawn(|| run_slice(s1));
        let mut r2 = run_slice(s2);
        let mut r1 = h.join().unwrap();
        r1.append(&mut r2);
        r1
    });
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

fn run_oracle_suite(label: &str, exact_data: &[ExactSeedData], horizon: usize, beam_w: usize, exact_cheese: bool, mdl: &LearnedValueModel) {
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
                let sc = mdl.evaluate_board_with_bag(&b, rem_g, in_h, 0x7F);
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
                        let sc = mdl.evaluate_board_with_bag(&b, rem_g, in_h, 0x7F);
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
                    let has_o = cur_ch == 'O' || hold_ch == Some('O');
                    let has_i = cur_ch == 'I' || hold_ch == Some('I');
                    if has_o { st.zds_o += 1; }
                    else if chute_min >= 1 && !has_i { st.zds_nar += 1; }
                    else { st.zds_wide += 1; }
                }
            }
        }
        if ds_total < 100 { st.topout = true; st.pcs = 400; }
        st.r_end = stacker.matrix().count();
        st
    };

    let (s1, s2) = exact_data.split_at(exact_data.len() / 2);
    let res: Vec<GameStat> = std::thread::scope(|sc| {
        let h = sc.spawn(|| s1.iter().map(run_one).collect::<Vec<_>>());
        let mut r2: Vec<GameStat> = s2.iter().map(run_one).collect();
        let mut r1 = h.join().unwrap();
        r1.append(&mut r2);
        r1
    });
    print_summary(label, &res, t0.elapsed().as_secs_f64());
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
    } else if mode == "legal20" {
        println!("=== Legal Held-out Dev20 (seeds 201..=220) ===");
        run_legal_suite("LEGAL_h2nog0_p5_b120", &seeds20, &cfg_legal);
    }
}
