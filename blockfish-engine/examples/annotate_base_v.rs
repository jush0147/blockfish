use blockfish::ai::{fast::FastBoard, spatial_val::SpatialValueModel};
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::time::Instant;

fn avail_mask_from_rec(held: u8, next5: &[u8; 5], bag_mask: u8) -> u8 {
    let mut m = 0u8;
    if held < 7 {
        m |= 1u8 << held;
    }
    for &p in next5 {
        if p < 7 {
            m |= 1u8 << p;
        }
    }
    if m == 0 {
        if bag_mask != 0 { bag_mask } else { 0x7F }
    } else {
        m
    }
}

fn eval_rec(sm: &SpatialValueModel, rec: &[u8]) -> f32 {
    let mut b = FastBoard::empty();
    for r in 0..16 {
        let lo = rec[2 * r] as u16;
        let hi = rec[2 * r + 1] as u16;
        b.data[r] = 0xFC00 | ((lo | (hi << 8)) & 0x3FF);
    }
    b.rows = rec[32];
    b.garbage_rows = rec[33];
    let rem_g = rec[34] as u16;
    let held = rec[35];
    let mut next5 = [7u8; 5];
    next5.copy_from_slice(&rec[36..41]);
    let bag_mask = rec[41];
    let in_h = if held < 7 {
        Some(held)
    } else if next5[0] < 7 {
        Some(next5[0])
    } else {
        None
    };
    let av = avail_mask_from_rec(if held < 7 { held } else { 7 }, &next5, bag_mask);
    let raw = sm.evaluate_board_with_bag(&b, rem_g, in_h, av);
    ((raw as f32) / 100.0).clamp(-50.0, 500.0)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 4 {
        eprintln!("Usage: annotate_base_v <model.txt> <in_bfds1.bin> <out_bfds2.bin>");
        std::process::exit(1);
    }
    let model_path = &args[1];
    let in_path = &args[2];
    let out_path = &args[3];
    let threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4);

    let t0 = Instant::now();
    let mut sm = SpatialValueModel::load_from_file(model_path).expect("load model");
    sm.prepare_fast();

    let mut f_in = BufReader::new(File::open(in_path).expect("open input"));
    let mut buf = Vec::new();
    f_in.read_to_end(&mut buf).expect("read input");
    assert!(&buf[0..8] == b"BFDS0001", "expected BFDS0001 input");
    let n_games = u32::from_le_bytes([buf[8], buf[9], buf[10], buf[11]]);
    let n_decs = u32::from_le_bytes([buf[12], buf[13], buf[14], buf[15]]) as usize;

    // Index all decisions
    let mut dec_offs = Vec::with_capacity(n_decs);
    let mut dec_nms = Vec::with_capacity(n_decs);
    let mut pos = 16usize;
    let mut total_moves = 0usize;
    for _ in 0..n_decs {
        dec_offs.push(pos);
        let nm = buf[pos + 8] as usize;
        dec_nms.push(nm);
        total_moves += nm;
        pos += 12 + nm * 44;
    }

    // Parallel annotate all moves across CPU cores
    let mut base_vs = vec![0.0f32; total_moves];
    let mut mv_starts = Vec::with_capacity(n_decs);
    let mut cur = 0usize;
    for &nm in &dec_nms {
        mv_starts.push(cur);
        cur += nm;
    }

    let chunk_decs = (n_decs + threads - 1) / threads;
    let buf_ref = &buf;
    let sm_ref = &sm;
    let dec_offs_ref = &dec_offs;
    let dec_nms_ref = &dec_nms;

    let mut slices: Vec<(usize, usize, &mut [f32])> = Vec::new();
    let mut rem_slice = base_vs.as_mut_slice();
    for t in 0..threads {
        let d0 = (t * chunk_decs).min(n_decs);
        let d1 = ((t + 1) * chunk_decs).min(n_decs);
        if d0 >= d1 {
            break;
        }
        let m_cnt: usize = dec_nms[d0..d1].iter().sum();
        let (head, tail) = rem_slice.split_at_mut(m_cnt);
        slices.push((d0, d1, head));
        rem_slice = tail;
    }

    std::thread::scope(|s| {
        for (d0, d1, out_sl) in slices {
            s.spawn(move || {
                let mut idx = 0usize;
                for di in d0..d1 {
                    let off = dec_offs_ref[di] + 12;
                    let nm = dec_nms_ref[di];
                    for m in 0..nm {
                        let mo = off + m * 44;
                        out_sl[idx] = eval_rec(sm_ref, &buf_ref[mo..mo + 44]);
                        idx += 1;
                    }
                }
            });
        }
    });

    // Write BFDS0002 output
    let mut w = BufWriter::with_capacity(1 << 20, File::create(out_path).expect("create out"));
    w.write_all(b"BFDS0002").unwrap();
    w.write_all(&n_games.to_le_bytes()).unwrap();
    w.write_all(&(n_decs as u32).to_le_bytes()).unwrap();
    let mut mv_idx = 0usize;
    for di in 0..n_decs {
        let off = dec_offs[di];
        let nm = dec_nms[di];
        w.write_all(&buf[off..off + 12]).unwrap();
        for m in 0..nm {
            let mo = off + 12 + m * 44;
            w.write_all(&buf[mo..mo + 44]).unwrap();
            w.write_all(&base_vs[mv_idx].to_le_bytes()).unwrap();
            mv_idx += 1;
        }
    }
    w.flush().unwrap();
    println!(
        "Annotated {} games, {} decs, {} moves -> {} in {:.2}s",
        n_games, n_decs, total_moves, out_path, t0.elapsed().as_secs_f64()
    );
}
