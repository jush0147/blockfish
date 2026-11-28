//! Native BC-4 Mass Legal Replay Player for Blockfish SDL2 GUI.
//!
//! Plays back recorded BC-4 Mass Legal Replay JSON files (`bc4_mass_legal_v1` schema)
//! using `block-stacker`'s Guideline ruleset and the existing Blockfish SDL2 `View`.
//! Never invokes AI search or `best_move()`.

use crate::view::{Quit, View};
use block_stacker::{Config as BSConfig, PieceType, Ruleset, Stacker};
use blockfish::{Input, StackerExt as _};
use sdl2::{
    event::Event,
    keyboard::{Keycode, Mod},
};
use serde::Deserialize;
use std::path::Path;
use std::rc::Rc;
use std::time::{Duration, Instant};
use thiserror::Error;

const SPEED_PRESETS: &[f64] = &[0.5, 1.0, 1.5, 2.5, 4.0, 6.0, 10.0, 15.0, 25.0];

#[derive(Debug, Error)]
pub enum ReplayLoadError {
    #[error("failed to read replay file: {0}")]
    Io(#[from] std::io::Error),
    #[error("failed to parse replay JSON: {0}")]
    Json(#[from] serde_json::Error),
}

#[derive(Clone, Debug, Error)]
#[error("replay mismatch at step {step} ({field}): expected {expected}, actual {actual}")]
pub struct ReplayValidationError {
    pub step: usize,
    pub field: &'static str,
    pub expected: String,
    pub actual: String,
}

#[derive(Clone, Debug, Deserialize)]
pub struct ReplayFile {
    pub record: ReplayRecord,
    pub moves: Vec<ReplayMove>,
}

#[derive(Clone, Debug, Deserialize)]
pub struct ReplayRecord {
    pub schema_version: Option<String>,
    pub record_type: Option<String>,
    pub seed: u64,
    pub status: String,
    pub completed_100l: bool,
    pub pieces: usize,
    pub lines_cleared: usize,
    pub ds_cleared: usize,
    pub model_name: Option<String>,
    pub model_sha256: Option<String>,
    pub batch_id: Option<String>,
    pub ruleset: Option<ReplayRuleset>,
}

#[derive(Clone, Debug, Deserialize)]
pub struct ReplayRuleset {
    pub garbage_total: Option<usize>,
    pub garbage_min_height: Option<usize>,
    pub garbage_max_height: Option<usize>,
    pub previews: Option<usize>,
}

#[derive(Clone, Debug, Deserialize)]
pub struct ReplayMove {
    pub step: usize,
    pub cur: char,
    pub hold_before: Option<char>,
    pub queue_p5: String,
    pub did_hold: bool,
    pub placed_piece: char,
    pub inputs: Vec<String>,
    pub lines_cleared: usize,
    pub ds_cleared: usize,
    pub ds_total: usize,
    pub g_rows_before: Option<u16>,
    pub rem_g_before: Option<u16>,
}

/// Cached snapshot of the exact game state at a given step in the replay.
#[derive(Clone)]
pub struct ReplayFrame {
    pub step: usize,
    pub stacker: Stacker,
    pub pieces: usize,
    pub lines_cleared: usize,
    pub color_clears: usize,
    pub ds_total: usize,
    pub last_move: Option<ReplayMove>,
    /// Preview of the upcoming step's landing ghost (board + ghost piece before hard_drop),
    /// computed strictly from the recorded replay inputs (no AI).
    pub next_preview_piece: Option<(PieceType, i16, i16, i32)>,
}

/// Verified, pre-cached replay session allowing O(1) forward/backward stepping and jumping.
pub struct ReplaySession {
    pub record: ReplayRecord,
    pub total_recorded_moves: usize,
    pub frames: Vec<ReplayFrame>,
    pub validation_error: Option<ReplayValidationError>,
}

fn parse_input_token(tok: &str) -> Option<Input> {
    match tok {
        "Left" => Some(Input::Left),
        "Right" => Some(Input::Right),
        "CW" => Some(Input::CW),
        "CCW" => Some(Input::CCW),
        "SD" => Some(Input::SD),
        "HD" => Some(Input::HD),
        "Hold" => Some(Input::Hold),
        _ => None,
    }
}

fn visible_queue_string(stacker: &Stacker) -> Option<String> {
    let cur = stacker.current_piece_type()?;
    let mut s = String::with_capacity(1 + stacker.next().len());
    s.push(cur);
    for &p in stacker.next() {
        s.push(p);
    }
    Some(s)
}

impl ReplaySession {
    pub fn load_from_file(path: &Path, rules: Rc<Ruleset>) -> Result<Self, ReplayLoadError> {
        let raw = std::fs::read_to_string(path)?;
        let file: ReplayFile = serde_json::from_str(&raw)?;
        Ok(Self::from_replay_file(file, rules))
    }

    pub fn from_replay_file(file: ReplayFile, rules: Rc<Ruleset>) -> Self {
        let mut game_cfg = BSConfig::default();
        game_cfg.prng_seed = Some(file.record.seed);
        let rs = file.record.ruleset.as_ref();
        let goal = rs.and_then(|r| r.garbage_total).unwrap_or(100);
        game_cfg.garbage.total_lines = Some(goal);
        game_cfg.garbage.min_height = rs.and_then(|r| r.garbage_min_height).unwrap_or(4);
        game_cfg.garbage.max_height = rs.and_then(|r| r.garbage_max_height).unwrap_or(9);

        let mut stacker = Stacker::new(rules.clone(), game_cfg);
        let total_recorded_moves = file.moves.len();
        let mut frames = Vec::with_capacity(total_recorded_moves + 1);
        frames.push(ReplayFrame {
            step: 0,
            stacker: stacker.clone(),
            pieces: 0,
            lines_cleared: 0,
            color_clears: 0,
            ds_total: 0,
            last_move: None,
            next_preview_piece: None,
        });

        let mut validation_error: Option<ReplayValidationError> = None;
        let mut pieces = 0usize;
        let mut lines_total = 0usize;
        let mut color_clears = 0usize;
        let mut ds_total = 0usize;

        if let Some(expected_p) = rs.and_then(|r| r.previews) {
            if rules.previews != expected_p {
                validation_error = Some(ReplayValidationError {
                    step: 0,
                    field: "ruleset.previews",
                    expected: expected_p.to_string(),
                    actual: rules.previews.to_string(),
                });
            }
        }

        if validation_error.is_none() {
            for (idx, mv) in file.moves.iter().enumerate() {
                let expected_step = idx + 1;
                if mv.step != expected_step {
                    validation_error = Some(ReplayValidationError {
                        step: expected_step,
                        field: "step",
                        expected: expected_step.to_string(),
                        actual: mv.step.to_string(),
                    });
                    break;
                }

                let actual_cur = stacker.current_piece_type();
                if actual_cur != Some(mv.cur) {
                    validation_error = Some(ReplayValidationError {
                        step: expected_step,
                        field: "cur",
                        expected: mv.cur.to_string(),
                        actual: format!("{:?}", actual_cur),
                    });
                    break;
                }

                let actual_hold = stacker.held();
                if actual_hold != mv.hold_before {
                    validation_error = Some(ReplayValidationError {
                        step: expected_step,
                        field: "hold_before",
                        expected: format!("{:?}", mv.hold_before),
                        actual: format!("{:?}", actual_hold),
                    });
                    break;
                }

                let actual_queue = visible_queue_string(&stacker).unwrap_or_default();
                if actual_queue != mv.queue_p5 {
                    validation_error = Some(ReplayValidationError {
                        step: expected_step,
                        field: "queue_p5",
                        expected: mv.queue_p5.clone(),
                        actual: actual_queue,
                    });
                    break;
                }

                let mut parsed_inputs = Vec::with_capacity(mv.inputs.len());
                let mut bad_tok: Option<String> = None;
                for tok in &mv.inputs {
                    match parse_input_token(tok) {
                        Some(inp) => parsed_inputs.push(inp),
                        None => {
                            bad_tok = Some(tok.clone());
                            break;
                        }
                    }
                }
                if let Some(tok) = bad_tok {
                    validation_error = Some(ReplayValidationError {
                        step: expected_step,
                        field: "inputs",
                        expected: "valid Input token".to_string(),
                        actual: tok,
                    });
                    break;
                }

                if parsed_inputs.last() != Some(&Input::HD) {
                    validation_error = Some(ReplayValidationError {
                        step: expected_step,
                        field: "inputs.last",
                        expected: "HD".to_string(),
                        actual: format!("{:?}", parsed_inputs.last()),
                    });
                    break;
                }
                parsed_inputs.pop();

                let actual_did_hold = parsed_inputs.first() == Some(&Input::Hold);
                if actual_did_hold != mv.did_hold {
                    validation_error = Some(ReplayValidationError {
                        step: expected_step,
                        field: "did_hold",
                        expected: mv.did_hold.to_string(),
                        actual: actual_did_hold.to_string(),
                    });
                    break;
                }

                // Execute inputs up to hard drop, record exact landing ghost for previous frame's preview
                stacker.run(parsed_inputs);
                let placed_actual = stacker.current_piece_type();
                if placed_actual != Some(mv.placed_piece) {
                    validation_error = Some(ReplayValidationError {
                        step: expected_step,
                        field: "placed_piece",
                        expected: mv.placed_piece.to_string(),
                        actual: format!("{:?}", placed_actual),
                    });
                    break;
                }
                let landing_ghost = stacker.current_piece_ghost();
                if let Some(prev) = frames.last_mut() {
                    prev.next_preview_piece = landing_ghost;
                }

                let (lc, ds) = stacker.hard_drop();
                pieces += 1;
                lines_total += lc;
                ds_total += ds;
                if stacker.is_matrix_colorless() {
                    color_clears += 1;
                }

                if lc != mv.lines_cleared {
                    validation_error = Some(ReplayValidationError {
                        step: expected_step,
                        field: "lines_cleared",
                        expected: mv.lines_cleared.to_string(),
                        actual: lc.to_string(),
                    });
                    break;
                }
                if ds != mv.ds_cleared {
                    validation_error = Some(ReplayValidationError {
                        step: expected_step,
                        field: "ds_cleared",
                        expected: mv.ds_cleared.to_string(),
                        actual: ds.to_string(),
                    });
                    break;
                }
                if ds_total != mv.ds_total {
                    validation_error = Some(ReplayValidationError {
                        step: expected_step,
                        field: "ds_total",
                        expected: mv.ds_total.to_string(),
                        actual: ds_total.to_string(),
                    });
                    break;
                }

                frames.push(ReplayFrame {
                    step: expected_step,
                    stacker: stacker.clone(),
                    pieces,
                    lines_cleared: lines_total,
                    color_clears,
                    ds_total,
                    last_move: Some(mv.clone()),
                    next_preview_piece: None,
                });
            }
        }

        // Verify final summary counters against record header
        if validation_error.is_none() {
            if pieces != file.record.pieces {
                validation_error = Some(ReplayValidationError {
                    step: pieces,
                    field: "record.pieces",
                    expected: file.record.pieces.to_string(),
                    actual: pieces.to_string(),
                });
            } else if lines_total != file.record.lines_cleared {
                validation_error = Some(ReplayValidationError {
                    step: pieces,
                    field: "record.lines_cleared",
                    expected: file.record.lines_cleared.to_string(),
                    actual: lines_total.to_string(),
                });
            } else if ds_total != file.record.ds_cleared {
                validation_error = Some(ReplayValidationError {
                    step: pieces,
                    field: "record.ds_cleared",
                    expected: file.record.ds_cleared.to_string(),
                    actual: ds_total.to_string(),
                });
            } else if (ds_total >= goal) != file.record.completed_100l {
                validation_error = Some(ReplayValidationError {
                    step: pieces,
                    field: "record.completed_100l",
                    expected: file.record.completed_100l.to_string(),
                    actual: (ds_total >= goal).to_string(),
                });
            }
        }

        Self {
            record: file.record,
            total_recorded_moves,
            frames,
            validation_error,
        }
    }

    pub fn max_valid_step(&self) -> usize {
        self.frames.len().saturating_sub(1)
    }

    pub fn frame(&self, step: usize) -> &ReplayFrame {
        let clamped = step.min(self.max_valid_step());
        &self.frames[clamped]
    }
}

/// Interactive SDL2 replay controller using Blockfish's existing `View`.
pub struct ReplayPlayer<'v> {
    view: View<'v>,
    session: ReplaySession,
    step: usize,
    playing: bool,
    speed_idx: usize,
    speed_pps: f64,
    last_tick: Instant,
    accum_sec: f64,
    goto_input: Option<String>,
}

impl<'v> ReplayPlayer<'v> {
    pub fn new(
        mut view: View<'v>,
        session: ReplaySession,
        start_step: usize,
        initial_speed_pps: f64,
        start_paused: bool,
    ) -> Self {
        let clamped_step = start_step.min(session.max_valid_step());
        // If validation failed, stop playback immediately and jump to the last valid step
        // so the user sees the error right where it occurred.
        let (step, playing) = if session.validation_error.is_some() {
            (session.max_valid_step(), false)
        } else {
            (
                clamped_step,
                !start_paused && clamped_step < session.max_valid_step(),
            )
        };

        let speed_pps = initial_speed_pps.clamp(0.25, 60.0);
        let speed_idx = SPEED_PRESETS
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| {
                ((*a) - speed_pps)
                    .abs()
                    .partial_cmp(&((*b) - speed_pps).abs())
                    .unwrap()
            })
            .map(|(i, _)| i)
            .unwrap_or(3);

        view.set_replay_motd(session.record.seed, session.record.pieces);
        view.set_replay_help();

        let mut player = Self {
            view,
            session,
            step,
            playing,
            speed_idx,
            speed_pps,
            last_tick: Instant::now(),
            accum_sec: 0.0,
            goto_input: None,
        };
        player.sync_view();
        player
    }

    pub fn view(&self) -> &View<'v> {
        &self.view
    }

    pub fn total_steps(&self) -> usize {
        self.session.max_valid_step()
    }

    pub fn validation_error(&self) -> Option<&ReplayValidationError> {
        self.session.validation_error.as_ref()
    }

    pub fn record(&self) -> &ReplayRecord {
        &self.session.record
    }

    pub fn set_export_step(&mut self, step: usize, playing: bool) {
        self.step = step.min(self.session.max_valid_step());
        self.playing = playing;
        self.goto_input = None;
        self.sync_view();
    }

    pub fn tick(&mut self) {
        let now = Instant::now();
        let dt = now.duration_since(self.last_tick).as_secs_f64();
        self.last_tick = now;

        if !self.playing || self.session.validation_error.is_some() {
            self.accum_sec = 0.0;
            return;
        }

        let step_interval = 1.0 / self.speed_pps;
        self.accum_sec += dt;
        let mut changed = false;
        while self.accum_sec >= step_interval {
            self.accum_sec -= step_interval;
            if self.step < self.session.max_valid_step() {
                self.step += 1;
                changed = true;
            } else {
                self.playing = false;
                self.accum_sec = 0.0;
                changed = true;
                break;
            }
        }
        if changed {
            self.sync_view();
        }
    }

    pub fn handle_event(&mut self, evt: Event) -> Result<(), Quit> {
        match evt {
            Event::Quit { .. } => return Err(Quit),
            Event::KeyDown {
                keycode: Some(kc),
                keymod,
                ..
            } => self.handle_key(kc, keymod)?,
            _ => {}
        }
        Ok(())
    }

    fn handle_key(&mut self, kc: Keycode, keymod: Mod) -> Result<(), Quit> {
        // Handle Goto-step numeric entry mode first
        if let Some(ref mut buf) = self.goto_input {
            match kc {
                Keycode::Escape => {
                    self.goto_input = None;
                    self.sync_view();
                    return Ok(());
                }
                Keycode::Return | Keycode::KpEnter => {
                    if let Ok(target) = buf.parse::<usize>() {
                        self.jump_to(target);
                    }
                    self.goto_input = None;
                    self.sync_view();
                    return Ok(());
                }
                Keycode::Backspace => {
                    buf.pop();
                    self.sync_view();
                    return Ok(());
                }
                _ => {
                    if let Some(ch) = keycode_digit(kc) {
                        if buf.len() < 5 {
                            buf.push(ch);
                            self.sync_view();
                        }
                    }
                    return Ok(());
                }
            }
        }

        let shift = keymod.contains(Mod::LSHIFTMOD) || keymod.contains(Mod::RSHIFTMOD);
        match kc {
            Keycode::Escape | Keycode::Q => return Err(Quit),
            Keycode::Space | Keycode::P => {
                if self.session.validation_error.is_none() {
                    if !self.playing && self.step >= self.session.max_valid_step() {
                        self.step = 0;
                    }
                    self.playing = !self.playing;
                    self.last_tick = Instant::now();
                    self.accum_sec = 0.0;
                    self.sync_view();
                }
            }
            Keycode::Right | Keycode::Period => {
                self.playing = false;
                let delta = if shift { 10 } else { 1 };
                self.jump_to(self.step.saturating_add(delta));
            }
            Keycode::Left | Keycode::Comma => {
                self.playing = false;
                let delta = if shift { 10 } else { 1 };
                self.jump_to(self.step.saturating_sub(delta));
            }
            Keycode::PageDown => {
                self.playing = false;
                self.jump_to(self.step.saturating_add(10));
            }
            Keycode::PageUp => {
                self.playing = false;
                self.jump_to(self.step.saturating_sub(10));
            }
            Keycode::R | Keycode::Home => {
                self.playing = false;
                self.jump_to(0);
            }
            Keycode::End => {
                self.playing = false;
                self.jump_to(self.session.max_valid_step());
            }
            Keycode::Up | Keycode::Equals | Keycode::Plus | Keycode::KpPlus => {
                if self.speed_idx + 1 < SPEED_PRESETS.len() {
                    self.speed_idx += 1;
                    self.speed_pps = SPEED_PRESETS[self.speed_idx];
                    self.sync_view();
                }
            }
            Keycode::Down | Keycode::Minus | Keycode::KpMinus => {
                if self.speed_idx > 0 {
                    self.speed_idx -= 1;
                    self.speed_pps = SPEED_PRESETS[self.speed_idx];
                    self.sync_view();
                }
            }
            Keycode::G => {
                self.playing = false;
                self.goto_input = Some(String::new());
                self.sync_view();
            }
            _ => {}
        }
        Ok(())
    }

    fn jump_to(&mut self, target: usize) {
        self.step = target.min(self.session.max_valid_step());
        self.accum_sec = 0.0;
        self.last_tick = Instant::now();
        self.sync_view();
    }

    fn sync_view(&mut self) {
        let frame = self.session.frame(self.step);
        let stacker = &frame.stacker;

        // Reuse Blockfish's native board, Hold, Next Queue, piece/ghost, and stats renderer
        let next = stacker.next().iter().cloned();
        self.view.set_queue(next, stacker.held());
        self.view.set_piece(stacker.current_piece());
        self.view.set_matrix(stacker.matrix());
        self.view.set_stats(
            frame.pieces,
            frame.lines_cleared,
            frame.color_clears,
            frame.ds_total,
            0,
            stacker.config().garbage.total_lines,
        );

        // Reuse right-side matrix to show the upcoming recorded piece landing preview (no AI)
        self.view.set_engine_matrix(stacker.matrix());
        self.view.set_engine_piece(frame.next_preview_piece);

        let total = self.session.total_recorded_moves;
        let state_label = if self.session.validation_error.is_some() {
            "REPLAY ERR"
        } else if self.goto_input.is_some() {
            "GOTO STEP"
        } else if self.step >= self.session.max_valid_step() {
            "REPLAY DONE"
        } else if self.playing {
            "PLAYING"
        } else {
            "PAUSED"
        };

        let line1 = if let Some(ref buf) = self.goto_input {
            format!("step: {}_", buf)
        } else {
            format!("{}/{} {:.1}p/s", self.step, total, self.speed_pps)
        };

        let line2 = format!("seed {}", self.session.record.seed);

        let line3 = if let Some(ref err) = self.session.validation_error {
            format!("s{}: {}", err.step, err.field)
        } else if let Some(ref mv) = frame.last_move {
            let hold_tag = if mv.did_hold { "H " } else { "" };
            format!("{}{} +{}L/+{}D", hold_tag, mv.placed_piece, mv.lines_cleared, mv.ds_cleared)
        } else {
            "start".to_string()
        };

        self.view
            .set_replay_overlay([state_label.to_string(), line1, line2, line3]);

        let status_bar = if let Some(ref err) = self.session.validation_error {
            format!("VALIDATION ERROR: {}", err)
        } else {
            let sha_short = self
                .session
                .record
                .model_sha256
                .as_deref()
                .map(|s| &s[..s.len().min(8)])
                .unwrap_or("n/a");
            let last_inputs = frame
                .last_move
                .as_ref()
                .map(|m| m.inputs.join(" "))
                .unwrap_or_else(|| "(start)".to_string());
            format!(
                "BC-4 Replay (AI OFF) | seed={} | sha={} | step={}/{} | inputs: {}",
                self.session.record.seed, sha_short, self.step, total, last_inputs
            )
        };
        self.view.set_replay_status(&status_bar);
    }

    pub fn idle_sleep(&self) -> Duration {
        Duration::from_millis(8)
    }
}

fn keycode_digit(kc: Keycode) -> Option<char> {
    match kc {
        Keycode::Num0 | Keycode::Kp0 => Some('0'),
        Keycode::Num1 | Keycode::Kp1 => Some('1'),
        Keycode::Num2 | Keycode::Kp2 => Some('2'),
        Keycode::Num3 | Keycode::Kp3 => Some('3'),
        Keycode::Num4 | Keycode::Kp4 => Some('4'),
        Keycode::Num5 | Keycode::Kp5 => Some('5'),
        Keycode::Num6 | Keycode::Kp6 => Some('6'),
        Keycode::Num7 | Keycode::Kp7 => Some('7'),
        Keycode::Num8 | Keycode::Kp8 => Some('8'),
        Keycode::Num9 | Keycode::Kp9 => Some('9'),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detects_mismatch_and_halts_validation() {
        let rules = Rc::new(Ruleset::guideline());
        let mut cfg = BSConfig::default();
        cfg.prng_seed = Some(102015);
        cfg.garbage.total_lines = Some(100);
        let stacker = Stacker::new(rules.clone(), cfg);
        let cur = stacker.current_piece_type().unwrap();
        let wrong_cur = if cur == 'I' { 'O' } else { 'I' };

        let bad_file = ReplayFile {
            record: ReplayRecord {
                schema_version: Some("bc4_mass_legal_v1".into()),
                record_type: None,
                seed: 102015,
                status: "completed".into(),
                completed_100l: true,
                pieces: 1,
                lines_cleared: 0,
                ds_cleared: 0,
                model_name: None,
                model_sha256: None,
                batch_id: None,
                ruleset: None,
            },
            moves: vec![ReplayMove {
                step: 1,
                cur: wrong_cur,
                hold_before: None,
                queue_p5: "INVALID".into(),
                did_hold: false,
                placed_piece: wrong_cur,
                inputs: vec!["HD".into()],
                lines_cleared: 0,
                ds_cleared: 0,
                ds_total: 0,
                g_rows_before: None,
                rem_g_before: None,
            }],
        };

        let session = ReplaySession::from_replay_file(bad_file, rules);
        let err = session
            .validation_error
            .as_ref()
            .expect("mismatch must be detected and reported");
        assert_eq!(err.step, 1);
        assert_eq!(err.field, "cur");
        assert_eq!(session.max_valid_step(), 0);
    }
}
