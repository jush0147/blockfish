#![windows_subsystem = "windows"]

mod controller;
mod controls;
mod replay;
mod resources;
mod theme;
mod timer;
mod util;
mod view;

use block_stacker::{Config as BSConfig, Ruleset};
use blockfish::Config as BFConfig;

use argh::FromArgs;
use thiserror::Error;

const DEFAULT_CONFIG_DIR: &str = "./config";

// Error handling
#[derive(Debug, Error)]
enum Error {
    #[error("{0}")]
    Sdl(String),
    #[error("failed to load resources")]
    Resources(#[from] resources::ResourceLoadError),
    #[error("failed to parse {0} config")]
    ParseConfig(&'static str, #[source] serde_json::Error),
    #[error("failed to load replay")]
    ReplayLoad(#[from] replay::ReplayLoadError),
    #[error("{0}")]
    ReplayValidation(#[from] replay::ReplayValidationError),
}

fn sdl_error(e: impl std::fmt::Display) -> Error {
    Error::Sdl(format!("{}", e))
}

// Args

/// Blockfish GUI client.
#[derive(FromArgs)]
struct Args {
    #[argh(positional)]
    goal: Option<Goal>,
    /// configuration directory
    #[argh(option, short = 'c')]
    config_dir: Option<std::path::PathBuf>,
    /// garbage level, defaults to 9
    #[argh(option, short = 'g')]
    garbage: Option<usize>,
    /// minimum garbage level, defaults to 2
    #[argh(option, short = 'G')]
    min_garbage: Option<usize>,
    /// integer used to seed the random number generator
    #[argh(option, short = 's')]
    seed: Option<u64>,
    /// integer used to seed the random number generator
    #[argh(option, short = 'A')]
    ai_params: Option<BFConfig>,
    /// path to a BC-4 Mass Legal Replay JSON file to launch in Replay Mode (AI disabled)
    #[argh(option)]
    replay: Option<std::path::PathBuf>,
    /// initial step to jump to in Replay Mode (default: 0)
    #[argh(option)]
    start_step: Option<usize>,
    /// initial playback speed in pieces per second in Replay Mode (default: 2.5)
    #[argh(option)]
    speed: Option<f64>,
    /// start Replay Mode paused instead of auto-playing
    #[argh(switch)]
    paused: bool,
    /// headlessly verify the --replay file step-by-step and exit without opening a window
    #[argh(switch)]
    verify_only: bool,
    /// render a single Replay Mode frame at --start-step, save BMP to path, and exit
    #[argh(option)]
    screenshot: Option<std::path::PathBuf>,
}

impl Args {
    fn game_config(&self) -> BSConfig {
        let mut cfg = BSConfig::default();
        cfg.prng_seed = self.seed;
        cfg.garbage.total_lines = self.goal.unwrap_or_default().lines();
        if let Some(h) = self.garbage {
            cfg.garbage.max_height = h;
        }
        if let Some(h) = self.min_garbage {
            cfg.garbage.min_height = h;
        }
        cfg
    }

    fn ai_config(&self) -> BFConfig {
        self.ai_params.clone().unwrap_or_default()
    }
}

#[derive(Copy, Clone, Debug)]
enum Goal {
    Infinite,
    Lines(usize),
}

impl Goal {
    fn lines(&self) -> Option<usize> {
        match *self {
            Goal::Infinite => None,
            Goal::Lines(n) => Some(n),
        }
    }
}

impl Default for Goal {
    fn default() -> Self {
        Goal::Lines(100)
    }
}

#[derive(Debug, Error)]
#[error("invalid goal, expected number or 'infinite'")]
struct ParseGoalError;

impl std::str::FromStr for Goal {
    type Err = ParseGoalError;
    fn from_str(s: &str) -> std::result::Result<Self, ParseGoalError> {
        if s == "infinite" {
            Ok(Goal::Infinite)
        } else {
            s.parse()
                .map(|n| Goal::Lines(n))
                .map_err(|_| ParseGoalError)
        }
    }
}

// main

type Result<T> = std::result::Result<T, Error>;

pub fn main() {
    pretty_env_logger::init();
    std::process::exit(match entry(argh::from_env()) {
        Ok(_) => 0,
        Err(err) => {
            error_trace_log(&err);
            #[cfg(feature = "msgbox")]
            error_trace_msgbox(&err);
            1
        }
    });
}

fn error_trace_log(err: &(dyn std::error::Error + 'static)) {
    let mut trace = Some(err);
    while let Some(err) = trace {
        log::error!("{}", err);
        eprintln!("Error: {}", err);
        trace = err.source();
    }
}

#[cfg(feature = "msgbox")]
fn error_trace_msgbox(err: &(dyn std::error::Error + 'static)) {
    use std::fmt::Write as _;
    let mut contents = "Error: ".to_string();
    let mut trace = Some(err);
    while let Some(err) = trace {
        let _ = write!(&mut contents, "{}\n", err);
        trace = err.source();
    }
    let _ = msgbox::create("Blockfish crashed", &contents, msgbox::IconType::Error);
}

// Entry point

fn entry(mut args: Args) -> Result<()> {
    let rules = std::rc::Rc::new(Ruleset::guideline());

    // Headless verification mode for --replay --verify-only (no SDL2 required)
    if args.verify_only {
        let path = args.replay.as_ref().ok_or_else(|| {
            Error::Sdl("--verify-only requires --replay <path>".to_string())
        })?;
        let session = replay::ReplaySession::load_from_file(path, rules)?;
        if let Some(err) = session.validation_error {
            return Err(Error::ReplayValidation(err));
        }
        let last = session.frame(session.max_valid_step());
        println!(
            "REPLAY VERIFIED OK: file={} seed={} status={} pieces={} lines={} ds={}/100 steps={}",
            path.display(),
            session.record.seed,
            session.record.status,
            last.pieces,
            last.lines_cleared,
            last.ds_total,
            session.max_valid_step(),
        );
        return Ok(());
    }

    // init subsystems
    let sdl = sdl2::init().map_err(sdl_error)?;
    let sdl_video = sdl.video().map_err(sdl_error)?;
    let ttf = sdl2::ttf::init().map_err(sdl_error)?;

    // create window
    let window_title = if args.replay.is_some() {
        "Blockfish — BC-4 Replay Mode"
    } else {
        "Blockfish"
    };
    let mut canvas = sdl_video
        .window(window_title, view::DEFAULT_SIZE.0, view::DEFAULT_SIZE.1)
        .position_centered()
        .build()
        .map_err(sdl_error)?
        .into_canvas()
        .present_vsync()
        .build()
        .map_err(sdl_error)?;

    // load resources
    let resources = resources::Resources::load(&ttf)?;

    // more necessary init
    let mut events = sdl.event_pump().map_err(sdl_error)?;
    let texture_creator = canvas.texture_creator();

    // load config files
    let config_dir = args
        .config_dir
        .take()
        .unwrap_or_else(|| DEFAULT_CONFIG_DIR.into());
    let controls = controls::Controls::from_config_file(&config_dir)?;
    let theme = theme::Theme::from_config_file(&config_dir)?;

    // BC-4 Replay Mode: AI is NOT constructed and best_move() is never called.
    if let Some(replay_path) = args.replay.take() {
        let session = replay::ReplaySession::load_from_file(&replay_path, rules.clone())?;
        if let Some(ref err) = session.validation_error {
            log::error!("{}", err);
            eprintln!("Replay validation error: {}", err);
        }
        let view = view::View::new(resources, rules, controls, &theme);
        let mut player = replay::ReplayPlayer::new(
            view,
            session,
            args.start_step.unwrap_or(0),
            args.speed.unwrap_or(2.5),
            args.paused || args.screenshot.is_some(),
        );

        if let Some(shot_path) = args.screenshot.take() {
            player.view().render_labels(&texture_creator);
            player.view().paint(&mut canvas);
            canvas.present();
            save_canvas_bmp(&canvas, &shot_path)?;
            return Ok(());
        }

        loop {
            for evt in events.poll_iter() {
                if let Err(view::Quit) = player.handle_event(evt) {
                    return Ok(());
                }
            }
            player.tick();
            player.view().render_labels(&texture_creator);
            player.view().paint(&mut canvas);
            canvas.present();
            std::thread::sleep(player.idle_sleep());
        }
    }

    // Normal interactive mode: build ai, game state, view and controller
    let ai = blockfish::ai::AI::new(args.ai_config());
    let stacker = block_stacker::Stacker::new(rules.clone(), args.game_config());
    let view = view::View::new(resources, rules, controls, &theme);
    let mut ctl = controller::Controller::new(ai, view, stacker);
    let mut tmr = ctl.view().make_timer();

    loop {
        // poll user input events
        tmr.update();
        for evt in events.poll_iter() {
            match ctl.view().handle(evt, &mut tmr) {
                Err(view::Quit) => return Ok(()),
                Ok(Some(action)) => ctl.handle(action),
                Ok(None) => {}
            }
        }

        // poll time based events
        while let Some(action) = tmr.poll() {
            ctl.handle(action);
        }

        // poll for engine progress
        ctl.poll_engine();

        // render and present
        ctl.view().render_labels(&texture_creator);
        ctl.view().paint(&mut canvas);
        canvas.present();
    }
}

fn save_canvas_bmp(canvas: &view::Canvas, path: &std::path::Path) -> Result<()> {
    let (w, h) = canvas.output_size().map_err(sdl_error)?;
    let fmt = sdl2::pixels::PixelFormatEnum::RGB24;
    let mut pixels = canvas.read_pixels(None, fmt).map_err(sdl_error)?;
    let surf = sdl2::surface::Surface::from_data(&mut pixels, w, h, w * 3, fmt)
        .map_err(sdl_error)?;
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent).map_err(sdl_error)?;
        }
    }
    surf.save_bmp(path).map_err(sdl_error)?;
    Ok(())
}

trait FromConfigFile: (for<'de> serde::Deserialize<'de>) + Default {
    fn config_name() -> &'static str;

    fn from_config_file(dir: &std::path::Path) -> Result<Self> {
        let name = Self::config_name();
        let filename = format!("{}.json", name);
        if let Ok(f) = std::fs::File::open(dir.join(filename)) {
            serde_json::from_reader(f).map_err(|e| Error::ParseConfig(name, e))
        } else {
            log::warn!("{} config not found; using default", name);
            Ok(Self::default())
        }
    }
}

impl FromConfigFile for controls::Controls {
    fn config_name() -> &'static str {
        "controls"
    }
}

impl FromConfigFile for theme::Theme {
    fn config_name() -> &'static str {
        "theme"
    }
}
