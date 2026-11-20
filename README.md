# Blockfish

Blockfish is a bot dedicated to *Cheese Race*, a niche game mode where one must clear 100
lines of garbage with randomly placed gaps ("cheese"). The goal of cheese race is to clear
all 100 lines using as few tetromino's as possible. Blockfish is meant to be a practice
tool for evaluating placements during a cheese race, spotting mistakes, and helping the
user get better at recognizing more efficient ways of utilizing every piece.

Currently, Blockfish tends to outperform even highly skilled players, and has already
[beaten](https://jstris.jezevec10.com/replay/53445238) the [world
record](https://jstris.jezevec10.com/replay/54532835) for least-pieces 100L cheese
race. However, there are still improvements to be made to the underlying algorithms. We
hope to see Blockfish reaching before-unheard piece counts as low as 160 or 150.

![screenshot of the Blockfish client](https://raw.githubusercontent.com/iitalics/blockfish/dev/support/readme-screenshot.png)

Blockfish is under development by `iitalics`, with lots of help from cheese race legend
`mystery` in designing and fine-tuning the algorithms that make it perform so well.

## Getting Blockfish

### Quality research profile

For the independently validated 50k-node strategy, select `cheese-robust`:

```sh
cargo run --release --locked -p blockfish --features race,cheese-robust \
  --bin blockfish-race -- 100 -q -s 3 -A 50/5,10,10,10 -g 9 -G 4
```

This keeps the learned evaluator and averages the best four distinct terminal
states for each first move. On 64 previously sealed paired seeds, it reduced
mean pieces from 199.15625 to 196.921875 (paired 95% bootstrap interval for the
difference [-4.4375, -0.125]), with 64/64 completions in both groups. The interval
is close to zero; the evidence is specific to this simulator and 50k budget.
At 5k, a separate diagnostic did not support improvement. The prior
`cheese-quality` profile and frozen benchmark binary remain unchanged.

An independent 64-seed replication at 50k also favored the robust strategy:
195.000 versus 198.40625 pieces, with paired 95% interval [-5.671875, -1.140625]
and full completion. This used an observer-only control whose decisions matched
the frozen robust implementation on all 32 development seeds. Enabling exact
seven-bag inference did **not** improve on that control (195.046875 pieces), so
it remains experimental. See the [bag inference study](research/BAG_REPORT.md)
for the locked cohorts, public-information audit, and negative result.
The [Hold-context study](research/CONTEXT_REPORT.md) also failed to improve on
`cheese-robust` in a fresh 64-seed final (196.546875 versus 196.21875 pieces;
paired interval [-1.515625, 2.15625]). Its learned correction and exact cache
remain optional research features, not recommended profiles.
The [root allocation study](research/ROOT_SEARCH_REPORT.md) and
[public-world rollout study](research/PUBLIC_ROLLOUT_REPORT.md) likewise failed
to improve full 100L piece counts. Their optional prototypes and reproducible
negative results are retained; `cheese-robust` remains the recommended profile.
The [S27 synthesis and transposition experiment](research/SYNTHESIS_S27.md)
also found no piece-count improvement. The bounded
[S28 public-label diagnostic](research/PUBLIC_LABEL_REPORT.md) tests continuation
policy and target reliability without training or promoting another model.
The [S29/S30 global-search study](research/GLOBAL_SEARCH_REPORT.md) tested two
fixed frontier policies on 32 new seeds each. Neither passed: relative-cost
search used more pieces, and absolute-f search failed to return a move in four
games. These are experimental profiles; the recommended strategy is unchanged.


This checkout includes a reproducible 100L research harness and an offline-learned
board evaluator. Use the `cheese-quality` feature to select the learned evaluator
with a cumulative generated-node budget:

```sh
cargo run --release --locked -p blockfish --features race,cheese-quality \
  --bin blockfish-race -- 100 -q -s 3 -A 5/5,10,10,10 \
  --trace-file cheese-trace.json
```

The recorded seed-3 run uses 178 pieces in the local simulator. Replay an operation
trace without searching using `blockfish-race --replay-file cheese-trace.json`.
For a larger search budget, use `-A 50/5,10,10,10`. The quality profile uses fixed
learned board weights; the first three `-A` weights apply to the legacy/blended
evaluators, while the fourth remains the path cost per piece.

The ordinary default retains the legacy evaluator with exact speed optimizations.
See the [quality research report](research/QUALITY_REPORT.md) for independent
piece-count comparisons, limitations, training splits, and reproducible commands.
The [budget follow-up](research/BUDGET_REPORT.md) compares 50k and 200k generated
nodes on 32 new paired seeds: the learned evaluator retains its advantage at
200k, but additional search has not shown a reliable further piece-count gain.
The [generalization study](research/GENERALIZATION_REPORT.md) diagnoses failed
weight optimization and tests alternative search backups against the frozen model.
Local simulator results are not official Jstris records; the record links in the
original introduction above are historical upstream claims, not a verification
of the current leaderboard.

Blockfish is still in early development, but latest builds of the Blockfish GUI can be
downloaded from the public [GitHub repository](https://github.com/iitalics/blockfish/releases).

## Controls and theme configuration

The Blockfish client looks in `./config/controls.json` for your personal controls
configuration, and `./config/themes.json` for the theme. The default controls file can be
found in `support/default-controls.json`; edit this file to specify your own controls.

There are example themes in `support/themes` that you may use to change the colors in the
client. See `support/themes/README.md` for more information.

## Compiling Blockfish

Blockfish is written primarily in Rust. You can build it with the `cargo` tool.

* Requires: `rust` version 1.48
* Requires SDL2 and SDL2_ttf dev libraries.

  On MacOS, the dev libraries can be installed with [homebrew](https://brew.sh/).
  Once homebrew has been installed, the following commands are used to installed the libraries
  ```sh
  brew install sdl2
  brew install sld2_ttf
  ```

The following command should be used to build the client:

```sh
cargo build --manifest-path blockfish-client/Cargo.toml --release --features msgbox
```

Note that on Windows you must have the `.LIB` files for SDL2 and SDL2_ttf placed in the
same directory that you run this command in. Running Blockfish requires the associated
`.DLL` files for these depencies.
