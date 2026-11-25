"""Generate deterministic CSV fixtures for the MATLAB lambda CV test.

The reference implements the one-step-ahead cross-validation criterion:
  - one-step-ahead expanding windows,
  - projection loss (1 - R_t(h_lambda))^2,
  - mean loss across validation dates,
  - lambda search on [1e-5, 1e-2].

Losses are evaluated on an explicit log grid, as in the MATLAB implementation;
candidate-wise losses are identical. When optuna is installed, a continuous
Optuna/TPE search on the same interval is also recorded in the summary.
"""

from pathlib import Path
import json

import numpy as np

try:
    import optuna
    from optuna.samplers import TPESampler
except ImportError:
    optuna = None
    TPESampler = None


SEED = 84
N_BASIS = 5
T = 24
MIN_TRAIN = 8
LAMBDA_GRID = np.logspace(-5, -2, 31)


def make_fixture():
    rng = np.random.default_rng(SEED)

    A = rng.normal(size=(N_BASIS, N_BASIS))
    K_h = A @ A.T + 0.5 * np.eye(N_BASIS)

    direction = rng.normal(size=N_BASIS)
    direction /= np.linalg.norm(direction)
    alpha = (
        np.outer(direction, 0.8 + 0.2 * rng.normal(size=T))
        + 0.7 * rng.normal(size=(N_BASIS, T))
    )

    return alpha, K_h


def one_step_loss(alpha, K_h, lmbda, min_train):
    losses = []

    for t in range(min_train, alpha.shape[1]):
        alpha_train = alpha[:, :t]
        alpha_bar = alpha_train.mean(axis=1)
        Q_alpha = alpha_train @ alpha_train.T / t
        Q_alpha = 0.5 * (Q_alpha + Q_alpha.T)

        system = Q_alpha + lmbda * K_h
        system = 0.5 * (system + system.T)
        h_lambda = np.linalg.solve(system, alpha_bar)

        replicating_portfolio = alpha[:, t] @ h_lambda
        losses.append((1.0 - replicating_portfolio) ** 2)

    return float(np.mean(losses))


def grid_search(alpha, K_h):
    mean_loss = np.array([
        one_step_loss(alpha, K_h, lmbda, MIN_TRAIN)
        for lmbda in LAMBDA_GRID
    ])
    best_idx = int(np.argmin(mean_loss))
    return mean_loss, float(LAMBDA_GRID[best_idx])


def optuna_search(alpha, K_h):
    if optuna is None:
        return None

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="maximize",
        sampler=TPESampler(seed=42),
    )

    def objective(trial):
        lmbda = trial.suggest_float("lmbda", 1e-5, 1e-2, log=True)
        return -one_step_loss(alpha, K_h, lmbda, MIN_TRAIN)

    study.optimize(objective, n_trials=100)
    return {
        "lambda": float(study.best_params["lmbda"]),
        "mean_loss": float(-study.best_value),
    }


def main():
    out_dir = Path(__file__).resolve().parents[1] / "fixtures"
    out_dir.mkdir(parents=True, exist_ok=True)

    alpha, K_h = make_fixture()
    mean_loss, selected_lambda = grid_search(alpha, K_h)
    optuna_result = optuna_search(alpha, K_h)

    np.savetxt(out_dir / "python_cv_alpha.csv", alpha, delimiter=",", fmt="%.17g")
    np.savetxt(out_dir / "python_cv_K_h.csv", K_h, delimiter=",", fmt="%.17g")
    np.savetxt(
        out_dir / "python_cv_results.csv",
        np.column_stack([LAMBDA_GRID, mean_loss]),
        delimiter=",",
        header="lambda,mean_loss",
        comments="",
        fmt="%.17g",
    )

    summary = {
        "seed": SEED,
        "n_basis": N_BASIS,
        "T": T,
        "min_train_dates": MIN_TRAIN,
        "grid_selected_lambda": selected_lambda,
        "grid_selected_mean_loss": float(mean_loss.min()),
        "optuna_100_trial": optuna_result,
    }
    (out_dir / "python_cv_reference_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
