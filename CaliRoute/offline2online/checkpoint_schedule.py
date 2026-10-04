"""Independent archive and rolling recovery checkpoint schedules."""

from dataclasses import dataclass


@dataclass(frozen=True)
class EpochCheckpointPlan:
    evaluate: bool
    archive: bool
    latest: bool

    @property
    def capture_resume_state(self) -> bool:
        return self.evaluate or self.archive or self.latest


def epoch_checkpoint_plan(
    epoch: int,
    epochs: int,
    *,
    eval_interval: int,
    checkpoint_interval: int,
    latest_checkpoint_interval: int = 0,
) -> EpochCheckpointPlan:
    """Plan work after a completed training epoch; zero disables rolling saves.

    Keep the existing unconditional final archive even when periodic archives
    are disabled. Rolling saves also include the final epoch when enabled.
    """
    if not 1 <= epoch <= epochs:
        raise ValueError("checkpoint planning requires a completed epoch in [1, epochs]")
    final = epoch == epochs
    return EpochCheckpointPlan(
        evaluate=eval_interval > 0 and (epoch % eval_interval == 0 or final),
        archive=final or (checkpoint_interval > 0 and epoch % checkpoint_interval == 0),
        latest=latest_checkpoint_interval > 0 and (epoch % latest_checkpoint_interval == 0 or final),
    )
