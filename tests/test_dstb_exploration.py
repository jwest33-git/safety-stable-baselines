"""Disturbance-exploration options of the two-player SAC learners:
``dstb_min_alpha`` (the dstb temperature's own floor), ``dstb_alpha_reset``
(whether a gamma jump resets it), ``league_deterministic`` (archived league
dstb played at its deterministic action), ``dstb_target_entropy`` (the dstb
temperature's tuning target) and the ``train/dstb_sample_cos`` metric, plus
the matching ctrl options (``ctrl_min_alpha``, ``ctrl_alpha_reset``,
``ctrl_target_entropy``, ``train/ctrl_sample_cos``). CPU
only, on the two-player double integrator.

  python tests/test_dstb_exploration.py
"""

from __future__ import annotations

import copy
import math
import os
import sys

import torch as th

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from safety_sb3 import SafetySAC2P  # noqa: E402
from test_tensor_sac import DEV, TwoPlayerDoubleIntegratorEnv  # noqa: E402


def _model(**kw):
  """A briefly trained SafetySAC2P (real weights, a filled buffer, a logger)."""
  env = TwoPlayerDoubleIntegratorEnv()
  model = SafetySAC2P(
    "MlpPolicy", env, ctrl_action_dim=1, buffer_size=20_000, batch_size=256,
    learning_starts=0, train_freq=1, gradient_steps=1, gamma=0.95,
    ent_coef="auto_0.1", min_alpha=1e-3, gamma_anneal=False,
    policy_kwargs=dict(net_arch=[32, 32]), seed=0, device=DEV, verbose=0, **kw)
  model.learn(total_timesteps=1_000, log_interval=None)
  return model


def _alphas_after_push_down(model):
  """Set both temperatures far below every floor, take one train step, and
  return (ctrl alpha, dstb alpha) after the clamp."""
  with th.no_grad():
    model.log_ent_coef.fill_(math.log(1e-8))
    model.dstb_log_ent_coef.fill_(math.log(1e-8))
  model.train(gradient_steps=1, batch_size=256)
  ctrl = float(model.log_ent_coef.detach().exp())
  dstb = float(model.dstb_log_ent_coef.detach().exp())
  return ctrl, dstb


def test_dstb_min_alpha():
  # None: the dstb temperature shares the ctrl floor
  ctrl, dstb = _alphas_after_push_down(_model())
  assert ctrl >= 1e-3 * 0.999 and dstb >= 1e-3 * 0.999, (ctrl, dstb)
  # its own, lower floor: the dstb clamps there, the ctrl keeps 1e-3
  ctrl, dstb = _alphas_after_push_down(_model(dstb_min_alpha=1e-6))
  assert ctrl >= 1e-3 * 0.999, ctrl
  assert 1e-6 * 0.999 <= dstb < 1e-4, dstb
  # <= 0: no dstb floor at all
  ctrl, dstb = _alphas_after_push_down(_model(dstb_min_alpha=0.0))
  assert ctrl >= 1e-3 * 0.999, ctrl
  assert dstb < 1e-6, dstb
  print("[ok] dstb_min_alpha: shared / own / no floor")


def test_dstb_alpha_reset():
  for reset in (True, False):
    model = _model(dstb_alpha_reset=reset)
    init_ctrl = float(model._init_log_ent_coef.exp())
    init_dstb = float(model._init_dstb_log_ent_coef.exp())
    with th.no_grad():
      model.log_ent_coef.fill_(math.log(0.005))
      model.dstb_log_ent_coef.fill_(math.log(0.005))
    model._reset_entropy_temp()
    ctrl = float(model.log_ent_coef.detach().exp())
    dstb = float(model.dstb_log_ent_coef.detach().exp())
    # the ctrl temperature is always reset
    assert abs(ctrl - init_ctrl) < 1e-6, (ctrl, init_ctrl)
    if reset:
      assert abs(dstb - init_dstb) < 1e-6, (dstb, init_dstb)
    else:
      assert abs(dstb - 0.005) < 1e-6, dstb
  print("[ok] dstb_alpha_reset: reset / kept across a gamma jump")


def _dstb_actions_twice(model, obs):
  first = model._tensor_policy_actions(obs)[:, 1]
  second = model._tensor_policy_actions(obs)[:, 1]
  return first, second


def test_league_deterministic():
  obs = th.rand(64, 2, device=DEV) * 1.6 - 0.8
  for det in (True, False):
    model = _model(league_deterministic=det)
    # the current dstb is always sampled
    model._rollout_dstb = None
    a, b = _dstb_actions_twice(model, obs)
    assert not th.allclose(a, b), "the current dstb must stay stochastic"
    # an archived league dstb: deterministic only when asked
    archived = copy.deepcopy(model.policy.dstb_actor)
    archived.set_training_mode(False)
    model._rollout_dstb = archived
    a, b = _dstb_actions_twice(model, obs)
    if det:
      assert th.allclose(a, b), "archived dstb must play its deterministic action"
      with th.no_grad():
        mean = archived(obs, deterministic=True)[:, 0]
      assert th.allclose(a, mean)
    else:
      assert not th.allclose(a, b), "archived dstb is sampled by default"
  print("[ok] league_deterministic: archived dstb deterministic, current sampled")


def test_dstb_target_entropy():
  # default: -dstb_dim (1 here)
  assert _model().dstb_target_entropy == -1.0
  # a target above the policy's entropy raises the temperature, one below it
  # lowers it (same weights and batch, one train step each)
  moved = {}
  for target in (20.0, -20.0):
    model = _model(dstb_target_entropy=target, dstb_min_alpha=0.0)
    assert model.dstb_target_entropy == target
    before = float(model.dstb_log_ent_coef.detach())
    th.manual_seed(0)
    model.train(gradient_steps=1, batch_size=256)
    moved[target] = float(model.dstb_log_ent_coef.detach()) - before
  assert moved[20.0] > 0 > moved[-20.0], moved
  print(f"[ok] dstb_target_entropy: alpha moves toward the target {moved}")


def test_dstb_sample_cos_logged():
  model = _model()
  model.train(gradient_steps=2, batch_size=256)
  cos = model.logger.name_to_value["train/dstb_sample_cos"]
  assert -1.0 <= cos <= 1.0, cos
  print(f"[ok] train/dstb_sample_cos logged: {cos:+.3f}")


def test_ctrl_min_alpha():
  # its own, lower floor: the ctrl clamps there, the dstb keeps 1e-3
  ctrl, dstb = _alphas_after_push_down(_model(ctrl_min_alpha=1e-6))
  assert 1e-6 * 0.999 <= ctrl < 1e-4, ctrl
  assert dstb >= 1e-3 * 0.999, dstb
  # <= 0: no ctrl floor at all
  ctrl, dstb = _alphas_after_push_down(_model(ctrl_min_alpha=0.0))
  assert ctrl < 1e-6, ctrl
  assert dstb >= 1e-3 * 0.999, dstb
  print("[ok] ctrl_min_alpha: own / no floor")


def test_ctrl_alpha_reset():
  for reset in (True, False):
    model = _model(ctrl_alpha_reset=reset)
    init_ctrl = float(model._init_log_ent_coef.exp())
    init_dstb = float(model._init_dstb_log_ent_coef.exp())
    with th.no_grad():
      model.log_ent_coef.fill_(math.log(0.005))
      model.dstb_log_ent_coef.fill_(math.log(0.005))
    model._reset_entropy_temp()
    ctrl = float(model.log_ent_coef.detach().exp())
    dstb = float(model.dstb_log_ent_coef.detach().exp())
    # the dstb temperature is reset by default
    assert abs(dstb - init_dstb) < 1e-6, (dstb, init_dstb)
    if reset:
      assert abs(ctrl - init_ctrl) < 1e-6, (ctrl, init_ctrl)
    else:
      assert abs(ctrl - 0.005) < 1e-6, ctrl
  print("[ok] ctrl_alpha_reset: reset / kept across a gamma jump")


def test_ctrl_target_entropy():
  # default: -ctrl_action_dim (1 here)
  assert _model().target_entropy == -1.0
  moved = {}
  for target in (20.0, -20.0):
    model = _model(ctrl_target_entropy=target, ctrl_min_alpha=0.0)
    assert model.target_entropy == target
    before = float(model.log_ent_coef.detach())
    th.manual_seed(0)
    model.train(gradient_steps=1, batch_size=256)
    moved[target] = float(model.log_ent_coef.detach()) - before
  assert moved[20.0] > 0 > moved[-20.0], moved
  print(f"[ok] ctrl_target_entropy: alpha moves toward the target {moved}")


def test_ctrl_sample_cos_logged():
  model = _model()
  model.train(gradient_steps=2, batch_size=256)
  cos = model.logger.name_to_value["train/ctrl_sample_cos"]
  assert -1.0 <= cos <= 1.0, cos
  print(f"[ok] train/ctrl_sample_cos logged: {cos:+.3f}")


if __name__ == "__main__":
  th.manual_seed(0)
  test_dstb_min_alpha()
  test_dstb_alpha_reset()
  test_league_deterministic()
  test_dstb_target_entropy()
  test_dstb_sample_cos_logged()
  test_ctrl_min_alpha()
  test_ctrl_alpha_reset()
  test_ctrl_target_entropy()
  test_ctrl_sample_cos_logged()
  print("ALL DSTB-EXPLORATION TESTS PASSED")
