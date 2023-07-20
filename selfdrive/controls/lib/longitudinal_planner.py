#!/usr/bin/env python3
import math
import numpy as np
from common.numpy_fast import clip, interp
from common.params import Params
from cereal import log

import cereal.messaging as messaging
from common.conversions import Conversions as CV
from common.filter_simple import FirstOrderFilter
from common.realtime import DT_MDL
from selfdrive.hybrid_modeld.constants import T_IDXS
from selfdrive.controls.lib.longcontrol import LongCtrlState
from selfdrive.controls.lib.longitudinal_mpc_lib.long_mpc import LongitudinalMpc, MIN_ACCEL, MAX_ACCEL, STOP_DISTANCE
from selfdrive.controls.lib.longitudinal_mpc_lib.long_mpc import T_IDXS as T_IDXS_MPC
from selfdrive.controls.lib.drive_helpers import V_CRUISE_MAX, CONTROL_N, get_speed_error
from system.swaglog import cloudlog
from selfdrive.controls.lib.vision_turn_controller import VisionTurnController

LON_MPC_STEP = 0.2  # first step is 0.2s
A_CRUISE_MIN = -1.2
A_CRUISE_MAX_VALS = [1.6, 1.2, 0.8, 0.6]
A_CRUISE_MAX_BP = [0., 10.0, 25., 40.]

# Lookup table for turns
_A_TOTAL_MAX_V = [1.7, 3.2]
_A_TOTAL_MAX_BP = [20., 40.]

# d-e2e, from legacy_modeldata.h
TRAJECTORY_SIZE = 33

_DP_E2E_LEAD_COUNT = 5

_DP_E2E_STOP_BP = [0., 10., 20., 30., 40., 50., 55.]
_DP_E2E_STOP_DIST = [10, 30., 50., 70., 80., 90., 120.]
_DP_E2E_STOP_COUNT = 3

_DP_E2E_SNG_COUNT = 3
_DP_E2E_SNG_ACC_COUNT = 5
_DP_E2E_SWAP_COUNT = 10

_DP_E2E_TF_COUNT = 5

def get_max_accel(v_ego):
  return interp(v_ego, A_CRUISE_MAX_BP, A_CRUISE_MAX_VALS)


def limit_accel_in_turns(v_ego, angle_steers, a_target, CP):
  """
  This function returns a limited long acceleration allowed, depending on the existing lateral acceleration
  this should avoid accelerating when losing the target in turns
  """

  # FIXME: This function to calculate lateral accel is incorrect and should use the VehicleModel
  # The lookup table for turns should also be updated if we do this
  a_total_max = interp(v_ego, _A_TOTAL_MAX_BP, _A_TOTAL_MAX_V)
  a_y = v_ego ** 2 * angle_steers * CV.DEG_TO_RAD / (CP.steerRatio * CP.wheelbase)
  a_x_allowed = math.sqrt(max(a_total_max ** 2 - a_y ** 2, 0.))

  return [a_target[0], min(a_target[1], a_x_allowed)]


class LongitudinalPlanner:
  def __init__(self, CP, init_v=0.0, init_a=0.0):
    self.vision_turn_controller = VisionTurnController(CP)
    # conditional e2e
    self.dp_e2e_has_lead = False
    self.dp_e2e_lead_last = False
    self.dp_e2e_lead_count = 0
    self.dp_e2e_sng = False
    self.dp_e2e_sng_count = 0
    self.dp_e2e_standstill_last = False
    self.dp_e2e_swap_count = 0
    self.dp_e2e_stop_count = 0
    self.dp_e2e_tf_count = 0

    self.CP = CP
    self.mpc = LongitudinalMpc()
    self.fcw = False

    self.a_desired = init_a
    self.v_desired_filter = FirstOrderFilter(init_v, 2.0, DT_MDL)
    self.v_model_error = 0.0

    self.v_desired_trajectory = np.zeros(CONTROL_N)
    self.a_desired_trajectory = np.zeros(CONTROL_N)
    self.j_desired_trajectory = np.zeros(CONTROL_N)
    self.solverExecutionTime = 0.0
    self.params = Params()
    self.param_read_counter = 0
    self.read_param()
    self.personality = log.LongitudinalPersonality.standard

  def read_param(self):
    param_value = self.params.get('LongitudinalPersonality')
    if param_value is not None:
      self.personality = int(param_value)
    else:
      self.personality = log.LongitudinalPersonality.standard

  def _set_dp_e2e_mode(self, mode, force=False):
    reset_state = False

    if force:
      self.dp_e2e_swap_count = 0
      if self.mpc.mode != mode:
        reset_state = True
      self.mpc.mode = mode
      return reset_state

    # prevent switching in a short period of time.
    if self.mpc.mode == mode:
      self.dp_e2e_swap_count = 0
    else:
      self.dp_e2e_swap_count += 1

    if self.dp_e2e_swap_count >= _DP_E2E_SWAP_COUNT:
      self.mpc.mode = mode
      reset_state = True

    return reset_state

  def conditional_e2e(self, sm):
    if not sm['controlsState'].experimentalMode:
      return self._set_dp_e2e_mode('acc', True)
    v_ego_kph = sm['carState'].vEgo * 3.6
    standstill = sm['carState'].standstill

    # lead detection with buffer
    lead = sm['radarState'].leadOne
    lead_dist = lead.dRel

    # make sure it see lead enough time
    if lead.status != self.dp_e2e_lead_last:
      self.dp_e2e_lead_count = 0
    else:
      self.dp_e2e_lead_count += 1
      if self.dp_e2e_lead_count >= _DP_E2E_LEAD_COUNT:
        self.dp_e2e_has_lead = lead.status
    self.dp_e2e_lead_last = lead.status

    # when standstill, always e2e
    if standstill:
      self.dp_e2e_sng_count = 0
      self.dp_e2e_sng = False
      return self._set_dp_e2e_mode('blended')

    if self.dp_e2e_standstill_last and not standstill:
      self.dp_e2e_sng = True

    # when sng, we e2e for 0.5 secs
    if self.dp_e2e_sng:
      self.dp_e2e_sng_count += 1
      if self.dp_e2e_sng_count > _DP_E2E_SNG_COUNT:
        if self.dp_e2e_sng_count > _DP_E2E_SNG_ACC_COUNT:
          self.dp_e2e_sng = False
        return self._set_dp_e2e_mode('acc', True)
      return self._set_dp_e2e_mode('blended')

    # when we see a lead
    # if sm['dragonConf'].dpE2EConditionalVoacc and self.dp_e2e_has_lead:
    if self.CP.radarUnavailable and self.dp_e2e_has_lead:
      # drive above conditional speed and lead is too close
      if lead_dist <= v_ego_kph * self.mpc.t_follow * interp(v_ego_kph, [50., 60., 80., 85, 90.], [1.25, 1.20, 1.10, 1.05, 1.]) / 3.6:
        self.dp_e2e_tf_count += 1
      else:
        self.dp_e2e_tf_count = 0
      if self.dp_e2e_tf_count > _DP_E2E_TF_COUNT:
        return self._set_dp_e2e_mode('blended', True)

    # stop sign detection
    md = sm['modelV2']
    if abs(sm['carState'].steeringAngleDeg) <= 60 and len(md.orientation.x) == len(md.position.x) == TRAJECTORY_SIZE:
      if md.position.x[TRAJECTORY_SIZE - 1] < interp(v_ego_kph, _DP_E2E_STOP_BP, _DP_E2E_STOP_DIST):
        #self.dp_e2e_stop_count += 1
        self.dp_e2e_stop_count = 0 # disable stop sign
      else:
        self.dp_e2e_stop_count = 0
    else:
      self.dp_e2e_stop_count = 0

    if self.dp_e2e_stop_count >= _DP_E2E_STOP_COUNT:
      return self._set_dp_e2e_mode('blended', True)

    return self._set_dp_e2e_mode('acc')

  @staticmethod
  def parse_model(model_msg, model_error):
    if (len(model_msg.position.x) == 33 and
       len(model_msg.velocity.x) == 33 and
       len(model_msg.acceleration.x) == 33):
      x = np.interp(T_IDXS_MPC, T_IDXS, model_msg.position.x) - model_error * T_IDXS_MPC
      v = np.interp(T_IDXS_MPC, T_IDXS, model_msg.velocity.x) - model_error
      a = np.interp(T_IDXS_MPC, T_IDXS, model_msg.acceleration.x)
      j = np.zeros(len(T_IDXS_MPC))
    else:
      x = np.zeros(len(T_IDXS_MPC))
      v = np.zeros(len(T_IDXS_MPC))
      a = np.zeros(len(T_IDXS_MPC))
      j = np.zeros(len(T_IDXS_MPC))
    return x, v, a, j

  def update(self, sm):
    if self.param_read_counter % 50 == 0:
      self.read_param()
    self.param_read_counter += 1
    # self.mpc.mode = 'blended' if sm['controlsState'].experimentalMode else 'acc'
    dp_reset_state = self.conditional_e2e(sm)

    v_ego = sm['carState'].vEgo
    v_cruise_kph = sm['controlsState'].vCruise
    v_cruise_kph = min(v_cruise_kph, V_CRUISE_MAX)
    v_cruise = v_cruise_kph * CV.KPH_TO_MS

    long_control_off = sm['controlsState'].longControlState == LongCtrlState.off
    force_slow_decel = sm['controlsState'].forceDecel

    # Reset current state when not engaged, or user is controlling the speed
    reset_state = long_control_off if self.CP.openpilotLongitudinalControl else not sm['controlsState'].enabled

    # No change cost when user is controlling the speed, or when standstill
    prev_accel_constraint = not (reset_state or sm['carState'].standstill)

    if self.mpc.mode == 'acc':
      accel_limits = [A_CRUISE_MIN, get_max_accel(v_ego)]
      accel_limits_turns = limit_accel_in_turns(v_ego, sm['carState'].steeringAngleDeg, accel_limits, self.CP)
    else:
      accel_limits = [MIN_ACCEL, MAX_ACCEL]
      accel_limits_turns = [MIN_ACCEL, MAX_ACCEL]

    if reset_state or dp_reset_state:
      self.v_desired_filter.x = v_ego
      # Clip aEgo to cruise limits to prevent large accelerations when becoming active
      self.a_desired = clip(sm['carState'].aEgo, accel_limits[0], accel_limits[1])

    # Prevent divergence, smooth in current v_ego
    self.v_desired_filter.x = max(0.0, self.v_desired_filter.update(v_ego))
    # Compute model v_ego error
    self.v_model_error = get_speed_error(sm['modelV2'], v_ego)

    # rick - vision turn controller from move-fast team
    # https://github.com/move-fast/openpilot/blob/develop/selfdrive/controls/lib/vision_turn_controller.py
    #self.vision_turn_controller.update(not reset_state, self.v_desired_filter.x, self.a_desired, v_cruise, sm)
    #if self.vision_turn_controller.is_active:
      #if min(v_cruise, self.vision_turn_controller.v_turn) == self.vision_turn_controller.v_turn:
        #self.a_desired = self.vision_turn_controller.a_target
        #v_cruise = self.vision_turn_controller.v_turn

    if force_slow_decel:
      v_cruise = 0.0
    # clip limits, cannot init MPC outside of bounds
    accel_limits_turns[0] = min(accel_limits_turns[0], self.a_desired + 0.05)
    accel_limits_turns[1] = max(accel_limits_turns[1], self.a_desired - 0.05)

    self.mpc.set_weights(prev_accel_constraint, personality=self.personality)
    self.mpc.set_accel_limits(accel_limits_turns[0], accel_limits_turns[1])
    self.mpc.set_cur_state(self.v_desired_filter.x, self.a_desired)
    x, v, a, j = self.parse_model(sm['modelV2'], self.v_model_error)
    # dynamic stopping distance ONLY when on radarUnavailable vehicles (e.g. Toyota C-HR, VW)
    stop_distance = STOP_DISTANCE if not self.CP.radarUnavailable else interp(sm['carState'].vEgo, [0., 2.78, 5.55, 22.], [3.7, 4., 5, STOP_DISTANCE])
    self.mpc.update(sm['radarState'], v_cruise, x, v, a, j, personality=self.personality, stop_distance=stop_distance)

    self.v_desired_trajectory_full = np.interp(T_IDXS, T_IDXS_MPC, self.mpc.v_solution)
    self.a_desired_trajectory_full = np.interp(T_IDXS, T_IDXS_MPC, self.mpc.a_solution)
    self.v_desired_trajectory = self.v_desired_trajectory_full[:CONTROL_N]
    self.a_desired_trajectory = self.a_desired_trajectory_full[:CONTROL_N]
    self.j_desired_trajectory = np.interp(T_IDXS[:CONTROL_N], T_IDXS_MPC[:-1], self.mpc.j_solution)

    # TODO counter is only needed because radar is glitchy, remove once radar is gone
    self.fcw = self.mpc.crash_cnt > 2 and not sm['carState'].standstill
    if self.fcw:
      cloudlog.info("FCW triggered")
      self._set_dp_e2e_mode('blended', True)

    # Interpolate 0.05 seconds and save as starting point for next iteration
    a_prev = self.a_desired
    self.a_desired = float(interp(DT_MDL, T_IDXS[:CONTROL_N], self.a_desired_trajectory))
    self.v_desired_filter.x = self.v_desired_filter.x + DT_MDL * (self.a_desired + a_prev) / 2.0

  def publish(self, sm, pm):
    plan_send = messaging.new_message('longitudinalPlan')

    plan_send.valid = sm.all_checks(service_list=['carState', 'controlsState'])

    longitudinalPlan = plan_send.longitudinalPlan
    longitudinalPlan.modelMonoTime = sm.logMonoTime['modelV2']
    longitudinalPlan.processingDelay = (plan_send.logMonoTime / 1e9) - sm.logMonoTime['modelV2']

    longitudinalPlan.speeds = self.v_desired_trajectory.tolist()
    longitudinalPlan.accels = self.a_desired_trajectory.tolist()
    longitudinalPlan.jerks = self.j_desired_trajectory.tolist()

    longitudinalPlan.hasLead = sm['radarState'].leadOne.status
    longitudinalPlan.longitudinalPlanSource = self.mpc.source
    longitudinalPlan.fcw = self.fcw
    longitudinalPlan.longitudinalValid = self.mpc.mode == 'acc'

    longitudinalPlan.solverExecutionTime = self.mpc.solve_time
    longitudinalPlan.personality = self.personality

    pm.send('longitudinalPlan', plan_send)
