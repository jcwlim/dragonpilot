from common.numpy_fast import interp

# d-e2e, from modeldata.h
TRAJECTORY_SIZE = 33

LEAD_WINDOW_SIZE = 6 #5
LEAD_PROB = 0.6 #0.6

SLOW_DOWN_WINDOW_SIZE = 5
SLOW_DOWN_PROB = 0.6
SLOW_DOWN_BP = [0., 10., 20., 30., 40., 50., 55.] # not ori [0., 10., 20., 40., 50., 60., 70.] #
SLOW_DOWN_DIST = [10, 30., 50., 70., 80., 90., 120.] # not ori [10, 20., 40., 60., 70., 80., 120.] #

SLOWNESS_WINDOW_SIZE = 20 #20
SLOWNESS_PROB = 0.6 #0.6
SLOWNESS_CRUISE_OFFSET = 1.0 #1.05

DANGEROUS_TTC_WINDOW_SIZE = 5
DANGEROUS_TTC = 2.15 #1.55

HIGHWAY_CRUISE_KPH = 70 #75

STOP_AND_GO_FRAME = 300 #500

MODE_SWITCH_DELAY_FRAME = 100 #500


class SNG_State:
  off = 0
  stopped = 1
  going = 2

class GenericMovingAverageCalculator:
  def __init__(self, window_size):
    self.window_size = window_size
    self.data = []
    self.total = 0

  def add_data(self, value):
    if len(self.data) == self.window_size:
      self.total -= self.data.pop(0)
    self.data.append(value)
    self.total += value

  def get_moving_average(self):
    if len(self.data) == 0:
      return None
    return self.total / len(self.data)

  def reset_data(self):
    self.data = []
    self.total = 0


class DynamicEndtoEndController:

  def __init__(self):
    self._is_enabled = False
    self._mode = 'acc'
    self._mode_prev = 'acc'
    self._mode_switch_allowed = True
    self._mode_switch_frame = 0
    self._frame = 0

    self._lead_gmac = GenericMovingAverageCalculator(window_size=LEAD_WINDOW_SIZE)
    self._has_lead_filtered = False
    self._has_lead_filtered_prev = False

    self._slow_down_gmac = GenericMovingAverageCalculator(window_size=SLOW_DOWN_WINDOW_SIZE)
    self._has_slow_down = False
    self._has_slow_down_prev = False

    self._has_blinkers = False
    self._has_blinkers_prev = False

    self._slowness_gmac = GenericMovingAverageCalculator(window_size=SLOWNESS_WINDOW_SIZE)
    self._has_slowness = False
    self._has_slowness_prev = False

    self._has_nav_enabled = False
    self._has_nav_enabled_prev = False

    self._dangerous_ttc_gmac = GenericMovingAverageCalculator(window_size=DANGEROUS_TTC_WINDOW_SIZE)
    self._has_dangerous_ttc = False
    self._has_dangerous_ttc_prev = False

    self._v_ego_kph = 0.
    self._v_cruise_kph = 0.

    self._has_lead = False
    self._has_lead_prev = False

    self._has_standstill = False
    self._has_standstill_prev = False

    self._sng_transit_frame = 0
    self._sng_state = SNG_State.off
    pass

  def _update(self, car_state, lead_one, md, controls_state, radar_unavailable):
    self._v_ego_kph = car_state.vEgo * 3.6
    self._v_cruise_kph = controls_state.vCruise
    self._has_lead = lead_one.status
    self._has_standstill = car_state.standstill

    # nav enable detection
    self._has_nav_enabled = md.navEnabled

    # lead detection
    self._lead_gmac.add_data(lead_one.status)
    self._has_lead_filtered = self._lead_gmac.get_moving_average() >= LEAD_PROB

    # slow down detection
    self._slow_down_gmac.add_data(len(md.orientation.x) == len(md.position.x) == TRAJECTORY_SIZE and md.position.x[TRAJECTORY_SIZE - 1] < interp(self._v_ego_kph, SLOW_DOWN_BP, SLOW_DOWN_DIST))
    self._has_slow_down = self._slow_down_gmac.get_moving_average() >= SLOW_DOWN_PROB

    # blinker detection
    self._has_blinkers = car_state.leftBlinker or car_state.rightBlinker

    # sng detection
    if self._has_standstill:
      self._sng_state = SNG_State.stopped
      self._sng_transit_frame = 0
    else:
      if self._sng_transit_frame == 0:
        if self._sng_state == SNG_State.stopped:
          self._sng_state = SNG_State.going
          self._sng_transit_frame = STOP_AND_GO_FRAME
        elif self._sng_state == SNG_State.going:
          self._sng_state = SNG_State.off
      elif self._sng_transit_frame > 0:
        self._sng_transit_frame -= 1

    # slowness detection
    self._slowness_gmac.add_data(self._v_ego_kph <= (self._v_cruise_kph*SLOWNESS_CRUISE_OFFSET))
    self._has_slowness = self._slowness_gmac.get_moving_average() >= SLOWNESS_PROB

    # dangerous TTC detection
    if not self._has_lead_filtered and self._has_lead_filtered_prev:
      self._dangerous_ttc_gmac.reset_data()
      self._has_dangerous_ttc = False

    if self._has_lead and car_state.vEgo >= 0.01:
      self._dangerous_ttc_gmac.add_data(lead_one.dRel/car_state.vEgo)

    self._has_dangerous_ttc = self._dangerous_ttc_gmac.get_moving_average() is not None and self._dangerous_ttc_gmac.get_moving_average() <= DANGEROUS_TTC


    # keep prev values
    self._has_standstill_prev = self._has_standstill
    self._has_slowness_prev = self._has_slowness
    self._has_slow_down_prev = self._has_slow_down
    self._has_lead_filtered_prev = self._has_lead_filtered
    self._frame += 1

  def _blended_priority_mode(self):
    # when blinker is on and speed is driving below highway cruise speed: blended
    # we dont want it to switch mode at higher speed, blended may trigger hard brake
    if self._has_blinkers and self._v_ego_kph < HIGHWAY_CRUISE_KPH:
      #self._mode = 'blended'
      return

    # when at highway cruise and SNG: blended
    # ensuring blended mode is used because acc is bad at catching SNG lead car
    # especially those who accel very fast and then brake very hard.
    if self._sng_state == SNG_State.going and self._v_cruise_kph >= HIGHWAY_CRUISE_KPH:
      self._mode = 'blended'
      return

    # when standstill: blended
    # in case of lead car suddenly move away under traffic light, acc mode wont brake at traffic light.
    if self._has_standstill:
      self._mode = 'blended'
      return

    # when detecting slow down scenario: blended
    # e.g. traffic light, curve, stop sign etc.
    if self._has_slow_down:
      #self._mode = 'blended'
      return

    # when detecting lead slow down: blended
    # use blended for higher braking capability
    if self._has_dangerous_ttc:
      self._mode = 'blended'
      return

    # car driving at speed lower than set speed: acc
    if self._has_slowness:
      self._mode = 'acc'
      return

    self._mode = 'blended'

  def _acc_priority_mode(self):
    if self._has_lead_filtered:
      self._mode = 'acc'
      return

    if self._has_blinkers:
      self._mode = 'blended'
      return

    if self._has_standstill:
      self._mode = 'blended'
      return

    if self._has_slow_down:
      #self._mode = 'blended'
      return

    if self._has_slowness:
      self._mode = 'acc'
      return

    self._mode = 'acc'

  def get_mpc_mode(self, radar_unavailable, car_state, lead_one, md, controls_state):
    if self._is_enabled:
      self._update(car_state, lead_one, md, controls_state, radar_unavailable)
      if self._frame > self._mode_switch_frame:
        self._mode_switch_allowed = True
      if radar_unavailable:
        self._blended_priority_mode()
      else:
        self._blended_priority_mode()
        #self._acc_priority_mode()

    self._mode_prev = self._mode
    return self._mode

  def set_enabled(self, enabled):
    self._is_enabled = enabled

  def is_enabled(self):
    return self._is_enabled
