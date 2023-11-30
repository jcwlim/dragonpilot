using Cxx = import "./include/c++.capnp";
$Cxx.namespace("cereal");

using Custom = import "custom.capnp";

@0xf3b1f17e25a4285b;

const logVersion :Int32 = 1;

struct Event {


  # *********** Custom: reserved for forks ***********
  controlsStateExt @110 :Custom.ControlsStateExt;

}
