"""Bring CARLA-Air back to a usable state after probe 2: AirSim unpaused, CARLA asynchronous."""
import airsim
import carla

ac = airsim.MultirotorClient(port=41451, timeout_value=15)
ac.confirmConnection()
print("airsim paused:", ac.simIsPause())
ac.simPause(False)
print("airsim paused after unpause:", ac.simIsPause())
client = carla.Client("localhost", 2000)
client.set_timeout(20.0)
world = client.get_world()
s = world.get_settings()
print("carla sync:", s.synchronous_mode, s.fixed_delta_seconds)
s.synchronous_mode = False
s.fixed_delta_seconds = None
world.apply_settings(s)
for a in world.get_actors().filter("vehicle.*"):
    a.destroy()
print("carla sync now:", world.get_settings().synchronous_mode, "frame", world.get_snapshot().frame)
p = ac.getMultirotorState().kinematics_estimated.position
print("drone ned:", round(p.x_val, 1), round(p.y_val, 1), round(p.z_val, 1))
