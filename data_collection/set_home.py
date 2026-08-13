"""
Set home position for the Dobot Magician robotic arm.
"""
import time
import dobot.DobotDllType as dType
from ctypes import c_uint64, byref

# 1. Load DLL and connect to the robotic arm
api = dType.load()
CON_STR = {
    dType.DobotConnect.DobotConnect_NoError:  "DobotConnect_NoError",
    dType.DobotConnect.DobotConnect_NotFound: "DobotConnect_NotFound",
    dType.DobotConnect.DobotConnect_Occupied: "DobotConnect_Occupied"
}

# Try to connect
connectResult = dType.ConnectDobot(api, "/dev/ttyUSB0", 115200)
state = connectResult[0]
print("Connect status:", CON_STR[state])

if state == dType.DobotConnect.DobotConnect_NoError:
    print("Connection successful!")
    
    # Clear alarms
    dType.ClearAllAlarmsState(api)
    print("Alarms cleared!")
    
    # Set home parameters (default axis order: J1, J2, J3, J4)
    dType.SetHOMEParams(api, x=200, y=0, z=0, r=0, isQueued=1)  # Set home position (optional)
    print("Home parameters set!")

    # Clear command queue and start execution
    dType.SetQueuedCmdClear(api)
    dType.SetQueuedCmdStartExec(api)

    # Send home command
    homeIndex = dType.SetHOMECmd(api, temp=0, isQueued=1)[0]
    print("Home command sent, waiting for completion...")

    # Wait for homing to complete
    # while True:
    #     currentIndex = dType.GetQueuedCmdCurrentIndex(api)[0]
    #     if currentIndex >= homeIndex:
    #         print("Homing completed!")
    #         break
    #     time.sleep(0.1)

    # Disconnect
    dType.SetQueuedCmdStopExec(api)
    dType.DisconnectDobot(api)
    time.sleep(1)
    print("Disconnected.")

else:
    print("Connection failed, please check the device or port.")
