import psutil
import platform
import getpass
import datetime
from pynvml import *
from pprint import pprint

def physical_system_time():
    return {"system_time": datetime.datetime.fromtimestamp(psutil.boot_time()).strftime("%Y-%m-%d %H:%M:%S")}

def physical_username():
    return {
        "system_user": getpass.getuser()
    }

def physical_platfrom_system():
    u_name = platform.uname()
    return {"system_name": u_name.system, "system_version": u_name.version}

def physical_cpu():
    return {"system_cpu_count": psutil.cpu_count(logical=False)}

def physical_memory():
    # return round(psutil.virtual_memory().total / (1024.0 * 1024.0 * 1024.0), 2)
    return {"system_memory": round(psutil.virtual_memory().total, 2)}

def physical_hard_disk():
    result = []
    for disk_partition in psutil.disk_partitions():
        o_usage = psutil.disk_usage(disk_partition.device)
        result.append(
            {
                "device": disk_partition.device,
                "fstype":disk_partition.fstype,
                "opts": disk_partition.opts,
                "total": o_usage.total,
            }
        )
    return {"system_hard_disk": result}

def nvidia_info():
    nvidia_dict = {
        "state": True,
        "nvidia_version": "",
        "nvidia_count": 0,
        "gpus": []
    }
    try:
        nvmlInit()
        nvidia_dict["nvidia_version"] = nvmlSystemGetDriverVersion()
        nvidia_dict["nvidia_count"] = nvmlDeviceGetCount()
        for i in range(nvidia_dict["nvidia_count"]):
            handle = nvmlDeviceGetHandleByIndex(i)
            memory_info = nvmlDeviceGetMemoryInfo(handle)
            gpu = {
                "gpu_name": nvmlDeviceGetName(handle),
                "total": memory_info.total,
                "free": memory_info.free,
                "used": memory_info.used,
                "temperature": f"{nvmlDeviceGetTemperature(handle, 0)}℃",
                "powerStatus": nvmlDeviceGetPowerState(handle)
            }
            nvidia_dict['gpus'].append(gpu)
    except NVMLError as e1:
        nvidia_dict["state"] = False
        print(f'nvidia_error_1_{e1}')
    except Exception as e2:
        nvidia_dict["state"] = False
        print(f'nvidia_erro_2_{e2}')
    finally:
        try:
            nvmlShutdown()
        except:
            pass
    return nvidia_dict

def merge(info_list):
    data = {}
    for item in info_list:
        data.update(
            item()
        )
    return data

def computer_info():
    data = merge(
        [
            physical_system_time,
            physical_username,
            physical_platfrom_system,
            physical_cpu,
            physical_memory,
            # physical_hard_disk,
            nvidia_info
        ]
    )
    pprint(data)

