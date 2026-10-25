"""Place only owned workers on separate eligible physical CPU cores.

Respect the inherited affinity/cpuset. Prefer higher-capacity/frequency cores,
then lower recent load. No priority, governor, service, or global setting changes.
"""
import os
from pathlib import Path
import time


def counters():
    result={}
    for line in Path('/proc/stat').read_text().splitlines():
        fields=line.split()
        if fields and fields[0].startswith('cpu') and fields[0][3:].isdigit():
            values=list(map(int,fields[1:9]))
            result[int(fields[0][3:])]=(sum(values),values[3]+values[4])
    return result


def number(path,default=0):
    try:return int(Path(path).read_text().strip())
    except (OSError,ValueError):return default


def select(rows):
    ordered=sorted(rows,key=lambda r:(-r['capacity'],-r['maximum_frequency_kHz'],r['busy_fraction'],r['cpu']))
    if not ordered:return None,None
    control=ordered[0]
    distinct=[r for r in ordered[1:] if (r['package'],r['core'])!=(control['package'],control['core'])]
    # If there is only one physical core, preserve unrestricted bridge placement.
    return control['cpu'],distinct[0]['cpu'] if distinct else None


def plan():
    if not hasattr(os,'sched_getaffinity'):
        return dict(enabled=False,reason='CPU affinity unavailable')
    allowed=sorted(os.sched_getaffinity(0))
    try:
        first=counters();time.sleep(.15);last=counters();rows=[]
        for cpu in allowed:
            path=Path('/sys/devices/system/cpu')/('cpu'+str(cpu))
            a=first.get(cpu,(0,0));b=last.get(cpu,a);total=b[0]-a[0]
            rows.append(dict(cpu=cpu,capacity=number(path/'cpu_capacity'),
                maximum_frequency_kHz=number(path/'cpufreq/cpuinfo_max_freq'),
                busy_fraction=max(0.,min(1.,1-(b[1]-a[1])/total)) if total else 1.,
                package=number(path/'topology/physical_package_id'),core=number(path/'topology/core_id',cpu)))
        control,bridge=select(rows)
        return dict(enabled=control is not None,control_cpu=control,bridge_cpu=bridge,
                    inherited_allowed_cpus=allowed,candidates=rows,system_settings_changed=False)
    except OSError as exc:
        return dict(enabled=False,reason=repr(exc),inherited_allowed_cpus=allowed)
