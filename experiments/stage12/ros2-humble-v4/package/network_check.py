"""Use the existing verified Ethernet profile; never rewrite address or DNS."""
import json
from pathlib import Path
import socket
import subprocess

from probe import clean_env, command

PROPERTIES = ('connection.interface-name','ipv4.method','ipv4.addresses',
              'ipv4.gateway','ipv4.dns','ipv4.never-default','ipv6.method')


def profile_matches(values):
    return (values.get(PROPERTIES[0])=='enp4s0' and values.get(PROPERTIES[1])=='manual'
            and values.get(PROPERTIES[2])=='192.168.50.1/24'
            and values.get(PROPERTIES[3]) in ('','--')
            and values.get(PROPERTIES[4]) in ('','--')
            and values.get(PROPERTIES[5])=='yes' and values.get(PROPERTIES[6])=='disabled')


def snapshot():
    values={}
    for name in PROPERTIES:
        result=command(['nmcli','-g',name,'connection','show','jetson-direct'])
        values[name]=result.get('stdout','').strip() if result['returncode']==0 else None
    def text(path):
        try:return Path(path).read_text().strip()
        except OSError:return None
    result=dict(carrier=text('/sys/class/net/enp4s0/carrier'),
                operstate=text('/sys/class/net/enp4s0/operstate'),profile=values,
                profile_matches_verified_guide=profile_matches(values),
                active_connection=command(['nmcli','-g','GENERAL.CONNECTION','device','show','enp4s0']),
                addresses=command(['ip','-j','-4','address','show','dev','enp4s0']),
                route=command(['ip','-j','-4','route','get','192.168.50.2']),
                neighbour=command(['ip','-j','neighbour','show','to','192.168.50.2','dev','enp4s0']),
                default_routes=command(['ip','-j','-4','route','show','default']))
    return result


def tcp_ready():
    try:
        with socket.create_connection(('192.168.50.2',22),timeout=3):
            return dict(ready=True,error=None)
    except OSError as exc:
        return dict(ready=False,error=str(exc),errno=exc.errno)


def active_correct(s):
    try:
        addresses=json.loads(s['addresses']['stdout']);route=json.loads(s['route']['stdout'])[0]
        return (s['active_connection'].get('stdout','').strip()=='jetson-direct'
                and any(a.get('local')=='192.168.50.1' and a.get('prefixlen')==24
                        for row in addresses for a in row.get('addr_info',[]))
                and route.get('dev')=='enp4s0' and route.get('prefsrc')=='192.168.50.1')
    except (KeyError,ValueError,IndexError):
        return False


def diagnose(activate=False, sudo=None):
    before=snapshot();activation=None
    if (activate and before['profile_matches_verified_guide']
            and not active_correct(before)):
        # Connection up activates the stored, verified profile; no modify/down,
        # no changes to Wi-Fi, firewall, addresses, routes, SSH or resolver.
        argv=sudo()+['nmcli','--wait','15','connection','up','id','jetson-direct','ifname','enp4s0']
        p=subprocess.run(argv,env=clean_env(),text=True,capture_output=True,timeout=25)
        activation=dict(returncode=p.returncode,stdout=p.stdout,stderr=p.stderr)
    after=snapshot()
    if after['profile']!=before['profile']:
        raise RuntimeError('Network profile changed unexpectedly; preserve and stop')
    ping=command(['ping','-I','enp4s0','-c','2','-W','1','192.168.50.2'],timeout=5)
    port=tcp_ready() if active_correct(after) else dict(ready=False,error='Existing wired address/route not active')
    neighbour=command(['ip','-j','neighbour','show','to','192.168.50.2','dev','enp4s0'])
    if after['carrier']!='1':
        status='PHYSICAL_LINK_DOWN'
        next_step='给 Orin 上电并等待启动，检查或重新插拔两端网线；carrier 必须为 1。'
    elif not after['profile_matches_verified_guide'] or not active_correct(after):
        status='PC_PROFILE_NOT_READY'
        next_step='电脑现有有线连接与指南不符，报告已保留完整属性；不要改 Wi-Fi 默认路由。'
    elif port['ready']:
        status='SSH_PORT_READY';next_step=None
    elif port.get('errno')==111:
        status='SSH_SERVICE_REFUSED'
        next_step='在 Orin 本机检查 sudo systemctl status ssh --no-pager；未启动时执行 sudo systemctl start ssh。'
    else:
        status='ORIN_UNREACHABLE'
        next_step='在 Orin 本机终端执行 sudo nmcli connection up "jetson-wired"，并检查 ip -4 -br addr show enP8p1s0 是否有 192.168.50.2/24。'
    return dict(status=status,before=before,after=after,activation=activation,ping=ping,ssh_port=port,
                neighbour_after_attempt=neighbour,next_step=next_step,configuration_rewritten=False)
