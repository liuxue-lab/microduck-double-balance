"""Bounded, read-only internet/DNS diagnostics after the HIL workers exit."""
import subprocess


def command(argv):
    try:
        result = subprocess.run(argv, capture_output=True, text=True, timeout=6)
        return dict(returncode=result.returncode, stdout=result.stdout[-6000:], stderr=result.stderr[-2000:])
    except (OSError, subprocess.TimeoutExpired) as exc:
        return dict(returncode=None, error=type(exc).__name__)


def snapshot():
    dns = command(['getent','ahostsv4','github.com'])
    return dict(schema='stage12-post-run-dns-diagnostic-v1', changes_made=False,
        Git_push_attempted=False, executed_after_control=True,
        github_dns_resolves=dns.get('returncode')==0 and bool(dns.get('stdout','').strip()),
        github_lookup=dns,
        default_route=command(['ip','-j','-4','route','show','default']),
        interfaces=command(['nmcli','-t','-f','DEVICE,TYPE,STATE','device','status']),
        wifi_dns=command(['nmcli','-f','GENERAL.STATE,IP4.GATEWAY,IP4.DNS','device','show','wlo1']),
        resolver=command(['resolvectl','status','--no-pager']))
