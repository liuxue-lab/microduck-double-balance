"""Read-only pin for the independently reviewed six-cell cache/rebuild return."""
from pathlib import Path,PurePosixPath
import hashlib,json,zipfile
RUN_ID='20261015T064724Z-e6f7a8'
COMMIT='cd3c0926c137bf91a1da3da5e65282b64564b284'
BUNDLE='4a96eaeba432b4d3dadaad8fea4661e09534b754aea051bbffc578d2fcd150ce'
ARCHIVE='03ae46ec915af7917f36a5526812a9e61b57e7e1547e4e77b3300bd391401aaf'
def need(ok,msg):
    if not ok:raise RuntimeError(msg)
def sha(raw):return hashlib.sha256(raw).hexdigest()
def verify(root,current_git):
    base=Path(root)/'g10-rk4-cache-resume';run=base/RUN_ID;marker=base/'started-v1.json'
    m=json.loads(marker.read_bytes())
    need(m['run']==str(run) and m['bundle_sha256']==BUNDLE,'Cache-resume started marker differs')
    archive=run/'g10-rk4-cache-resume-return.zip'
    need(not archive.is_symlink() and archive.stat().st_size==178121576 and sha(archive.read_bytes())==ARCHIVE,'Cache-resume archive differs')
    with zipfile.ZipFile(archive) as z:
        inv=json.loads(z.read('evidence-inventory.json'));names=z.namelist()
        need(len(names)==len(set(names))==8201 and set(names)==set(inv)|{'evidence-inventory.json'},'Cache-resume roster differs')
        for n,row in inv.items():
            p=PurePosixPath(n);local=run/n;raw=z.read(n)
            need(not p.is_absolute() and '..' not in p.parts and '\\' not in n and not local.is_symlink() and local.resolve().is_relative_to(run.resolve()),'Unsafe cache-resume path')
            need(row==dict(bytes=len(raw),sha256=sha(raw)) and local.read_bytes()==raw,'Cache-resume evidence changed: '+n)
        need((run/'evidence-inventory.json').read_bytes()==z.read('evidence-inventory.json'),'Cache inventory changed')
        j=lambda n:json.loads(z.read(n))
        s=j('run-summary.json');f=j('execution-finished.json');a=j('new-raw-review.json')
        need(s['canonical_completed']==667608 and s['canonical_attempted']==667609,'Historical cache ledger differs')
        need(s['counts']['integrations_completed']==s['counts']['integrations_attempted']==1408 and s['counts']['GPU_forward_completed']==5658,'Cache physical ledger differs')
        need(s['worker_status']==a['status']=='G10_RESUME_RAW_HOST_REVIEW_PASS' and a['comparisons_passed']==a['comparisons_total']==6,'Cache matrix incomplete')
        need(j('git-before.json')==j('git-after.json')==current_git,'Project state differs from cache return')
        need(f['process_has_exited'] and f['exit_code']==0 and not f['timed_out'] and not f['interrupted'],'Cache process receipt differs')
        need(s['source_unchanged'] and s['old_evidence_unchanged'] and s['git_state_unchanged'] and s['counters_consistent'],'Cache protection failed')
    return dict(commit=COMMIT,run=str(run),archive_sha256=ARCHIVE,marker_sha256=sha(marker.read_bytes()),members=8201,
                comparisons_passed=6,native_calls=0,canonical_completed=667608,canonical_attempted=667609,prior_counts=s['counts'])
