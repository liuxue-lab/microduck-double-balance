"""Compact live counters: emit in the worker, persist in the supervisor."""
import json
import os
import stat

PREFIX='NATIVE_PROGRESS_JSON='
_fd=None
_pipe_buf=0
_sent=_coalesced=0


def configure():
    global _fd,_pipe_buf,_sent,_coalesced
    value=os.environ.get('NATIVE_PROGRESS_FD')
    if value is None:return
    fd=int(value)
    if fd<=2 or not stat.S_ISFIFO(os.fstat(fd).st_mode):
        raise RuntimeError('Expected owned progress pipe')
    os.set_blocking(fd,False)
    _fd=fd;_pipe_buf=os.fpathconf(fd,'PC_PIPE_BUF');_sent=_coalesced=0


def snapshot():
    return dict(transport='nonblocking owned pipe' if _fd is not None else 'test stdout',
        sent=_sent,coalesced_live_display_updates=_coalesced,control_ticks_dropped=0)


def complete_lines(pending, chunk):
    rows=(pending+chunk).split('\n')
    return rows[:-1],rows[-1]


def emit(row, stream=None):
    global _sent,_coalesced
    import sys
    text=PREFIX+json.dumps(row,ensure_ascii=True,allow_nan=False,separators=(',',':'))+'\n'
    if stream is None and _fd is not None:
        data=text.encode('utf-8')
        if len(data)>_pipe_buf:raise RuntimeError('Progress message exceeds atomic pipe size')
        try:
            if os.write(_fd,data)!=len(data):raise RuntimeError('Partial progress pipe write')
            _sent+=1
        except BlockingIOError:
            # Live display can catch up with a later counter. Physics, actions,
            # timing rows and complete traces are never discarded.
            _coalesced+=1
        return
    stream=sys.stdout if stream is None else stream
    stream.write(text)
    stream.flush()


def parse(line):
    if not line.startswith(PREFIX):return None
    row=json.loads(line[len(PREFIX):])
    if not isinstance(row,dict) or row.get('total_control_steps')!=6000 \
            or type(row.get('completed_control_steps')) is not int \
            or not 0<=row['completed_control_steps']<=6000:
        raise RuntimeError('Invalid native progress record')
    return row


def display(row):
    return '闭环 %d/%d | %.2f%% | %s | %d/%d 步 | %.1f 秒' % (
        row['case_number'],row['cases_total'],row['percent'],row.get('detail',''),
        row['completed_control_steps'],row['total_control_steps'],row['elapsed_seconds'])
