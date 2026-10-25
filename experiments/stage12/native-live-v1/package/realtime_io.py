"""Bounded in-memory traces and scoped GC suspension for the control interval."""
import gc
import os
import resource
from pathlib import Path


def need(ok,message):
    if not ok:raise RuntimeError(message)


class DeferredTrace:
    def __init__(self,path,capacity,max_rows):
        self.path=Path(path);self.capacity=capacity;self.max_rows=max_rows
        self.buffer=bytearray(capacity)
        # Touch each page before the first control release.
        for i in range(0,capacity,4096):self.buffer[i]=1;self.buffer[i]=0
        self.file=self.path.open('xb')
        self.used=self.rows=self.flush_requests=0
        self.closed=self.durable=False

    def write(self,text):
        need(not self.closed,'Closed deferred trace')
        data=text.encode('utf-8')
        need(text.endswith('\n') and text.count('\n')==1,'Trace requires one complete record')
        need(self.rows<self.max_rows and self.used+len(data)<=self.capacity,'Deferred trace budget exhausted')
        self.buffer[self.used:self.used+len(data)]=data
        self.used+=len(data);self.rows+=1
        return len(text)

    def flush(self):
        need(not self.closed,'Closed deferred trace')
        self.flush_requests+=1

    def close(self):
        if self.closed:return
        try:
            self.file.write(memoryview(self.buffer)[:self.used])
            self.file.flush();os.fsync(self.file.fileno())
            self.durable=True
        finally:
            self.file.close();self.closed=True
            self.buffer=None

    def snapshot(self):
        return dict(path=str(self.path),rows=self.rows,bytes=self.used,capacity_bytes=self.capacity,
            flush_requests=self.flush_requests,closed=self.closed,durable=self.durable,
            dropped_records=0,control_interval_disk_writes=0)


class ControlGC:
    def __init__(self):
        self.was_enabled=gc.isenabled();self.closed=False
        gc.collect()
        self.before=[x['collections'] for x in gc.get_stats()]
        self.resources_before=resource.getrusage(resource.RUSAGE_SELF)
        gc.disable()
        self.after=None

    def close(self):
        if self.closed:return
        self.after=[x['collections'] for x in gc.get_stats()]
        self.resources_after=resource.getrusage(resource.RUSAGE_SELF)
        if self.was_enabled:gc.enable()
        self.closed=True

    def snapshot(self):
        return dict(cyclic_gc_paused=True,reference_counting_unchanged=True,
            collections_before=self.before,collections_after=self.after,
            control_interval_collections=None if self.after is None else [b-a for a,b in zip(self.before,self.after)],
            original_enabled=self.was_enabled,restored=self.closed and gc.isenabled()==self.was_enabled,
            major_page_faults=None if not self.closed else self.resources_after.ru_majflt-self.resources_before.ru_majflt,
            minor_page_faults=None if not self.closed else self.resources_after.ru_minflt-self.resources_before.ru_minflt,
            involuntary_context_switches=None if not self.closed else self.resources_after.ru_nivcsw-self.resources_before.ru_nivcsw,
            peak_process_RSS_MiB=None if not self.closed else self.resources_after.ru_maxrss/1024.)


def export_models(pending):
    """Large immutable model images are written after all control intervals."""
    receipts=[]
    for path,buffer in pending:
        with Path(path).open('xb') as stream:
            stream.write(memoryview(buffer));stream.flush();os.fsync(stream.fileno())
        directory=os.open(Path(path).parent,os.O_RDONLY|os.O_DIRECTORY)
        try:os.fsync(directory)
        finally:os.close(directory)
        receipts.append(dict(path=str(path),bytes=int(buffer.nbytes),durable=True))
    pending.clear()
    return receipts
