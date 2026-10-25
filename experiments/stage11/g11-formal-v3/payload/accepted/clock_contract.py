"""Private prototype clock; no manager/sensor adoption."""
from dataclasses import dataclass
import numpy as np
STATIC_ORIGINS=(0.,.02,.1,1.,10.,60.)
LONG_CASES=('normal-soft','floor-fast')
@dataclass
class TickClock:
 tick:int=0
 def __post_init__(self):
  if type(self.tick) is not int or self.tick<0:raise ValueError('Nonnegative integer tick required')
 @property
 def nanoseconds(self):return self.tick*2_000_000
 @property
 def published(self):return float(np.float32(self.nanoseconds/1_000_000_000))
 def advance(self,expected):
  if type(expected) is not int or expected!=self.tick:raise RuntimeError('Macro tick order differs')
  self.tick+=1;return self.published

def physics_hashes(hashes):
 if 'Data.time' not in hashes:raise RuntimeError('Data clock missing')
 return {k:v for k,v in hashes.items() if k!='Data.time'}

def publication_bound(tick,value):
 clock=TickClock(tick);expected=clock.nanoseconds/1_000_000_000
 if value!=clock.published:raise RuntimeError('Public clock is not the rounded integer tick')
 bound=.5*float(np.spacing(np.float32(value)))
 if abs(value-expected)>bound+1e-15:raise RuntimeError('Publication error exceeds half ULP')
 return dict(tick=tick,nominal_nanoseconds=clock.nanoseconds,published_time_s=value,rounding_error_s=value-expected,half_ulp_bound_s=bound)
