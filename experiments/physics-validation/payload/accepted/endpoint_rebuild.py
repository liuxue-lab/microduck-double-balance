"""Explicit materialization of the reviewed post-one-macro owner schema.

Only three cold-to-live tensors and five scalar slots are constructed. Actual
history, motor state, model, Data and RNG bytes are loaded from the checkpoint.
This is a macro-boundary recovery, not interruption in a private RK stage.
"""
import hashlib,json
from portable_state import PortableState,read,write
from lifecycle_support import require
from rebuild_boundary import differences
from report_io import save

def schema_digest(schema):
    return hashlib.sha256(json.dumps(schema,sort_keys=True,separators=(',',':')).encode()).hexdigest()

class EndpointPortableState(PortableState):
    def configure(self,act,env,bank,world,expected_schema_sha,out):
        self.config=(act,env,bank,world,expected_schema_sha,out);self.used=False
        return self

    def restore(self,path):
        require(not self.used,'Endpoint loader is single-use; preserve a failed attempt')
        self.used=True
        act,env,bank,world,expected,out=self.config
        raw,schema=read(path)
        require(schema_digest(schema)==expected,'Unreviewed post-macro schema')
        require(len(raw)==636,'Complete post-macro numeric field roster differs')
        cb=act._delay_buffer._buffer;firmware=act._bam_model.actuator
        require(cb._buffer is None and cb._pointer==-1 and act._friction_fields_checked is False,'Receiver BAM state is not cold')
        require(type(firmware.kp) is float and type(firmware.vin) is float,'Receiver firmware state is not cold')
        require(env._sim_step_counter==env.metrics_manager._substep_count==0,'Receiver clocks are not fresh')
        require(bank.draw_counts['bam_lag/'+str(world)]==0,'Receiver private BAM stream already advanced')
        cb._buffer=self.torch.zeros((7,1,14),device='cuda:0',dtype=self.torch.float32)
        firmware.kp=self.torch.zeros((1,1),device='cuda:0',dtype=self.torch.float32)
        firmware.vin=self.torch.zeros((1,1),device='cuda:0',dtype=self.torch.float32)
        cb._pointer=0;act._friction_fields_checked=True
        env._sim_step_counter=1;env.metrics_manager._substep_count=1
        bank.draw_counts['bam_lag/'+str(world)]=1
        current,meta=self.snapshot();meta=json.loads(json.dumps(meta))
        write(out/'receiver-materialized',current,meta)
        proof=dict(schema_exact=meta==schema,schema_differences=differences(schema,meta),
                   new_tensors=['BAM._delay_buffer._buffer._buffer','firmware.kp','firmware.vin'],
                   scalar_slots=['BAM._delay_buffer._buffer._pointer','BAM._friction_fields_checked',
                                 'env._sim_step_counter','MetricsManager._substep_count','RNGService.draw_counts.bam_lag/'+str(world)],
                   physics_integrations=0,action_applications=0,history_pushes=0,native_handles_imported=0)
        save(out/'endpoint-materialization.json',proof)
        require(proof['schema_exact'],'Post-macro structure/aliases differ before loading; raw saved')
        result=super().restore(path)
        result['scope']='reviewed post-one-accepted-macro complete numeric state and selected owners, with live BAM delay/firmware and clocks'
        result['materialization']=proof
        return result
