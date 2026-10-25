"""Two bounded local visual rollouts; never a replacement success-rate benchmark."""
from copy import deepcopy
from dataclasses import asdict
import gc
import html
import importlib.metadata
import importlib.util
import math
import os
from pathlib import Path
import shutil
import signal
import struct
import subprocess
import sys
import time
import traceback
import zlib
from replay_common import *
from observer import Observer,fingerprint

def find_encoder():
    path=shutil.which('ffmpeg')
    if path is None and importlib.util.find_spec('imageio_ffmpeg') is not None:
        import imageio_ffmpeg
        path=imageio_ffmpeg.get_ffmpeg_exe()
    need(path is not None and Path(path).is_file(),'No existing ffmpeg found; nothing installed and no simulation started')
    result=subprocess.run([path,'-hide_banner','-encoders'],capture_output=True,timeout=15)
    need(result.returncode==0 and b'libx264' in result.stdout,'Existing ffmpeg lacks libx264; no simulation started')
    return path

def png(path,frame):
    def chunk(kind,data):return struct.pack('!I',len(data))+kind+data+struct.pack('!I',zlib.crc32(kind+data)&0xffffffff)
    height,width,channels=frame.shape;need(channels==3 and str(frame.dtype)=='uint8','Unexpected video frame')
    pixels=b''.join(b'\x00'+frame[y].tobytes() for y in range(height))
    data=b'\x89PNG\r\n\x1a\n'+chunk(b'IHDR',struct.pack('!IIBBBBB',width,height,8,2,0,0,0))+chunk(b'IDAT',zlib.compress(pixels))+chunk(b'IEND',b'')
    with path.open('xb') as f:f.write(data)

def page(clips):
    parts=['<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>Stage12 4500本机回放</title><style>body{max-width:1280px;margin:24px auto;padding:0 18px;background:#111827;color:#e5e7eb;font:17px sans-serif}video{width:100%;background:black}section{margin:28px 0}a{color:#93c5fd}</style><h1>4500模型 · 本机视觉回放</h1><p>左：全身与脚下球；右：头壳与上球特写。两视角来自同一仿真时刻。固定相机；若主体移出特写范围，以全身画面为准。</p><p>仅两段单环境示例，不是云端128环境轨迹的精确重放，也不重新计算82.8%成功率。</p>']
    for clip in clips:
        name=html.escape(Path(clip['video']['path']).name);poster=html.escape(Path(clip['initial_frame']).name)
        r=clip['episode']
        parts.append('<section><h2>种子 %s</h2><p>实际仿真 %.2f 秒；最长连续稳定 %.2f 秒；本回合5秒条件：%s；物理终止：%s。</p><video controls preload="metadata" poster="%s" src="%s"></video><p><a href="%s">打开视频</a></p></section>'%(clip['seed'],r['steps']*.02,r['longest_stable_steps']*.02,'满足' if r['success'] else '未满足','是' if r['terminated'] else '否',poster,name,name))
    parts.append('</html>');target=SLOT/'index.html';temporary=target.with_suffix('.tmp')
    temporary.write_text(''.join(parts),encoding='utf-8');os.replace(temporary,target)

def main():
    started=read(SLOT/'started.json');deadline=started['deadline_utc'];stopped=[False];clips=[]
    progress=dict(status='INITIALIZING',phase='load_runtime',completed_control_steps=0,maximum_control_steps=1000,
                  progress_percent=0.,completed_clips=0,maximum_clips=2,added_PPO_updates=0,deadline_utc=deadline)
    def stop(signum,*_):
        stopped[0]=True
        raise RuntimeError('Local visual rollout interrupted by signal %s; preserve outputs'%signum)
    for sig in (signal.SIGINT,signal.SIGTERM,signal.SIGALRM):signal.signal(sig,stop)
    signal.alarm(max(1,math.ceil(deadline-time.time())))
    def boundary():need(not stopped[0] and time.time()<deadline,'Local visual rollout stopped or original 15-minute limit reached')
    def update(**kwargs):
        progress.update(kwargs,elapsed_seconds=time.time()-started['started_utc']);write(SLOT/'progress.json',progress)
    try:
        update();before_inputs=verify_inputs();encoder=find_encoder();boundary()
        expected={'mjlab':'1.3.0','mujoco':'3.10.0','mujoco-warp':'3.8.1','warp-lang':'1.12.0','rsl-rl-lib':'5.0.1','torch':'2.9.1'}
        need({k:importlib.metadata.version(k) for k in expected}==expected,'Local runtime versions changed')
        sys.path.insert(0,str(PROJECT/'src'));sys.path.insert(0,str(SELECTED/'task'))
        import numpy as np
        import torch
        from tensordict import TensorDict
        from mjlab.envs import ManagerBasedRlEnv
        from mjlab.rl import MjlabOnPolicyRunner,RslRlVecEnvWrapper
        from mjlab.utils.torch import configure_torch_backends
        from mjlab.viewer.offscreen_renderer import OffscreenRenderer
        from mjlab.viewer.viewer_config import ViewerConfig
        from mjlab_microduck.tasks.microduck_basketball_env_cfg import MicroduckBasketballRlCfg
        from mjlab_microduck.tasks import mdp
        from mjlab_microduck.robot.head_reference import REFERENCE_SITE
        from native_task import make_cfg,head_scale
        need(torch.cuda.is_available() and '5060' in torch.cuda.get_device_name(0),'This visual rollout is for the local RTX5060 only')
        configure_torch_backends();torch.set_num_threads(min(4,len(os.sched_getaffinity(0))))
        for index,seed in enumerate(SEEDS):
            boundary();raw=env=observer=writer=None;renderers=[];log=None;frames=0
            video=SLOT/('4500-seed-%d-whole-head.mp4'%seed)
            update(status='RECORDING',phase='build_environment',seed=seed,clip=index+1,control_steps_in_clip=0)
            try:
                cfg=make_cfg(1,seed);cfg.auto_reset=False
                need(abs(cfg.episode_length_s-10.)<1e-12 and not cfg.curriculum,'Visual rollout horizon or curriculum changed')
                raw=ManagerBasedRlEnv(cfg,device='cuda:0',render_mode=None);env=RslRlVecEnvWrapper(raw)
                agent=asdict(deepcopy(MicroduckBasketballRlCfg));agent.update(seed=seed,logger='tensorboard',upload_model=False)
                agent['algorithm'].update(learning_rate=5e-5,schedule='fixed',num_mini_batches=4)
                runner=MjlabOnPolicyRunner(env,agent,log_dir=None,device='cuda:0')
                saved=torch.load(MODEL,map_location='cpu',weights_only=False)
                need(saved['infos']['stage12_native']['package_identity']==ORIGIN_ID and saved['infos']['stage12_native']['policy_updates']==4500,'Checkpoint metadata changed')
                runner.alg.load(saved,{'actor':True,'critic':True,'optimizer':False,'iteration':False},strict=True)
                policy=runner.get_inference_policy(device='cuda:0');policy.reset()
                need(not policy.training and not policy.obs_normalizer.training,'Policy and normalization must remain frozen')
                before=fingerprint(policy.state_dict());raw.common_step_counter=1000*24;raw._sim_step_counter=1000*24*10
                raw_obs,_=raw.reset(seed=seed);obs=TensorDict(raw_obs,batch_size=[1]);policy.reset()
                need(raw.max_episode_length==500 and abs(raw.step_dt-.02)<1e-12 and head_scale(raw)==1.,'Visual rollout time contract changed')
                need(tuple(obs['actor'].shape)==(1,61) and tuple(obs['critic'].shape)==(1,85),'Observation layout changed')
                robot=raw.scene['robot'];site_ids,_=robot.find_sites('^'+REFERENCE_SITE+'$');need(len(site_ids)==1,'Head reference missing')
                head=robot.data.site_pose_w[0,site_ids[0],:3].detach().cpu().numpy().copy()
                ball=raw.scene['ball'].data.root_link_pos_w[0].detach().cpu().numpy().copy()
                views=[dict(lookat=tuple(float(x) for x in (head+ball)/2),distance=1.0,azimuth=135.,elevation=-15.),
                       dict(lookat=(float(head[0]),float(head[1]),float(head[2]+.02)),distance=.30,azimuth=45.,elevation=-12.)]
                for view in views:
                    renderer=OffscreenRenderer(raw.sim.mj_model,ViewerConfig(origin_type=ViewerConfig.OriginType.WORLD,
                        width=640,height=480,max_extra_envs=0,enable_reflections=False,**view),raw.scene)
                    renderers.append(renderer);renderer.initialize()
                def frame():
                    images=[]
                    for renderer in renderers:renderer.update(raw.sim.data);images.append(renderer.render())
                    out=np.concatenate(images,axis=1);need(out.shape==(480,1280,3) and out.dtype==np.uint8,'Unexpected render output')
                    return np.ascontiguousarray(out)
                first=SLOT/('4500-seed-%d-initial.png'%seed);png(first,frame());boundary()
                log=(SLOT/('ffmpeg-%d.log'%seed)).open('xb')
                writer=subprocess.Popen([encoder,'-hide_banner','-loglevel','error','-nostdin','-f','rawvideo','-pix_fmt','rgb24',
                    '-s','1280x480','-r','25','-i','pipe:0','-an','-c:v','libx264','-preset','veryfast','-crf','20',
                    '-threads','2','-pix_fmt','yuv420p','-movflags','+faststart','-n',str(video)],stdin=subprocess.PIPE,stdout=log,stderr=log)
                observer=Observer(raw,torch,mdp);raw.metrics_manager.compute=observer.compute
                terminations=[];physical=False;last_frame=None
                with torch.inference_mode():
                    for step in range(1,501):
                        boundary();actions=policy(obs,stochastic_output=False)
                        need(bool(torch.isfinite(actions).all()),'Nonfinite policy action')
                        obs,rewards,dones,_=env.step(actions)
                        need(bool(torch.stack([torch.isfinite(v).all() for v in obs.values()]+[torch.isfinite(rewards).all()]).all()),'Nonfinite observation/reward')
                        done=bool(dones[0])
                        if step%2==0 or done:
                            last_frame=frame();writer.stdin.write(last_frame.tobytes());frames+=1
                            if step==250:png(SLOT/('4500-seed-%d-middle.png'%seed),last_frame)
                        if step%25==0 or done:
                            update(phase='deterministic_video',control_steps_in_clip=step,
                                completed_control_steps=sum(c['episode']['steps'] for c in clips)+step,
                                progress_percent=100*(index+step/500)/2)
                        if done:
                            physical=bool(raw.reset_terminated[0])
                            terminations=[name for name in raw.termination_manager.active_terms if bool(raw.termination_manager.get_term(name)[0])]
                            break
                need(done,'First visual episode did not end by 500 control steps')
                need(fingerprint(policy.state_dict())==before,'Model or normalization changed during visual rollout')
                writer.stdin.close();writer.wait(timeout=30);need(writer.returncode==0,'Video encoder failed; see ffmpeg log')
                decode=subprocess.run([encoder,'-nostdin','-v','error','-i',str(video),'-f','null','-'],stdin=subprocess.DEVNULL,capture_output=True,timeout=30)
                need(decode.returncode==0,'Recorded video could not be decoded')
                need(last_frame is not None and frames>0,'No frames recorded');png(SLOT/('4500-seed-%d-final.png'%seed),last_frame)
                episode=observer.rows(seed,[physical],[terminations])[0]
                clip=dict(seed=seed,episode=episode,video=dict(path=str(video),bytes=video.stat().st_size,sha256=sha(video),frames=frames,fps=25,decode_verified=True),
                    initial_frame=str(first),views=views,policy_and_normalizer_unchanged=True,
                    not_exact_replay_of_cloud_batch=True,benchmark_success_rate_recomputed=False)
                write(SLOT/('clip-%d.json'%seed),clip,exclusive=True);clips.append(clip);page(clips)
                update(completed_clips=len(clips),progress_percent=100*len(clips)/2)
            finally:
                if observer is not None:raw.metrics_manager.compute=observer.original
                if writer is not None and writer.poll() is None:
                    try:
                        if writer.stdin and not writer.stdin.closed:writer.stdin.close()
                        writer.wait(timeout=10)
                    except Exception:
                        writer.kill();writer.wait(timeout=10)
                if log is not None:log.close()
                for renderer in reversed(renderers):renderer.close()
                if env is not None:env.close()
                elif raw is not None:raw.close()
            del policy,runner,saved,env,raw,observer,renderers
            gc.collect();torch.cuda.empty_cache()
        need(verify_inputs()==before_inputs,'Original selected files or project changed')
        result=dict(status='LOCAL_VIDEO_COMPLETE',clips=clips,inputs=before_inputs,original_inputs_unchanged=True,
            control_steps=sum(c['episode']['steps'] for c in clips),maximum_control_steps=1000,physics_substeps_per_control=10,
            added_PPO_updates=0,cloud_connections=0,task_acceptance_claimed=False,benchmark_success_rate_recomputed=False,
            device=torch.cuda.get_device_name(0),python=sys.version,torch=torch.__version__,index=str(SLOT/'index.html'),finished_utc=time.time())
        boundary();write(SLOT/'result.json',result,exclusive=True);update(status='LOCAL_VIDEO_COMPLETE',phase='finished',progress_percent=100.)
        signal.alarm(0)
        return 0
    except BaseException as exc:
        signal.alarm(0)
        update(status='LOCAL_VIDEO_ERROR_PRESERVE',phase=progress.get('phase'),error=repr(exc),traceback=traceback.format_exc()[-10000:],completed_clips=len(clips))
        print('LOCAL_VIDEO_ERROR='+repr(exc),flush=True);return 2

if __name__=='__main__':raise SystemExit(main())
