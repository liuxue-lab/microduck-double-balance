"""Self-contained local replay page; embeds timing data, no service or CDN."""
import html
import json
from pathlib import Path


def page(output, report, clips):
    rows = []
    for clip in clips:
        rows.append(dict(case_id=clip['case_id'], video=Path(clip['video']['path']).name,
            poster=Path(clip['snapshots']['initial']).name, timing=clip['plot'],
            statistics=clip['timing_statistics'], frames=clip['video']['frames'],
            first_frame_simulation_time_s=.02, last_frame_simulation_time_s=60.,
            perturbations=clip['perturbation_pulses']))
    payload = json.dumps(rows, ensure_ascii=False, separators=(',', ':')).replace('<', '\\u003c')
    content = TEMPLATE.replace('@@DATA@@', payload).replace('@@SOURCE@@', html.escape(report['directory']))
    path = Path(output)/'index.html'
    with path.open('x', encoding='utf-8') as stream:
        stream.write(content)
    return path


TEMPLATE = r'''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Microduck · 已验收 HIL 轨迹回放</title>
<style>
:root{color-scheme:dark;font:16px/1.65 system-ui,sans-serif;background:#0b1220;color:#e2e8f0}*{box-sizing:border-box}body{margin:0}main{max-width:1320px;margin:0 auto;padding:32px 24px 60px}h1{font-size:30px;margin:0 0 8px;letter-spacing:.02em}h2{margin:0;font-size:23px}.eyebrow{color:#66dfc3;font-size:13px;letter-spacing:.15em}.intro,.muted{color:#a9b8cb}.intro{max-width:1000px}.metrics{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:24px 0}.metric{border:1px solid #29384c;border-radius:12px;background:#121e30;padding:16px}.metric strong{font-size:26px;display:block;color:#fff}.metric span{color:#a9b8cb;font-size:13px}section{margin:32px 0;padding:22px;background:#111d2e;border:1px solid #29384c;border-radius:15px}video{display:block;width:100%;aspect-ratio:8/3;background:#03070d;border-radius:8px}.views{display:grid;grid-template-columns:1fr 1fr;text-align:center;color:#a9b8cb;font-size:13px;margin:10px 0 6px}.toolbar{display:flex;flex-wrap:wrap;align-items:center;gap:10px;margin:14px 0}button,select{font:inherit;background:#1b2d43;color:#e2e8f0;border:1px solid #41526a;border-radius:7px;padding:6px 12px;cursor:pointer}button:focus-visible,select:focus-visible,a:focus-visible{outline:2px solid #66dfc3;outline-offset:3px}.clock{font-variant-numeric:tabular-nums;color:#66dfc3;margin-left:auto}canvas{display:block;width:100%;height:250px;cursor:crosshair}a{color:#8ecaff}.legend{display:flex;gap:18px;font-size:13px;flex-wrap:wrap}.legend span::before{content:'●';margin-right:6px}.total{color:#66dfc3}.physics{color:#82b4ff}.ros{color:#ffcc85}.limit{color:#ff8d9c}code{overflow-wrap:anywhere}.detail{font-size:13px;color:#a9b8cb}.status{display:inline-block;background:#12453b;color:#9af0d7;border-radius:5px;padding:2px 9px;font-size:13px;margin-bottom:10px}summary{cursor:pointer}footer{color:#94a3b8;font-size:13px}@media(max-width:700px){main{padding:20px 12px}.metrics{grid-template-columns:repeat(2,1fr)}section{padding:12px}.clock{width:100%;margin:0}h1{font-size:24px}canvas{height:230px}}
</style></head><body><main>
<div class="eyebrow">MICRODUCK / STAGE 12</div>
<h1>已验收的硬件在环轨迹回放</h1>
<p class="intro">本页播放 PC 原生 MuJoCo 与 Orin 实际闭环运行后保存的状态。每段完整 60 秒、50 帧/秒，逐一呈现 3000 个正式控制时刻。相机跟随主体；左侧观察双球与全身，右侧观察头部和上球。</p>
<div class="metrics"><div class="metric"><strong>6000 / 6000</strong><span>正式周期均按期完成</span></div><div class="metric"><strong>0</strong><span>超时周期</span></div><div class="metric"><strong>14.519 ms</strong><span>本次完整周期最大耗时</span></div><div class="metric"><strong>5 ms / 20 ms</strong><span>物理步长 / 控制周期</span></div></div>
<p class="detail">这是历史状态回放，当前不连接 Orin、不运行策略、不重新积分。画面的时间与耗时均来自已保存记录，不表示当前设备正在执行 HIL。初始状态另存 PNG；视频第一帧为 0.02 秒，最后一帧为 60.00 秒。验收首周期计入，无跳步、无期限平移。</p>
<div id="cases"></div>
<details><summary>查看证据范围与数据位置</summary><p class="detail">报告、CSV、NPZ 和 MJB 在读取前后均进行 SHA256 校验。只重建记录状态的几何位置以绘图；未重建接触力、BAM 内部状态或新的动态轨迹，也不把这些数据作为可续跑的完整检查点。本次 HIL 期限验收包含通信、推理对照、物理计算和检查，不包含离线视频编码。物理机器人不在交付范围内。</p><p class="detail">来源运行：<code>@@SOURCE@@</code></p><p><a href="report.json">回放生成报告</a></p></details>
<footer>固定 FP32 策略 · ROS2 Humble · Orin Nano Super · 记录回放。每条曲线保留全部 3000 个样本；ROS 耗时包含 Orin 推理等嵌套区间，不能把曲线简单相加。</footer>
</main><script>
const cases=@@DATA@@;
const series=[['release_to_finish_ms','#66dfc3'],['environment_ms','#82b4ff'],['ros_roundtrip_ms','#ffcc85']];
function build(c){
 const section=document.createElement('section');
 section.innerHTML=`<span class="status">20 ms 验收通过</span><h2>${c.case_id}</h2><p class="muted">3000 周期 · 超时 0 次 · 最大 ${c.statistics.max.toFixed(3)} ms · 均值 ${c.statistics.mean.toFixed(3)} ms · p99 ${c.statistics.p99.toFixed(3)} ms · ${c.perturbations} 次扰动</p><div class="views"><span>全身与脚下球 · 跟随视角</span><span>头部与上球 · 跟随特写</span></div><video controls preload="metadata" playsinline src="${c.video}" poster="${c.poster}"></video><div class="toolbar"><button type="button" data-move="-1">上一帧</button><button type="button" data-move="1">下一帧</button><label>播放速度 <select aria-label="播放速度"><option value="0.25">0.25×</option><option value="0.5">0.5×</option><option value="1" selected>1×</option><option value="2">2×</option></select></label><span class="clock"></span></div><div class="legend"><span class="total">完整周期</span><span class="physics">环境步</span><span class="ros">ROS 往返</span><span class="limit">20 ms 期限</span></div><canvas aria-label="逐周期耗时曲线；点击选择视频时刻"></canvas><p class="detail">点击曲线跳到相应控制周期。绿色线是从计划释放到该周期完成的总耗时，包含调度迟到。</p><p><a href="${c.video}" download>下载完整视频</a> · <a href="${c.poster}">初始状态</a></p>`;
 document.getElementById('cases').append(section);
 const video=section.querySelector('video'), canvas=section.querySelector('canvas'),ctx=canvas.getContext('2d'),clock=section.querySelector('.clock');
 const index=()=>Math.max(0,Math.min(c.frames-1,Math.floor(video.currentTime*50+1e-5)));
 function draw(){
  const box=canvas.getBoundingClientRect(),dpr=window.devicePixelRatio||1,w=box.width,h=box.height;
  if(canvas.width!==Math.round(w*dpr)||canvas.height!==Math.round(h*dpr)){canvas.width=Math.round(w*dpr);canvas.height=Math.round(h*dpr)}
  ctx.setTransform(dpr,0,0,dpr,0,0);ctx.clearRect(0,0,w,h);
  const left=48,right=w-12,top=22,bottom=h-32,X=i=>left+i/(c.frames-1)*(right-left),Y=v=>bottom-v/24*(bottom-top);
  ctx.font='12px system-ui';ctx.textAlign='right';ctx.fillStyle='#a9b8cb';ctx.fillText('ms',left-8,13);
  for(let v=0;v<=20;v+=5){ctx.strokeStyle='#2b3a50';ctx.lineWidth=1;ctx.beginPath();ctx.moveTo(left,Y(v));ctx.lineTo(right,Y(v));ctx.stroke();ctx.fillText(v,left-8,Y(v)+4)}
  ctx.textAlign='center';for(let s=0;s<=60;s+=10){const x=left+s/60*(right-left);ctx.fillText(s+'s',x,h-10)}
  ctx.strokeStyle='#ff8d9c';ctx.setLineDash([5,5]);ctx.beginPath();ctx.moveTo(left,Y(20));ctx.lineTo(right,Y(20));ctx.stroke();ctx.setLineDash([]);
  for(const [key,color] of series){ctx.strokeStyle=color;ctx.lineWidth=1;ctx.beginPath();c.timing.forEach((r,i)=>{if(i===0)ctx.moveTo(X(i),Y(r[key]));else ctx.lineTo(X(i),Y(r[key]))});ctx.stroke()}
  const i=index(),r=c.timing[i];ctx.strokeStyle='#fff';ctx.lineWidth=1;ctx.beginPath();ctx.moveTo(X(i),top);ctx.lineTo(X(i),bottom);ctx.stroke();
  clock.textContent=`记录时间 ${((i+1)*.02).toFixed(2)} s · 第 ${i+1} / ${c.frames} 周期 · ${r.release_to_finish_ms.toFixed(3)} ms`;
 }
 let raf=0;function animate(){draw();if(!video.paused&&!video.ended)raf=requestAnimationFrame(animate)}
 video.addEventListener('play',()=>{cancelAnimationFrame(raf);animate()});['timeupdate','seeked','loadedmetadata','pause','ended'].forEach(e=>video.addEventListener(e,draw));
 section.querySelector('select').addEventListener('change',e=>video.playbackRate=Number(e.target.value));
 section.querySelectorAll('[data-move]').forEach(b=>b.addEventListener('click',()=>{video.pause();video.currentTime=Math.max(0,Math.min(c.frames-1,index()+Number(b.dataset.move)))/50;draw()}));
 canvas.addEventListener('click',e=>{const box=canvas.getBoundingClientRect(),i=Math.round(Math.max(0,Math.min(1,(e.clientX-box.left-48)/(box.width-60)))*(c.frames-1));video.pause();video.currentTime=i/50;draw()});
 new ResizeObserver(draw).observe(canvas);draw();
}
cases.forEach(build);
</script></body></html>'''
