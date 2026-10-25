#!/usr/bin/env python3
"""Retrieve the selected Stage12 model and completed evidence, without simulation."""
import argparse
import base64
import datetime as dt
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import time
import zlib

SSH_B85='c-oa$O^@3~620qJbS5xJArdtj!w%MlfH)onHsTo#M&lrhzz|wwQ|gH9X1bd*(#_5>NP=Lo*u&-nn}reNu-@EqiT_P9wsYFQuvN__MUFHRhY$$$S9Nv0s`r{ZJ3Ax62wP?e54j;+7^Y}w_$edf@jXIwGiSo^kQy#UhhU4O`LZiTvLc~OQz{}Iv|1-~22Ej<Oo*1@f`Khf^qeXdk?D%er8d1xD$~m~Qyn6tA(Uh^T+kUKQW2F45^=>sBh@Ns9ZDF~`b?@w6EH#)qft-8AlQRF!2nsXmByu7&!b+LaBviTi22N}2<KGHSQNB&c6M4k#mz|^3X_zLIo0!oPs^s%tyswv_BGf|T=x+)%PHDWQ~l4UnUW#Xy6iDLWv$k|(fEW6h^M49?|?f3jucEkp6&&jF+;;d=1~BV^IAuvW1M+o@9mq{Ta)qV?)yV*A4<VmljGsT@o*oS@6kkq*;d422*M{h#K@P64oP8Wh+X?ui@?KU8&bI_n)oQnH6zElK);4cN#&PooViIAaNn6RasdDj%FqM=;NFd!zHblp7V&O*@YY88=zF^zFcHd#`S5SfmJy#ZZT$8}xB7dJtT_xNSi=zdUeBbN*CsvUdui0)>jG@uB0;*>;HN!ujrdXrGiGug`Rz{OO}kw`AX*IAl53+&pUaJcbOGn_Q*daoE8pJTEf#769@y~&;q!~$GLqO8j3@WU`}dEb5oj|(18tjQMH`$Ug8`{|<_3$E2c!LAV<pVB+jlDO@bMlBP2-}f%0j8a@0CuOTnX!JiI7a9Nl{Pbai<heh5YKxWy*TVvX}Bu$tVvO-ALxsgmtGhp(12Gstk{zn02o?&;{3LtiRWbx!{I%jp8)vp7vc0b@z7nZuWQke}OLV%B--iyhU7cynb55a4M#*OpMphh{9OFHD)tl1B9^d(yKJ#Vo^^O7i3#yRerb?wFCj{Yxv)2!jGJD8TxNH^e^AgwOtT;#M54hb^^TXBi!q%#uid3F5j?OaD#<Nt=6OVNVu?(^0k;%S&yXCaNizn`>}5|@_EAp+ll*Dychtd`yY7#r`I7~H-mSTBiJ!^9rnQ3F~WJ_oP`VcgIV>cSwj~dZ6YoPwf**o9~?e>@SsCNnt^w0B6E}FW^kf%*0J;q9fSUE`(hkbQ2fEHwe@Q!uw}?HL;TU$vS+7$w3EWvaG603N9{|HDPVOXP)2Z;NGL?&2){!d)t7*mLI9W}R2OkCXsgH+Hx*-twBpp(Z3lWyn21<(8&;vZf_7Wrb6pBIloO1(`9uOWcpIT{d)V%ZuK?z{cG$r)4qWfAr$Azt>X!h(UOSK(6TU~G)Tt)ew?3EE<simRA735U^8hxV3_o0ZWy@he8rx{Ep5GfqR6c8k#ldQb0Z`3}P?&hYI`uZjt4(0Wg+1_jliV?&^K*<Y6@(^uby?M~QUKCvzmC({oF|NMduKw+!4{)OOmP|Uu)1}@{fA|+NE54vT+9fMzzpU&a3Ee~7^Wu-Y}&h+eQ;nnO^0BqG$wC`r$$Vb5?kKDj_sV=tlG=SsYx#v^m&7+t`g`gX#=Gg0t2B39O`n6x_g^siop%_6yfl;`}QWm+patlu-@HuV+sJ`#gvR7$?HZfodx8`S-r#{K%Wbg14;xtBZ8)|zt9FUhsc;jEMXSlTqu^<1VU$Zusy;UT*c5xqaJ-vqrxRDxE7(bVxk22nh1cfy?NLSa>w|VTYmrHA20s%{j2{zef86m|2_Zu{Hrft{_^bQXFtDq{N&fiPtKow^WxdR&;R!M`Pbk5_WWPxPrrEi_a9$8d-~h+f4=(Z+g~4l3Va?Xxt?1p-rVT*HRHf`AKG{9ZK}xDfOuVrQ|}g#98id>8mySM>+3^MV;T~=82bZLSi2{sq}*7W8Y9f=4wmtK{F*wYi5<VnTy4!=2^aPQx*S|zn>@_ZDO02P$bAtVL2YM75Bil%7HM`#bep14sx6|L+E6r@h!(N~H#Ry=xSq3UOBHOP(siy(#`o?V5BDeI;dp$1bT~L3-r4`_ZNylEBS3+Z^2=_Lxr6L($&U;AQ%EUQvr}KOR2nvMJm_e~2}7l|EytQlE?0MpnS&Tb84kT*+9^o46h9bBrtmdtE1AGYnuZz!mP64jqT~!lT+=LLRB1A0=8VC`xH~GsZz-viR<>kejv)=myVQj91H4)Qx12%%;EPtQoV_v@J@}d>MLYt_FZi8YeVD_?b|uuXI!4q`TSAk=RRU5FF7!b?Y=7q3wWT<OypXVNR&|tS`D{jc`|@pNg9+RrO-vuP{tt~Y4T}'
SSH_SHA='9dca22cb55ecee4eacdc49f273197c41b43c8d13c00f9c588e3ae7d418569ac8'
REMOTE_B85='c-oy;ZExE)5dQ98Ap|HSt$v{>OO}%8L$PKr*xD3n`(fw;fucxTvn&ZDl_o3t-*+Ts%TBU&n?Kmn-SN>q&mE8Q+1c3*Bc;QudsaHM-E}ona7BcwIH5|;owuzc*sfI!f@&roq+&bgv#1o4PAi-r-+gqdUDpbTJv%#_*lTA)<fg8Q{p}yJZ6;i_JEtQ`JBBsRJGk8k0rq5abAA0G(^k)2h_+P=(yO+tXKL3ib`=$E+0*T;Z2O{Svx3y5p=@zTcX8^NNk!Hyz^*^};pbnk-e+IjU0br+J$JsimK9vO4N=t{n<+u6rfSx+JG>Bmv!EMBw_RH`fZKN^Q7lJj86h<9lsk3Yx}`)zx!(#b9zZMz3=j%q6eBS0$bJ5dS#Ai9VvSjH6k>!4<tRXLf{HjSL%dosnygYDhYXj?1SKpg1DwQh6s9;NVX))@jtGs3`xW5UqV2mJVB!Gb;Ca^Uj?_I|w$1D=SeO+vo%?LRY_5xU!gpNqkMYMGF-liKgik203btYqCV0iuAjL6`2~UcYW4@w1T~dZ(nx-sdNgQGz%$F#N_>u*@1nvcYN)i7*iugZL#Q&ONRPv+@Qi5oNk~l5OGL8s}2@MjQhUGF!N)iL5WgMbq!3xG!A!c}qij-kiQbW<Q$Q6;>dH3+`-t7q8LfbDMtvjZtZO{1kRFWi!AF&%uBefz<QCg&w^CVuDlv9>q6rvcDRfH29tpZ9gT`?Y&aa>?RqJ*M|E_Iosh}i1MwimEXyQ+~^(|=9w$;v16o{QMCiIyQsA2Hn9Mv5q*t0X{4!f3H9p!gw;(hvx9PM{y+py0q|#Yq{Z2y@C;X%V3yib_J#IECtcoqYe($G5*^&@{{M^w(rkGVU~tm0r7@?wFK7#y_7p@TtJjDyQ9^17FFUo4!%ij$H}S3U4230m^F4r1JDIoz|>bt4;Q{ZT3K$Y#6_D%f<7J`E#W<X8z9Gjy0Yu3fFgtbhurmKOnu8Ix}%FK9%RgPcExJnNH9?%&DZKpXaa|^0c-C#^(*36C=_Hu0JpwNyg^_4jpe-HJ*`YdI<LY3!_cZwza2Y{i*x0VfP)Q`dGTU{=xN6@HTThaBc9#0ce&%R@OMN#8fXD2;Z>#LA*gq-88U%%N_>xOagH0RE=X!4V}WaX+JlIW6=TkkO-WEyUc`r0^{w4)-BVA*g|_g?VI^_gf_}nGS|nc2L)(KQ}(@CKJ4nM*$y6K6->7E9RzE?VD+$XTTee{^nzAzDk*`zKA8JawnN$)ZOB^}^tmFBd7?{wf0<bx<4(CfDaip_2k$lEMwa=SDR0<vzHS*!Q@72z!8aKU=m+HQ4@ap1bm)dOYgW1^RWxmHk~J07CHII}-wg$X?iOZrXWIA)-R_eb9hqA-cdSvZfO&e<{j#rB_T}rr!a+zV>r`iX=!dz!lb(Ot*@vXEqmH@&WqML)NBi}(e~l7D(4fUIe?W|DQLU?n)T7Ane|iNz*#4X|Az?(l)K57<v?NS3!egoOjJBE1Y-*u1`@E@b%}gp`6B}_l0R`~%(43e3u9Kbxol1C3%K=$tA4Jc5z29^}#-2GSti|%&PB3`0bf@r!)Hhq#Y-ZVa`fU$d=MX$p7l3WW0=tc#sA;RqxnwHS37TMhJIoApnOWbZCF}m&z#(3h!-}ytkK16#@ci$oA8BZT_9`CFVC(VyQ%PtG6MW1x95XeZ%gjtav(GKS%fWt{zY)(7dLy{sn&f;g8QXe8J{QO1)?oW%tqub+%ZzL4zTfEA@I$ldRr_i2AYnkj%e9&pz3#oa`3K;6?wi*E&R@U!@XC2nIxpn!jr(Fmo<2H<jPZB``~I9Y6kf?a|K&KWpFWM(z1--D4du)8hU{3LL-q1p@BKU<>ZioRxfuo;z;@F4t%YRwly%CvGC#UMK`pR7uDhjkbM@-=_1j;6%dc*3u5U8;<>Pi>`bf?M>x2g&z+$S0c{bz?-l<90?vio5m{a||`3H4%Unvj!lfMA`CQ29'
REMOTE_SHA='69a693d09f5f0c1c1d73e928176af4cfb8b4d465a220ac5fef488d7a81ec590b'
FILES={'model': {'remote': 'native-training-v2/run/checkpoints/update_004500.pt', 'local': 'checkpoints/update_004500.pt', 'bytes': 10036060, 'max_bytes': 10036060, 'sha256': 'df03202acf010680b63d3277ec879f63e2d4808e5d1286653923a314f125ac6a'}, 'group_20261021': {'remote': 'native-evaluation-v1/group-4500-20261021.json', 'local': 'evaluation/group-4500-20261021.json', 'max_bytes': 2097152, 'sha256': '7be7e52a27f9192626af8b9f2f7cf94ce06c99e3e86323e2f4055f4e1fdd48bf'}, 'group_20261022': {'remote': 'native-evaluation-v1/group-4500-20261022.json', 'local': 'evaluation/group-4500-20261022.json', 'max_bytes': 2097152, 'sha256': '5df8d19a0c50869bdd65a06ac18293d458da6f4e46304bebee732e240b9e2edc'}, 'native_task.py': {'remote': 'native-training-v2/package/native_task.py', 'local': 'task/native_task.py', 'bytes': 8816, 'max_bytes': 8816, 'sha256': 'b6909b9cff864dcfce8203062a75282571ca2c7ef5d66b2aa58c05c4626a55af'}, 'source-pins.json': {'remote': 'native-training-v2/package/source-pins.json', 'local': 'task/source-pins.json', 'bytes': 54309, 'max_bytes': 54309, 'sha256': '08c781088ecb4b2023c593ff8ffa45061bf6a57fad5902fcf79b50155daa969c'}}
CHUNK=1024*1024

def need(ok,message):
    if not ok:raise RuntimeError(message)

def decode(value,digest):
    data=zlib.decompress(base64.b85decode(value))
    need(hashlib.sha256(data).hexdigest()==digest,'Embedded source identity differs')
    return data

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for data in iter(lambda:f.read(CHUNK),b''):h.update(data)
    return h.hexdigest()

def request(ssh,args):
    source=decode(REMOTE_B85,REMOTE_SHA).decode()
    command=' '.join(map(shlex.quote,['python3','-B','-c',source,*map(str,args)]))
    p=subprocess.run(ssh['argv'](command),capture_output=True,timeout=180)
    need(p.returncode==0,'只读取回失败；已保留断点。'+p.stderr.decode(errors='replace')[-5000:])
    if args[0]!='chunk':return json.loads(p.stdout)
    header,separator,data=p.stdout.partition(b'\n');parts=header.split()
    need(separator and len(parts)==3 and parts[0]==b'MD12_DATA','Chunk response header is incomplete')
    need(len(data)==int(parts[1])==int(args[3]) and hashlib.sha256(data).hexdigest()==parts[2].decode(),
         'Chunk size or SHA256 mismatch; this chunk was not written')
    return data

def local_files(folder):
    found={}
    for key,item in FILES.items():
        path=folder/item['local']
        if not path.exists():continue
        need(path.is_file() and not path.is_symlink(),'Unexpected local artifact: '+str(path))
        need(sha(path)==item['sha256'],'Existing local artifact differs; preserved without overwrite: '+str(path))
        found[key]=dict(path=str(path),bytes=path.stat().st_size,sha256=item['sha256'])
    return found

def download(folder,ssh):
    found=local_files(folder)
    if len(found)==len(FILES):
        print('本机已有全部文件且SHA256一致：100%；无需SSH。',flush=True)
        return found
    print('只读读取4500模型和已完成证据；不启动仿真或PPO。',flush=True)
    inventory=request(ssh,['inventory'])
    need(set(inventory)==set(FILES),'Remote artifact inventory differs')
    for key,item in inventory.items():
        need(item['sha256']==FILES[key]['sha256'] and 0<item['bytes']<=FILES[key]['max_bytes'],'Remote artifact identity differs')
    total=sum(item['bytes'] for item in inventory.values());completed=sum(inventory[k]['bytes'] for k in found)
    started=time.monotonic()
    for key,item in FILES.items():
        if key in found:continue
        target=folder/item['local'];target.parent.mkdir(parents=True,exist_ok=True)
        partial=target.with_name(target.name+'.partial');size=inventory[key]['bytes']
        need(not partial.is_symlink(),'Unexpected partial symlink')
        if partial.exists():
            need(partial.is_file() and partial.stat().st_size<=size,'Unexpected partial file')
            offset=partial.stat().st_size;prefix=request(ssh,['prefix',key,offset])
            need(prefix['bytes']==offset and prefix['sha256']==sha(partial),'Partial prefix differs; preserved without truncation')
            print('续传 %s：%.1f%%。'%(key,100*offset/size),flush=True)
        else:offset=0
        mode='ab' if partial.exists() else 'xb'
        with partial.open(mode) as f:
            while offset<size:
                data=request(ssh,['chunk',key,offset,min(CHUNK,size-offset)])
                f.write(data);f.flush();os.fsync(f.fileno());offset+=len(data)
                print('取回 %.1f/%.1f MiB（%.1f%%）；%s %.1f%%；耗时 %.1f 秒。'%(
                    (completed+offset)/CHUNK,total/CHUNK,100*(completed+offset)/total,
                    key,100*offset/size,time.monotonic()-started),flush=True)
        need(sha(partial)==item['sha256'],'Completed file SHA256 mismatch; partial preserved')
        # Publish only verified bytes. link() refuses to overwrite an existing target.
        os.link(partial,target);partial.unlink()
        found[key]=dict(path=str(target),bytes=size,sha256=item['sha256']);completed+=size
    return found

def local_replay_inventory(folder):
    project=Path.home()/'microduck-double-balance/workspace'
    pins=json.loads((folder/'task/source-pins.json').read_bytes())
    matched=[];missing=[];different=[]
    for rel,pin in pins.items():
        path=Path(rel);need(not path.is_absolute() and '..' not in path.parts,'Invalid frozen source relative path')
        actual=project/path
        if not actual.is_file():missing.append(rel)
        elif actual.stat().st_size==pin['bytes'] and sha(actual)==pin['sha256']:matched.append(rel)
        else:different.append(rel)
    runtime=project/'.venv/bin/python';packages={}
    if runtime.is_file():
        probe="import importlib.metadata as m,json,sys; packages={}; names=['mjlab','mujoco','mujoco-warp','warp-lang','rsl-rl-lib','torch'];\nfor name in names:\n try: packages[name]=m.version(name)\n except m.PackageNotFoundError: packages[name]=None\nprint(json.dumps(dict(python=sys.version,executable=sys.executable,packages=packages)))"
        try:
            p=subprocess.run([str(runtime),'-B','-c',probe],capture_output=True,timeout=30)
            packages=json.loads(p.stdout) if p.returncode==0 else dict(error=p.stderr.decode(errors='replace')[-2000:])
        except Exception as exc:packages=dict(error=repr(exc))
    return dict(project=str(project),frozen_source_count=len(pins),matching_files=len(matched),
        missing_count=len(missing),different_count=len(different),missing=missing,different=different,
        existing_python=str(runtime) if runtime.is_file() else None,runtime_metadata=packages,
        modifications=0,dependency_installations=0,simulation_steps=0,PPO_updates=0,
        meaning='Read-only preparation inventory, not an inference or physical validation result')

def save_report(folder,value):
    reports=folder/'reports';reports.mkdir(exist_ok=True)
    value=dict(value,created_utc=dt.datetime.now(dt.timezone.utc).isoformat(),added_PPO_updates=0,
               added_simulation_steps=0,remote_writes=0,selected_model='4500',power_action='NONE')
    path=reports/('microduck-stage12-fetch-4500-'+dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')+'-'+str(os.getpid())+'.json')
    data=(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode()
    with path.open('xb') as f:f.write(data);f.flush();os.fsync(f.fileno())
    print('SELECTED_MODEL_REPORT_SAVED='+str(path),flush=True)
    print('报告 %.1f KiB；SHA256=%s'%(len(data)/1024,hashlib.sha256(data).hexdigest()),flush=True)

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--download',action='store_true',required=True);ap.parse_args()
    folder=Path.home()/'microduck-double-balance/artifacts/double-balance-stage12/native-selected-4500'
    folder.mkdir(parents=True,exist_ok=True);need(not folder.is_symlink(),'Unexpected destination symlink')
    print('候选4500：212/256成功；取回约10 MiB；不计算GPU、不改云端文件。',flush=True)
    ssh={'__name__':'shared_ssh'}
    exec(compile(decode(SSH_B85,SSH_SHA),'<shared_ssh>','exec'),ssh)
    with (folder/'download.lock').open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            print('已有本机下载正在运行，请接回原终端。',flush=True);return 2
        try:
            files=download(folder,ssh)
            print('取回100%，正在只读核对本机源码与依赖；不导入Torch、不运行仿真。',flush=True)
            inventory=local_replay_inventory(folder)
            save_report(folder,dict(status='SELECTED_MODEL_RETURN_VERIFIED',files=files,local_replay_inventory=inventory))
            print('4500模型已核对并保存在：'+files['model']['path'],flush=True)
            print('请只回传上述小JSON；不上传模型。本次取回完成，如无其他云任务可关闭5090并保留存储。',flush=True)
            return 0
        except (Exception,KeyboardInterrupt) as exc:
            save_report(folder,dict(status='SELECTED_MODEL_RETURN_PRESERVED',error=repr(exc)))
            print('具体错误：'+repr(exc)+'；已保留下载字节，重复同一命令续传。',flush=True)
            return 2

if __name__=='__main__':raise SystemExit(main())
