#!/usr/bin/env python3
"""Wait for the existing Stage12 campaign and collect a read-only final report."""
import argparse
import base64
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time
import traceback
import zlib

SSH_B85='c-oa$O^@3~620qJbS5xJArdtj!w%MlfH)onHsTo#M&lrhzz|wwQ|gH9X1bd*(#_5>NP=Lo*u&-nn}reNu-@EqiT_P9wsYFQuvN__MUFHRhY$$$S9Nv0s`r{ZJ3Ax62wP?e54j;+7^Y}w_$edf@jXIwGiSo^kQy#UhhU4O`LZiTvLc~OQz{}Iv|1-~22Ej<Oo*1@f`Khf^qeXdk?D%er8d1xD$~m~Qyn6tA(Uh^T+kUKQW2F45^=>sBh@Ns9ZDF~`b?@w6EH#)qft-8AlQRF!2nsXmByu7&!b+LaBviTi22N}2<KGHSQNB&c6M4k#mz|^3X_zLIo0!oPs^s%tyswv_BGf|T=x+)%PHDWQ~l4UnUW#Xy6iDLWv$k|(fEW6h^M49?|?f3jucEkp6&&jF+;;d=1~BV^IAuvW1M+o@9mq{Ta)qV?)yV*A4<VmljGsT@o*oS@6kkq*;d422*M{h#K@P64oP8Wh+X?ui@?KU8&bI_n)oQnH6zElK);4cN#&PooViIAaNn6RasdDj%FqM=;NFd!zHblp7V&O*@YY88=zF^zFcHd#`S5SfmJy#ZZT$8}xB7dJtT_xNSi=zdUeBbN*CsvUdui0)>jG@uB0;*>;HN!ujrdXrGiGug`Rz{OO}kw`AX*IAl53+&pUaJcbOGn_Q*daoE8pJTEf#769@y~&;q!~$GLqO8j3@WU`}dEb5oj|(18tjQMH`$Ug8`{|<_3$E2c!LAV<pVB+jlDO@bMlBP2-}f%0j8a@0CuOTnX!JiI7a9Nl{Pbai<heh5YKxWy*TVvX}Bu$tVvO-ALxsgmtGhp(12Gstk{zn02o?&;{3LtiRWbx!{I%jp8)vp7vc0b@z7nZuWQke}OLV%B--iyhU7cynb55a4M#*OpMphh{9OFHD)tl1B9^d(yKJ#Vo^^O7i3#yRerb?wFCj{Yxv)2!jGJD8TxNH^e^AgwOtT;#M54hb^^TXBi!q%#uid3F5j?OaD#<Nt=6OVNVu?(^0k;%S&yXCaNizn`>}5|@_EAp+ll*Dychtd`yY7#r`I7~H-mSTBiJ!^9rnQ3F~WJ_oP`VcgIV>cSwj~dZ6YoPwf**o9~?e>@SsCNnt^w0B6E}FW^kf%*0J;q9fSUE`(hkbQ2fEHwe@Q!uw}?HL;TU$vS+7$w3EWvaG603N9{|HDPVOXP)2Z;NGL?&2){!d)t7*mLI9W}R2OkCXsgH+Hx*-twBpp(Z3lWyn21<(8&;vZf_7Wrb6pBIloO1(`9uOWcpIT{d)V%ZuK?z{cG$r)4qWfAr$Azt>X!h(UOSK(6TU~G)Tt)ew?3EE<simRA735U^8hxV3_o0ZWy@he8rx{Ep5GfqR6c8k#ldQb0Z`3}P?&hYI`uZjt4(0Wg+1_jliV?&^K*<Y6@(^uby?M~QUKCvzmC({oF|NMduKw+!4{)OOmP|Uu)1}@{fA|+NE54vT+9fMzzpU&a3Ee~7^Wu-Y}&h+eQ;nnO^0BqG$wC`r$$Vb5?kKDj_sV=tlG=SsYx#v^m&7+t`g`gX#=Gg0t2B39O`n6x_g^siop%_6yfl;`}QWm+patlu-@HuV+sJ`#gvR7$?HZfodx8`S-r#{K%Wbg14;xtBZ8)|zt9FUhsc;jEMXSlTqu^<1VU$Zusy;UT*c5xqaJ-vqrxRDxE7(bVxk22nh1cfy?NLSa>w|VTYmrHA20s%{j2{zef86m|2_Zu{Hrft{_^bQXFtDq{N&fiPtKow^WxdR&;R!M`Pbk5_WWPxPrrEi_a9$8d-~h+f4=(Z+g~4l3Va?Xxt?1p-rVT*HRHf`AKG{9ZK}xDfOuVrQ|}g#98id>8mySM>+3^MV;T~=82bZLSi2{sq}*7W8Y9f=4wmtK{F*wYi5<VnTy4!=2^aPQx*S|zn>@_ZDO02P$bAtVL2YM75Bil%7HM`#bep14sx6|L+E6r@h!(N~H#Ry=xSq3UOBHOP(siy(#`o?V5BDeI;dp$1bT~L3-r4`_ZNylEBS3+Z^2=_Lxr6L($&U;AQ%EUQvr}KOR2nvMJm_e~2}7l|EytQlE?0MpnS&Tb84kT*+9^o46h9bBrtmdtE1AGYnuZz!mP64jqT~!lT+=LLRB1A0=8VC`xH~GsZz-viR<>kejv)=myVQj91H4)Qx12%%;EPtQoV_v@J@}d>MLYt_FZi8YeVD_?b|uuXI!4q`TSAk=RRU5FF7!b?Y=7q3wWT<OypXVNR&|tS`D{jc`|@pNg9+RrO-vuP{tt~Y4T}'
SSH_SHA='9dca22cb55ecee4eacdc49f273197c41b43c8d13c00f9c588e3ae7d418569ac8'
REMOTE_B85='c-oa3Yg5}u@;kpm$$gNN2!mlEfkazHh?7lac@&Vl-J@Kpu{5@4(d#r4;7#SfUw6+)8d)+M_yE#OPrs($;^oVi*Ck6MRz%yem=&y8c~-$$w4J5#%Hk<J%(sgwwZbfpd02`peP^YaB}G!jtjr2aBzabpa`f`$%PsY6Ety;bw)*uinWgn}!ph~=yvPzOhbI8i=dDZJ(9g1DrI4iv<<{2ZkITvF&E#Wnd2;&o<nzSxEjx-1qcI=z!=pE|kg@$=-bS-1I@*h(x8a-F{w`<xW5$od{k?;uqy7E2<K4rz2S<#*IXrlCu(!|bt+SIqg0qW{li!06f8I>40qNl{ug1H3zr1?2r#2@yS6@!Ahqkx2B0eW21u|iA?4Ts>@s<TY_~@f3YLV8&928d4)`&lXT+-3d_*3w*DpKn_OL+}Vkf;O4g+GB5fAe1t4-WPZjZ;iWZvC6Z6`vGERyg(*54p%oi$8{(M-mvezKRdWmi=0mh2loBbB~97ypvI!u}IS1h{Tz#N)F6**cMWxz)Ko(hfaEt2ut^8I1!Q%u+++Ce}U-?@S=q$#9FU;k%e5!fW_jTJGn?Tty+sB-K@K^SQ(qygYVE)*gJ^Sc6X4vNK3?eYv&@er>?tWqo!?FX`tkfsP#hTu_ztewY{->JD!fzySq*qXxxsEroK-Y0Z~!fQ;;~Ko!zO@U}`z`e{IkH!&V6Jia<Y~M9j~#@>5o&5vi^x*e4cCZXR(it_jln%2(>4RlN5ljrb!yZZI#UV41R9F0;~+aaLa>J}R@NMOvE!d(!{@L%o2br!3)07!s66BQzqAYgR=IUe<52%6V}wWLDHG;Znp6FvJqPjzs~Ntkw)}iRLX5n)-%=AyYfJsyG8>17mk0jgGt4qK@OnYg^%-y2LszY1SL=YQ}8$hgLDVra#NFSR?AnIkf<h-Y_MecHBi_X(dAAs-$n9Uj(NoXO|~mKA#7lF20?Av=z}9&argm+@rhP6cs0a?~z#)b40FzKG{ck_3ixp%lT&$5$K8#k>UJ2rt6!F%gaf_lH`4m%d3mi$@O(`eRFa%3BH|wJ-_(RxotWz#sx=%t3Mu@6*?VRm8UF<etY`jxV}1cAn7$!Tk8cIz)a+$YCIN2JPIx^FM=wM7{n@nSFh=G1lcK&CC_oAhpaaIKmgG)OGN0Co;+5;8ypUUJbU0pz$ncD4p+O(Juk$36@*J3-sKra5EKX#B}KY!gfarA-tUD>Lt%tk@JO@5(S3dUOh~}8uv&^WEgeESk#5E$*P)ISy+cmco|P2cWEO*qz-<6oQfC{@+N`CiKy-w++O*BI1iF)}Bo^tN<Dx_8<k3ey%2~mI?-xGthp<f+1SDt$W*Qt$C8w?YVAWYwN)d6ZTyj)LR=~}hDvAWGLP@ehVgS!9&8pqXG2`GBU~X-SHAO@Yo!$2esLs9b{q<@A1AG_8_Z!Vo5(U*gLpH=<MHyEK6HGB&*7uE~)F~ak!HPv9N<*MXEVwLvU2=`&lI<NF8d5!oa;dbQ<vexlVrIJ*lh(Y;AI6GsS*3RvFGR@;CuYekV#jkzjE*jTTyJLEYofJWjFekCgs5eK%>mygl{;NUdhKgcii8Ic7TP6|X6_5$A%x9|=B+u<tRVa|?jc;VbOAQ?0C|DK4WkR9^oF`2HdHM3wY*FIsGhnqDWDgvO{iP3r_bQ!`nMA}0*al_c>&jjC9`Drz+E>IYz!zAG0C%=qRS)O^w%`5nDZ6W{jiBYVV2}Ehg8@EUd_Y^(x7Aawr$tTvsi?yVfgNz?Yd9bI5(@DuU9){!*JpmQMKTC;lL5s)6CRN)nPv27{7ugWN-)~&nfE%0-DJVyJsS1vzTjsz^0&bfTAGNG^M49+U`?LK(8@*9=WqFEwfcP6CaH5@{Z~pD8;#iVu{!G4>c_ogzsi7@}mkW;!x#aT`uxCf^9p-9BLn{zd@BUNM#ajBdiL{2*JEi2^I>lztGeHmi`6{p6M&-gLcf)F8Bj<tZ*>lEDiLVXY(8iyNCxdPZ#A<RRT58;GU}dej-Gwh=4BDbz#UZ+*ctIA02Gd-nbnQZ6~<hkDD_%dAAM@EIJ78u6KRUx$Fz@0W_MZF)>8B;b!ZfPFYd^^cqIutq1mGzldp!UUk4=NzTL~4RXj^+=iHLLr8*J-#*?Df6`K<74L1T5aJW++krSV{RuV|tOpiIY^vAyyThFHWA>L6sEbISv(hS=f&CBaM}r7z9OW>9*0Emu<NnP<Q~UjnuZ~+~%B(6vuKOIm^3*#LQ7$=xY@R^f|6>SQow2nZwJMg>dW9_3npKxZtIF5ZCPeu`5qJ!RHUwYetX>Kvlx=B*`2k9Q37%~N>7`9*Q*w35p-D3iwX2ZrgP1|AEV4L;T5%H|S%uUy9(3^3@j-iw?S22@s-216(cu#s(GohAU_EBFXzPTwrfYet)C;X`_Rs;SGoR7S4<x^n66nmTlCSspZ66TTHiNZs0atKKbnm9hI?yPf^92WSaNSkjfvImUQZN(e(LL6&#qf_u&zXbcm*gQX6~Bo}dd;Z@kF1h}6%cluMUhpx!6aOWPXTK#3b-R}a0GR`JH55lf<5)<3On1|<IaAzSQv!}21`Y&ReEPy@k+S?4kp?Q>MBpQ#-J=UdZ6<K9wx--xmwbKEP^HxZa1`mcy#!+9bme#r+z!c9N~(JhSEiB>XAC_c6en5C_v=<Cd2*9MuG-Jkg!J}3cr7vP=h7Uq#%>7(REdfvQ&xl`0Fky<`EU6X{?tYfR2_(Z=K;0*c+bHcG-%fAV=W8U_PvhOUFSO#vTsPnoD=L$L`L~-odVSJ9S4ahcrzl?CSiTMp&BMu3H~b)_0iJr0O|{y1z0jA#M8YNzEatPd~5eZ#er>xlNfM7_iV9EBKIxfgB2zIE%a{;5P@_6wnV~TGjw`(ZK69?_rt6Jb>QcS5hfgQr!5O19YZA(fXuxBn+UB4@$^;DYQcYTPzB`AhINJ9X21hSS&H~);3h2MC-*#KPjfrsX;YT((J}GdDXxx<}Gv}D}##0qEN4|(!)uBUe|zB*1dxU)|ftnb~qXgP9rqCbq=tQ%aL6-0Qwt8fZi^b5G9QfDG&zh^t;nj+WD7t&XEOd!LfjV+?eWukwaw(I}cfL2Y(V&5wIxY7@A89E(%pTsw;j)fu@dS1zz`}9nAIWd<DCCcH0@(V(b!Nb$^qRo2hJ;&c(tU%f@)ErMWVL_wD-bx0ONY{eEht6Eex}__kut!~{rE37xFDrb@nTEGm1G;940pEwrcqI1{en`J;dWK=pMPxxVKk1Ufc9*QmIV8`%(%jTUh>a|prqE6tC4yZ!cf`lCZZS1Eyd|9KjUV-(zWH7$*5`&CC_gH-GO5JG&nOKq_3Xn!Rnu`WQ>mHO-2>|M7ChO>>Zn+;*wVbdHsPO%vz8usyq398!Gr75@ok~uafDVl#DH+1XJpjgl1gFM{Yc(jdQign<P!uN(FnrmbH*&T{MpV|ibhs9E4XZ@S#e0IQ41I>6od|};BEE-s;8KOb|O#f2{;3j7t#^3jIlZH}Y(2QaJLC;DBHMH$fvQSzGvkHI%nu+?3QI*a%kDjQLqZ#c+7~jt~?hP~zh|E#kDk<G3FmI-+P^;HKU1q=^m#o71$dGhU3OykgE1|Q?hpBf?4z@w2Y|qu}V5|M!hxI;B({6xPmDZ~Ez^!)lpnZlSxc-i$FG*)a*Xz$ApJ2D0*{<{CZ&sjZC~i=fE!wOQ3z4!oi1hR)sLHTDW3J&e6zUHupA^L!N-qy$%~^QI7CaDGxQKG4d;qvY2iC{eK=A(8|6jtu4BU|00%|PK^cNJL*1C#@=HvS8;_C#ETwg=|LAP$e4ch%dOG^8H>+Uu4L3@%s+c*8YJ-^9CHmn<0T^F1tZJK(w!3gv7<66jZ+ty|<R<&PGJ>z4wsuJ694}meDvZ_eXpYxtlZ>~<h;5Xag!?%y0CpW?A#o6WWlbeZB@LJTQHB&#dut5b+y7;=guUlXH$(Vw{v+T=Ih*4+<dc@CBZ|b{j1oo%ffJR|;xgW+%B+a<KL6|_P9!og+kYbV80O)mKHkem&j4ck@a|u|{u;Q%3FTxQwK69=Y*S`G;ZsPhk5aeodd2w|Ee@;IB+4-Wai|ppVl2&pn%Xy(%6?hx5VsU@FH{HU1s@`b-<wKLgN}Jk>x?UOjkD2-(u39t&|6e5vn$W5NY}<L%l*d!mr8YeIHBHsl_1hWa<W^o_8DgJ(9Pb567wY4_Z@;ElZ&W3@)H7V3$_lzbCPN|WkLGclJp?IB)jQ<2{{!C5c4h'
REMOTE_SHA='fa1fbc92ca6940a02f7508a35bb01fff9b1ad1ca9139e73920dc7ac2b6c3b489'
SLOT='/root/autodl-tmp/microduck-double-balance/artifacts/double-balance-stage12/native-training-v2'


def decode(value, digest):
    source=zlib.decompress(base64.b85decode(value))
    if hashlib.sha256(source).hexdigest()!=digest:
        raise RuntimeError('Embedded read-only source hash mismatch')
    return source


def request(ssh, operation):
    source=decode(REMOTE_B85,REMOTE_SHA).decode()
    command=' '.join(map(shlex.quote,['python3','-B','-c',source,SLOT,operation]))
    r=subprocess.run(ssh['argv'](command),capture_output=True,timeout=120)
    output=r.stdout.decode(errors='replace');err=r.stderr.decode(errors='replace')
    rows=[line.split('=',1)[1] for line in output.splitlines() if line.startswith('MD12_FINISH_STATE=')]
    if r.returncode or not rows:
        raise RuntimeError('只读连接返回不完整；云端任务状态未确认。'+err[-1500:]+output[-1500:])
    return json.loads(rows[-1])


def save_report(value):
    value=dict(value,local_return_utc=dt.datetime.now(dt.timezone.utc).isoformat())
    folder=Path.home()/'microduck-double-balance/artifacts/double-balance-stage12/native-training-v2-reports'
    folder.mkdir(parents=True,exist_ok=True)
    path=folder/('microduck-stage12-finish-readonly-'+dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')+'-'+str(os.getpid())+'.json')
    data=(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode()
    with path.open('xb') as f:
        f.write(data);f.flush();os.fsync(f.fileno())
    print('FINISH_READONLY_REPORT_SAVED='+str(path),flush=True)
    print('报告 %.1f KiB；SHA256=%s'%(len(data)/1024,hashlib.sha256(data).hexdigest()),flush=True)
    return path


def show(value):
    b=value.get('files',{}).get('budget.json') or {}
    p=value.get('files',{}).get('run/progress.json') or {}
    maximum=b.get('max_updates',6000)
    count=b.get('completed_updates',0)
    text='只读监看 %s；完成 %s/%s（%.1f%%）；阶段 %s。'%(value['status'],count,maximum,100*count/maximum if maximum else 0,p.get('phase','未知'))
    if 'joint_geometric_current_fraction' in p:
        text+=' 当前双平衡几何稳定 %.1f%%。'%(100*p['joint_geometric_current_fraction'])
    print(text,flush=True)


def run(watch, ssh):
    while True:
        value=request(ssh,'status');show(value)
        if value['status']=='STOPPED':
            print('后台任务已退出；正在读取训练曲线并核对最后/训练期最佳检查点SHA256。',flush=True)
            value=request(ssh,'collect')
            if value['status']=='RUNNING' and watch:
                print('检测到任务重新处于运行状态，继续只读等待。',flush=True)
            else:
                save_report(value)
                summary=value.get('summary',{})
                print('训练结束记录确认：%s；可用模型字节核对：%s。'%(summary.get('finished_training_run_confirmed'),summary.get('available_checkpoint_bytes_verified')),flush=True)
                print('只回传这个小JSON。模型和完整日志仍保留云端；尚未执行独立策略评估。',flush=True)
                return value
        elif not watch or value['status']!='RUNNING':
            save_report(value)
            return value
        time.sleep(20)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--watch',action='store_true',help='只读等待当前任务结束，然后自动收尾')
    mode.add_argument('--collect',action='store_true',help='只查一次；若任务已结束，收集完整收尾报告')
    args=parser.parse_args()
    print('ssh -p 52763 root@connect.bjb2.seetacloud.com',flush=True)
    print('只读收尾：不停止/启动/续训，不新增短测、PPO或仿真，不操作云机电源。',flush=True)
    helper=decode(SSH_B85,SSH_SHA)
    ssh={'__name__':'shared_ssh'}
    exec(compile(helper,'<shared_ssh>','exec'),ssh)
    try:
        run(args.watch,ssh)
    except KeyboardInterrupt:
        print('\n只退出本地收尾监控；云端训练不受影响。用同文件 --watch 接回。',flush=True)
    except Exception as exc:
        save_report(dict(status='READONLY_CLIENT_ERROR_PRESERVE',error=repr(exc),
                         cloud_training_state='UNKNOWN',added_PPO_updates=0,added_simulation_steps=0,
                         traceback=traceback.format_exc()[-4000:]))
        print('未发送任何训练操作。重新连接：python3 '+shlex.quote(str(Path(__file__).resolve()))+' --watch',flush=True)
        return 2
    return 0


if __name__=='__main__':
    raise SystemExit(main())
