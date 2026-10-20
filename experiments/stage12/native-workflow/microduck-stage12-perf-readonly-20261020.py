#!/usr/bin/env python3
"""Read existing training timing logs over SSH. Never starts or alters training."""
import base64
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import zlib

SSH_B85 = 'c-oa$O^@3~620qJbS5xJArdtj!w%MlfH)onHsTo#M&lrhzz|wwQ|gH9X1bd*(#_5>NP=Lo*u&-nn}reNu-@EqiT_P9wsYFQuvN__MUFHRhY$$$S9Nv0s`r{ZJ3Ax62wP?e54j;+7^Y}w_$edf@jXIwGiSo^kQy#UhhU4O`LZiTvLc~OQz{}Iv|1-~22Ej<Oo*1@f`Khf^qeXdk?D%er8d1xD$~m~Qyn6tA(Uh^T+kUKQW2F45^=>sBh@Ns9ZDF~`b?@w6EH#)qft-8AlQRF!2nsXmByu7&!b+LaBviTi22N}2<KGHSQNB&c6M4k#mz|^3X_zLIo0!oPs^s%tyswv_BGf|T=x+)%PHDWQ~l4UnUW#Xy6iDLWv$k|(fEW6h^M49?|?f3jucEkp6&&jF+;;d=1~BV^IAuvW1M+o@9mq{Ta)qV?)yV*A4<VmljGsT@o*oS@6kkq*;d422*M{h#K@P64oP8Wh+X?ui@?KU8&bI_n)oQnH6zElK);4cN#&PooViIAaNn6RasdDj%FqM=;NFd!zHblp7V&O*@YY88=zF^zFcHd#`S5SfmJy#ZZT$8}xB7dJtT_xNSi=zdUeBbN*CsvUdui0)>jG@uB0;*>;HN!ujrdXrGiGug`Rz{OO}kw`AX*IAl53+&pUaJcbOGn_Q*daoE8pJTEf#769@y~&;q!~$GLqO8j3@WU`}dEb5oj|(18tjQMH`$Ug8`{|<_3$E2c!LAV<pVB+jlDO@bMlBP2-}f%0j8a@0CuOTnX!JiI7a9Nl{Pbai<heh5YKxWy*TVvX}Bu$tVvO-ALxsgmtGhp(12Gstk{zn02o?&;{3LtiRWbx!{I%jp8)vp7vc0b@z7nZuWQke}OLV%B--iyhU7cynb55a4M#*OpMphh{9OFHD)tl1B9^d(yKJ#Vo^^O7i3#yRerb?wFCj{Yxv)2!jGJD8TxNH^e^AgwOtT;#M54hb^^TXBi!q%#uid3F5j?OaD#<Nt=6OVNVu?(^0k;%S&yXCaNizn`>}5|@_EAp+ll*Dychtd`yY7#r`I7~H-mSTBiJ!^9rnQ3F~WJ_oP`VcgIV>cSwj~dZ6YoPwf**o9~?e>@SsCNnt^w0B6E}FW^kf%*0J;q9fSUE`(hkbQ2fEHwe@Q!uw}?HL;TU$vS+7$w3EWvaG603N9{|HDPVOXP)2Z;NGL?&2){!d)t7*mLI9W}R2OkCXsgH+Hx*-twBpp(Z3lWyn21<(8&;vZf_7Wrb6pBIloO1(`9uOWcpIT{d)V%ZuK?z{cG$r)4qWfAr$Azt>X!h(UOSK(6TU~G)Tt)ew?3EE<simRA735U^8hxV3_o0ZWy@he8rx{Ep5GfqR6c8k#ldQb0Z`3}P?&hYI`uZjt4(0Wg+1_jliV?&^K*<Y6@(^uby?M~QUKCvzmC({oF|NMduKw+!4{)OOmP|Uu)1}@{fA|+NE54vT+9fMzzpU&a3Ee~7^Wu-Y}&h+eQ;nnO^0BqG$wC`r$$Vb5?kKDj_sV=tlG=SsYx#v^m&7+t`g`gX#=Gg0t2B39O`n6x_g^siop%_6yfl;`}QWm+patlu-@HuV+sJ`#gvR7$?HZfodx8`S-r#{K%Wbg14;xtBZ8)|zt9FUhsc;jEMXSlTqu^<1VU$Zusy;UT*c5xqaJ-vqrxRDxE7(bVxk22nh1cfy?NLSa>w|VTYmrHA20s%{j2{zef86m|2_Zu{Hrft{_^bQXFtDq{N&fiPtKow^WxdR&;R!M`Pbk5_WWPxPrrEi_a9$8d-~h+f4=(Z+g~4l3Va?Xxt?1p-rVT*HRHf`AKG{9ZK}xDfOuVrQ|}g#98id>8mySM>+3^MV;T~=82bZLSi2{sq}*7W8Y9f=4wmtK{F*wYi5<VnTy4!=2^aPQx*S|zn>@_ZDO02P$bAtVL2YM75Bil%7HM`#bep14sx6|L+E6r@h!(N~H#Ry=xSq3UOBHOP(siy(#`o?V5BDeI;dp$1bT~L3-r4`_ZNylEBS3+Z^2=_Lxr6L($&U;AQ%EUQvr}KOR2nvMJm_e~2}7l|EytQlE?0MpnS&Tb84kT*+9^o46h9bBrtmdtE1AGYnuZz!mP64jqT~!lT+=LLRB1A0=8VC`xH~GsZz-viR<>kejv)=myVQj91H4)Qx12%%;EPtQoV_v@J@}d>MLYt_FZi8YeVD_?b|uuXI!4q`TSAk=RRU5FF7!b?Y=7q3wWT<OypXVNR&|tS`D{jc`|@pNg9+RrO-vuP{tt~Y4T}'
SSH_SHA = '9dca22cb55ecee4eacdc49f273197c41b43c8d13c00f9c588e3ae7d418569ac8'
REMOTE_B85 = 'c-oy<TW{kw7Jm1yAhHjV5?hWlZGpP0ot<q`V9;C8Nf+C71c8=ln;VJ}NGh>AnEmZNhoo+{r!%_&64T`2xqj!uTwGlI!e~L*o@>M9mb^1`%N93e!v)i%lHw4PpDLnx*$8SXMR-|P%D|B#IW23-x01XlZkUcQE-ohSnEXdqvR#+d>?Rvkm86E1;A_u=+~a{?E7q>Gp$3)7^<?t!=IsyJKYn|-dq)zoxW4&<T-uMxWKysVF_eoCnNEq|B{xY=A1x;YeqOlQSvi(0imRH*Fi`6tB2<%&Bf=jh3lOo1HDgcV^_1L1-4Q4RY9(OD9#4>R=H&G>{-P!AcMEE$<*LBPp}ROB;T!jDd3*ssVqFVv!gcT{gDF`=tJxy8bc&fqNk>z#yee48l&X|Y0>x@Ua~4G2Kj=ywK2y;!?H%I2fQ;3iY#@ev!ezS&`f$x9+>$v9_frybX{KZ&DrzEd80e3&jJe)$2@wzXQQ8_JE92Sp)_6^Ssw5j}&T|uLzz!fR5fBfr;FOe%%A}8fjNiM{j&1u>vVtGe<IaE|s982;0-u9Tx_6rpPQTgutENh`OjO&@u)S$P_3GOD4mZJbip9EFhbnm7FV>H%)%EOkdbuLAIsE<m?7u05D}Z05c5z;dDu2@crj4o!eqdBdOgQkP0s|`fsIS7u;wpO7mvi{F>k%7Oz1YAxpJVVi22WABW#Iy}1y?#zVCAc*kHfku4sg+(t@zbV+8uRpc>kftGH{l(WWbTGnLrt<Fvu$*SdNNj8c-(-jdp(9?GnRFKpuc55Z<;SVP~%+OP8Z?0<7|#cMuTfJM`{4AXi?6E24rWIt85lb$S)PUOvWd8NH4)P)@Cp(TFLa((+xNAjes~LL*bmPXu^qX;+K1X95fc@FYn>ZpOSYA~CI@rWBz?br;M8GAC>uo`P2{FFULaVu>HCnT6TX1trG_tJOJor#3Rm#vJZB`r6Z<`1~xjsSFixLpNoZN8nam)X_<HDwZ>8vd$cxAde|I)hcl{CXWz*V-@^SeI|7Wd^U=@MYX}q+p;3dtFmqkOPrQ-r!)k^Ett^E)|6AWod%y^TK_Lh0kGDn9C}9*+_jBTHBvG~UZ@Zj(@K8>T{IU>!H5;9=J}50Pjv-QAk{NdJ7@ym*h2a3ueWc=6H}52&tAvDxCoWZC@5*;LbUKneOd_wmkdLV<gUixjt&P^z^KC*y%vzb277f3DNyJTFZK8bb69Oi_)jFf!yYpw4~N<<qcJmxIJl)@!{iAE-NXu;T6DQiVmIO~Gf-Om9Ff&Ks-ZYwrUhVCNkw33kNLQ3Owd|GWLv~XM-A&Fk{@S<;1J9Jv!;f|$}BVB6-6IUczgX5*{qP)I8qc!6-MfJUWYEJ?6t)*9mvY7Mq;a3ug<8J!W2Qz9`uoOu5lcW{p9bTY!5@{6!;stwoYr=wcn}pnr2!<lDUAbkM@onhYGfy^X%qU$Z7GxlxnrgYNot>zMNQ2XG+QomqqoGZMf2=y<t6jot}3hSZU+-C#mmw3VOs6MzQ~!r)7qw^t?Xj+PDhc^38@R1}N5K&CCm9vLE4(0t>#)D^+L&G?phjbV#V|&&)PtELQl5fgMz=mr6bLMG2CohJu}GQNip6e|&s#2`IA@w%zv8)Q!M>H|oJY1W!2vz&q5z3||(>gTk7iCFr}Wpxx*Uo){(1lkX^aV(K0t$@>peLW=?{y1)Oqb#`)n>LcRG{A0y-DZ^@{g{Ojk%0Mriw;lUvG8}d=CBtCd6oBFwD<TdpP~+#ds<v=Xk2VFWFs9RG^~7e*a({5DU|{95qPwXZ0QfW4mFl+g9T%QpkVM$$(SLJPc9^L(jN=(PDs*+VjuBj86eG;|?~MF4Azyv@<*Tp833<{D`natE>-Qk-2BblUsO?)OTOSN|mvN0f*Drp4=eh+xx8VF^GGXj9fHMk9c%GpI_oe}dy#Ek&iogR$g}}zi1>|CJ6@dHDf@_YE2%>l^s&yC`MY)_^&Riz}{kK6B4c{wxzgOt<xE--MVd3A2xIcv!e@Fx^*99f@auk^oM}}x}?0C4A;X^e*V=-l5<1CdW7(7?-A&j(+*bfgaWGp`Yo&}Z&TK(JI4nxZmX`W;tv71d$s{i68|ApMYy(PSW>5-d*t0$UJ#cbBkFza0hd9HW?AZ%R8qFAI3xqu!6`36-@lef3=WT=MUXHF6ST*2&p;@H+Pd;4VS@@SMoK$IL~+;%NwaTWNS8zvW`1g?Ez)H%r|DCQt3v{f;{3X%uW9K#8`z!F{rX*9=LU;thg?KrmqGL(BB29bO6FH-+r9Ou~?m#ol)w5_-xA?Zay^%VTswwu-B>lB-{J+!q5yovF^v+LB_YhC$o^x(gO=yVwL2|Xp>vx8r~l?B_M8wx*HQ8zlEvQew-phWxGz%iS2|0X)<Nv-gi75sR+xXJGCe)%qY_weT7E(xw|!77@v*5SBw9bgDi+h=^{O_|h<Vj0yr=f2~><N^I&Qt9PKlRp61P4or'
REMOTE_SHA = 'c6d31c20a5fd6ce53549f21a53b4eeb996c3713183ffa5e0ba0deb4daf552bab'
SLOT = '/root/autodl-tmp/microduck-double-balance/artifacts/double-balance-stage12/native-training-v2'


def main():
    print('只读统计现有训练日志；不停止、不重启任务，不新增仿真或PPO。', flush=True)
    helper = zlib.decompress(base64.b85decode(SSH_B85))
    remote = zlib.decompress(base64.b85decode(REMOTE_B85))
    if hashlib.sha256(helper).hexdigest() != SSH_SHA or hashlib.sha256(remote).hexdigest() != REMOTE_SHA:
        raise RuntimeError('Embedded read-only code hash differs')
    ssh = {'__name__': 'shared_ssh'}
    exec(compile(helper, '<shared_ssh>', 'exec'), ssh)
    command = ' '.join(map(shlex.quote, ['python3', '-B', '-c', remote.decode(), SLOT]))
    response = subprocess.run(ssh['argv'](command), capture_output=True, timeout=60)
    prefix = 'MD12_PERF_STATE='
    rows = [s[len(prefix):] for s in response.stdout.decode(errors='replace').splitlines() if s.startswith(prefix)]
    if response.returncode or not rows:
        raise RuntimeError('只读查询未完成；未发出任何训练操作。\n' + response.stderr.decode(errors='replace')[-2000:])
    result = json.loads(rows[-1])
    result['local_return_utc'] = dt.datetime.now(dt.timezone.utc).isoformat()
    folder = Path.home()/'microduck-double-balance/artifacts/double-balance-stage12/native-training-v2-reports'
    folder.mkdir(parents=True, exist_ok=True)
    target = folder/('microduck-stage12-perf-readonly-' + dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ') + '-' + str(os.getpid()) + '.json')
    data = (json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+'\n').encode()
    with target.open('xb') as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    log = result.get('log', {})
    for field, label in [('collection_seconds', '采样'), ('learning_seconds', 'PPO更新')]:
        item = log.get(field)
        if item:
            print('%s：最近%d轮中位 %.3f 秒/轮。' % (label, item['samples'], item['median']), flush=True)
    wall = result.get('metrics', {}).get('wall_seconds_per_update')
    if wall:
        print('含日志及保存开销的更新间隔：中位 %.3f 秒，均值 %.3f 秒。' % (wall['median'], wall['mean']), flush=True)
    if not log.get('recent_iterations'):
        print('日志中暂未提取到完整计时块；请以报告中的状态为准。', flush=True)
    print('PERF_READONLY_REPORT_SAVED=' + str(target), flush=True)
    print('报告 %.1f KiB；SHA256=%s' % (len(data)/1024, hashlib.sha256(data).hexdigest()), flush=True)
    print('训练继续按原配置和预算运行。只需回传这个小JSON。', flush=True)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('\n已结束只读查询。')
        raise SystemExit(130)
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(2)
