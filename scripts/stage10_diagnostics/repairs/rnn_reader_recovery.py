#!/usr/bin/env python3
"""Retain the reviewed 100-step trace with its missing postcheck; run only jobs 2/3.

Repairs live precision getter lookup without changing backend settings or frozen
task code. Before simulation, a separate process tests namespace shadowing with
zero environment construction and zero physics. Original failed result is kept.
"""
from __future__ import annotations

import argparse
import base64
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import uuid
import zlib

REPO = Path('/home/lx/microduck-double-balance/workspace')
ART = REPO.parent / 'artifacts/double-balance-stage10'
FOLDER = ART / 'substep/20261001T071355Z-6050d852'
PACKAGE = Path('/home/lx/下载/microduck-stage10-substep-diagnostics')
DRIVER = 'microduck-stage10-substep.py'
ATTEMPT = '01-substep/attempt-1f54cd0b'
PAYLOAD_SHA = 'a28b471a6e5414409b9dc93b20148a67e20d8992098dc671ececfc1778bcc051'
PAYLOAD_B85 = 'c-qvxiE`UWwjlZ|xbCO{Sw&JHxXaCskZo1jQ?{jN$z9V|Rv@q>B$T*ag0!vj)qMLqOJ*Vy3nhDcBHrtnTNZ)D$?fFX@+{w;trtnKTdx=Sv*T~imRY=AC;RxWoA1IK(f7LfKFW7u)7#uXJAO8ahDkc}y;&0a{bA%sGjB8+M}v3_Po~4kBpv!9PQ-LJ9>%GdP2;#fO@`@k7*8j&bn>k8Ou`M~^;R&@nIHPeIGlOYIP}JYnLqHx{r)JN^hXJNJMt#+XgZt3Q92oja2zM_+Vf`P!Dy1C$z)^z-EG5ILLJBBQQRMo!*G<0hoeah@cqP(haw3RU%=O+Xq1X+EaF)-jmOi;bl?jw3De<pI2FPiYPi|$)o90)D4t9bF$+aMnv9Y#8l>adFiOK@;zgr=zwh;9Kk>q}?|aE;+#gPp*<?10eJ=_}lNrF>id1aHDi+=CeznV%qW5RMUIA=xoTg$lfMpX?A4VL_#1Q`H$D?F08-xQdp2cZ68^Zbz`r#y;z`#R)I37=u;V_K)&mKC@R^rQ(w=*8h#B?$qd!ZjigK#qP$4NXKPLlp;6c5KyKOIL=C<gw-o59p3(=dsXLF5lTZ|Wtpw01kkvDcqO(Xc=A(?K}$r?8nGEa)IihRLiy84ss3;d`Ta5{`z`NfeI#@u2TdMz|N_(Xf6yqc|2xGK)k!oq5AaJnpBkXVYol7qdPr%WxK^Q9KyMu&bkZHi+VI>_tB8LOL4@;r;)4JJW0-p0b^P`;Vi2zCDVv)sa|zb~g9B+x2SjY4zzD{=W<S#OZrZ6z<|%=f$zJyWI-mq+5~SI-Bh}7I`l6j??$24t$?&*Q=#i?cj;m^PSugVSL+h{u0}@bAEo-W2m22>2|$zf*{@Fq6UGJEjR1!&Iwnm^)B3H>s1aTSgEJmn@zaQh5RgDZ|>Cxi4Yt7Eng+!E{qo;4oQQ|lPum9?*%Tl{#^d<IQZYc)+^Nqzh7igZz*;mf>ob<7TX-grw6yr<tLkPcMBh=cIWU@H_C4p;;Zp%zsnZI&-+~eO~@OC@tuCM-)D*b@mIFNeKpp`JW4QE2mh4=vR=vNTk&7}Yzy<<iZJOoU$)r}e&n}d$BA!6d?(=Heivg0uTMXmI&+#wD<E_Vf_4w~eZBZBTJ0We32gMI)y3)CUk|i{25(-SzC8H=ZCoUn>2Tl=25B^ky)h8^k>~lNeiVAqU@#hp5m3`<e=-x^G)j{J9G(zP3XlSzF>&mETK&(n7Z(8P%JupRk=>5tdcO5*pnqL|T3w!<y~S@@?xC}Q-<1fH`ACr!tC=PW__Y2=ySM@`&(3izzq#-xcaNO`G&fHRJc0uR599S}w_Ps+98(_P;lWp4=hMnj|8qCD_jwlQ<u)D;OaOpHKZGL)50M&dHtS%&L3GEyY<S4a#U{(wi3k>Ab+fw-BpmMBap6E^FlU$=eeAfOvqY@n?_a}3E*=nDKCKdwI><wU+c3Wsd8>WQooofH(aQO@3YX#-QLROob>!fKwv(>6^p}&Z9NH)Y+~;tS;h2NXB3!jn7^v7Pu=og;6!BQT0gx^})yvMk6T-Lf#g&@_k}tzRo?-Xena`a*P@Tl#_eACRv0K8B?hD-GKVZjhBr(KK8oF8Ux3LIrVQdVH@ae_L#i`L3@mqdvEgST5LEj(YV+U3mkZE@h=QCX0h@@#^>y2PwjBG_ef373h4IUkU{wfP^R_lD1#ZI(OZUoS<twR9HeMpiZ`=DQ&b++2M*BzeH`G+^}PCopr)V$b^Y?bY@aDk@{Fz+Z4pL?5~D?fho{>|l^legdVG3{)}krbtEAs7y_KplAmgCtAksmxnpyxWJ1jw8NqM7%>%w}CU8eVx0m!m2EFvOHTcvg;3!gS2T$jh6)!dVO3RDWHA(#SV$ob@34K{7&2>n&&`?VWzp*$sXFye>?a~foz-52-$(S0z4kB^NZv5T=*}6OVv{WM-d;{U&4EA25c33=I$Ih6!}o~TI#cyNiW=N#42fhbJI*L5;*rb9;6Ed4u10=J;0?F07ev6K;fnOvv^(YsPsP|)If4pHvuv-0HJ!SpAOBu=<%U#!^hXna}!^8IJahexq=R_3#5@haRu6KbI%D08K{MqSJyadIaD}>3J?$b%2-s|$Lg}yuja#b71<T7XzkWp6uZKGlI;Q-s$77-ITuK4B>{ApP@&xLf=Jp4%6Gz9tiuEbNKk-AWD7v@Y!xp698kYs;yo~{o&xv^Si?4p_9(SET3Tv}-c8T(d$V4@4W|as8{`Ez$BsbW0n6K=P2AmvJMO`G^Gbq5Zjdv({&NvVNOqU|KiBcP_^bOR+!CY5f4gw-Z-@csi|%&O1tClkFikv>_sU5lSNrz!&ypX>hpj3zqG3D$SXu`72&Tyk+Y1y20ZHef2Akb@pM>xQx0Y>t_^__=&1SFLVjDas0Z_XPSO^ZH<>7%^?Uw<{-XXDNZjB9%=dHSv<)SnR7!Ds1)s;Vkd=+l;+x3p8RQshNq2QYFg01tOd`{=NvTl!_ZP?d$AOGjs%d_COx43KP|GIpA_Wt$B#p{dH(^vT9<;NFqPcDK#&OZEpaenghG<bRT{?|9Z@vB!S!P}FM?_a(SUc5bf`TLvqzq##)2X*4)*;Rsplr#>AW<(&=iJYrR>h@Ro``W?_g;TOmgJ58<HDpA6P|y(BFG=66;*fR#Ns-y@afjV^Z!X@QT)uowM9=Jn$sNCIB&7urDY~yCBk<IK4KIve&A<s~V3Nhj*D_Ucq~yVAKL&=5zwv$jIctUzua`huKz`MO2$-(nDpmk><buRNjdyEB_YN$iJ}_dp1M-&#Qb-D-`+$TNB?s4w+KL<H`{gp+A|-O41ONUL<fs<Tbi4j?C0W3A8(Dj?H*;k2Td|Ri^zLE1=3@rOg!jJmxkQL-KCSQwk=fYol3a<@*~A6M{lK_=Bo{=!%^gxUHUGd))m%%Xgo&s7#RA3pJ>YR3eugs#gP>1fDGYXHi*^cy3y$|?y}c7#5=`aMS?_n7{jT71|CG7|acUU8Vs$WZmb9Hnq^KDuA{SebGqb;txgq-lac8^6u`NL*T1c)W<<hyM`>fH0%?0ESdRZQ9#Uk8gpGB}+w^~GmPd;2CSyDQxp)Mn-|AQu{0Od~TXfurbqE`3d6_8_*|DeDjM~*YMn<$D#DmA{wJ(o?yS2*0PjeK0?VJhU3<8r|<ARCQV6L3m!z_RpSp1fAuQqpSmIBp@kTizuZY$uH?pI>hG0;weoAz0td>2&GIk{x4ik;)Gr-@ivjfC@CweK;e*eiy?V_%BVtRLG7CZLD(Wr5<(6N>qdHT++bhv%++++$_XSXsS@@qtK%Q(1!idSPe;PQ%I6QgaUd~dM_2GDnw{}7v%dGjd1~NVj%E{g-q#rF}2F>zCXJRKAiqPA5SkXPhSPUo_z?sz`qDyy*c^q{n^Fko0nBu=z<C<wy?v2fFmF8Xw6|QkrFuh{VKi%aUlB(Q1!R-kAx`C;UIz3M@ADA{{Rh8aoX-tn1GXo=7kJ#sOLVI%#hK6SvHC1Qsd)oS2z%U0R%v6<=jytgFmG;2L9r1=gxaF=;0^3?Y*G_C}SJ)217$bIz}QVN;ukp_%2+*&Te18h9n;rr>oDkUu+ku{fB(<VewB9L(4z5s5xzGKfpc9;oRRK6V>)il)QuOB8bzQj{LD-tzhBfMP3UAEs1$=tw)>emz&@|TrTQb2ipGCOV;~nA%ZAepfVc}_Ifh~Wwj308zA<ur(4GI9|9MU>aKvIezVAKZg+BKQse(I$P&X*=aArAHr8IAT>Kuqc?D!dw}i8~O0$~|X=xy&0VFts>LCp$VAV&idg&nKB3uO0D#23&zVc3h3>2S9H0&s99HiJ#6u;pgQ1pLnvWd3X=zZ2FlFZ#5=Pp}`CCoAoK(NW8Elv1GdYjsthPL$WE<gi*P?!NrfJ^pJ_3cak*+D+KU!YKY1I(l0TS+ST%S-(62U3J><<ypEK?eyr{%ZYSArRf6*Z;IK6?17PQ`pyv7wdf@nTlnc11kCJcKw%F0e+%*s6TODpPsx@@{ZfC3<*dZie|M3G|$v+f^(5A_lwfTD4|X=&itvGi!w6iZpp?3gaUbbt3tLqM*p1}Dpj^QAuv$qZ`UNFwOnOehr=V#hUY>9P`?xW6BSX$NWQ#r+cgFkG~CQ_d?1da=n(|klEwX0Kau7h!$_sp>9}41R=X{C1XWZik-%T!4unkTza#r=w@rJDg3Jb<!)mPi2CaaL1#q%Nplu9x=w7Xs)DN6ILrv5YjM4gKq9&rnm0LF7ptZKqreknOZqX&eWht~KaHYPmv`e55*#VjXAX);wDZ_z75@)1Z$p$3FSD<|dduRFm4cVIn-e{5F5F=t8A$dLqY2qGqoS2pd&6$c-IV2l{xB)|qvIJiDj1gqJ+Zx8!Dx+#4o^>1|g_1P@%YZqXoU7I@Ua%VyV@tM3g`Wk)d0F;5-)5vd6=2I|c|Fqf2IZhKe`~pk_*`yFypD)Vh4r=ginL|O9xcf|Q5`A9Q3tZek}-UI1Jq{j?!+pG#a}aF*l!3Z=~0##ul}dS6$3TUP`J$THoAvzP@m-i!ob?)^(0ZSLhG>UoMb4rqs_@8ckB_Wtq@CbNOP1A%f@1oDy!771wFx9EV4}y5-R3%nm=s6@eH)<()dV@P<T_wa(aPV0x7e)#%>WzSacj&$5ErM1=>T9wJ3;InRq9)sAaebkev}mHUO*<eOgiia(LcNIG~6G?XZMEwBZ|&x?`L1YZ5HSA1+{j__~0pg?UHPLVLw}%PJj6T(R4JsP7su8XB@fN<G)vV~h1UcIuX_UMZ7@w9YmfCLvu+yk%24YR_zkLc{)782jShiSAJlKzX8OfuA$6+HBXiS(KrwR@co!gxeKfgrMyl`YV>MXT7TXmh3A5%1}%l<nCx7@qUZycAx?2rp?e`j!U$!!O-DvU_gM*1>EpUuckpAv_hiQV77oMtXSrFCAoqHG=zNFDo2|g3Zu_)7HAQXH|BZm@&Rv>lO#&6`#9TXH`xkBy!9fB?|UTEIAOX&vkT0Eez$TVlKHgn_1m?;3vpP(mEt219Y?7Zx9dgXc+b`SCd^47apzGxEL7D12XQ!~O#uYC*($w0Og>|OTCKM~EFBP@8;cUS-zcPID=AI}{I*r4r*&J1K;TUP!fg4uR|QaVCW!WFkMxJS&dMuYHwD?K^iC>bFoYVtjL9`I!j98)rCcV1Wf?`a1}qZUMhh$Qap_|af7vL}D+|M|1{naSkGD5uY(Pf@e@b)&pD)9&C7%pzG4ecW5k@q@9&JUkbGlB?^V`nfoxbSy{d&<=K?bhaOF(2>^0~r(ZJ^4yCE*%?8-!wo^~QA(A6F2DcnXf30;(qx@lkRm0$8I}(Qp^pXENoG{T{esZmk!v%-#J)%=vY+UN0)FzPdlGFynWI;fNLJR#}Z_EO`8;+LUEbMuFqHTgU6gT;g7Zugo=9aSW^E+)zSGHpjWw{;pxvS_0VY{NeQctlh9K=91tN@d=`Bs5!PvTCO};vgCy{1D4BNSM|vLR$1l@1>ZPu^tK|LXtSQQC-BIWe8ME6Q}i2!u(Sk8$`$%<pk96`6gZ^;DhU9k3G`Pg23WN1%C;4~i`_Lqf)&WrSn{cthXUO!ZxoHbQzrUOEz4Nmo>+p`#>zcua_G_~ZyV+wEHhDsa3Q5iSY-JZ9_Ua@n=A**AFw5J_`ik@GghI?)RELQKGp@ZLh-UyXH!TGk^(EVMLnt2?>PN_yTL-Bx>sM#)%S-D3V1bwWTvG!B#=yv{YdfUvzEL20#bq7?tzlZ*Et}|lmDwF%~0bq-W=>Sh{F6%>>^k*W^z3gEe>{L`0ce1uLkcvpCzT@Tl-(Xmzr9$`h&M)ELJo(Rdd^cZb%+h_+bx@*@mrgEpmkH4{=K8EWvOips&DM$b+LvAwD=OlJ06Jg;z~LF!K2~-IjV}4;<|9p`j!Ccq`BhhAau$RL2RV9Du#32_9YyX)JE3)*TBlU0~gkali1~1B6|`Ie#(+(bx)PIhBkVsTtb$@(g~E$M$o)bOqTHi5SUOOtR_D7#OVh@?A&^OTLu1h1O}o=Ph>?t{9cA;}X7ijL(|7t2azzr%yr(<!H+{%m!NYtHe&B$rHG+MkKj+s1+&naA^<F#e(sK9B-MC2HW))c%gY1{LHVq{p*LuwPxOBUqoIPmteiprA^((manli=pCSh{|CH9-VKvb*eDKLBS%WmpmCtU>J{Q_ow717xlEhF>P7h^Y0Z))QMX}uAi}n+{b|d79G7LsLXt`{zDgHgLH$IRH)#(5L7T+hrY_jU927)YB@*|?eY$?v{z-fksR0kW4C%12Q0oNAUH^b!0s}0|Inp=MeTUW{=ZnC*lSX1~U5pv<duVBBwXa;$P>aet%Gcz1tVQbPHttChE~P@xv=b0?5#Bc~{i7xkGSnI!;|uEOQYz|B(e&_yTTM0`>_XZU;oTZG?RM#2UzKb?*ymgM#G%g`yR+^0%>SNl|37dCPy+SDD*$T^ng%y-ytsgQzmT3s$`wmrwIus3E=O^y8<YqTnbvoCFJ7#3QKmzD`f6YCkd&#L{Z8iY#p|;Vm%-%+_E3Cx{`U0pH284(&o`%kkb}jW59rQ#h-p~ldu|cHj$@H$VoOL1<)s_yc8lT@`SdW00-3$^nSP=K_drH;mD(w><PdZTAlz&g_vO#wStbsD4k)zVqIHzX-`u2y9cRDF{%bGzjyMRn+weZ0BSGd?(E`QI!{Ea&fdoRI2hlyM*8Kf$y+bc+4h|`{fqou@%2w}!T0$2RBb7P?R7g@L`&Ah|8$GDR*H~<J4qk1X(w{8X<MZS4U0oSLe?58g_ViU1B^|NduD5g2@&QiUx{p29?s)jhyz5T2DxT%U-|&*+xSn7<Sug?WnM)f%iWb?0<)`zME+0AC07H8rH9%-M`D!jGoaeXFrwpe;US&UgwFv{#lEQ(7e43GEEe+Wgc)Y`EhK#C7Wv*fY-{EE2;SuTYlpkiJ+(1V|N7b<Z1NvPqvPk(&OUsCS+GC@O8kF+5?y$EgN;K?^i4POACsV8m+1YR}dI`ua@olT!i#L1t?`pr?SjKMdUY!^O+7qjo%nX43k^AYBhvy&!;y|RLRdPv{-`|R_@^Y(P6Sl^f(6U`qo4sBbRglC7g->(I?Xk8^%QA2&9;=zGbd=<<wkL5dhqOI)zlK^LtT|KPt=Q+Ha?eTPxazJHi&i3`RS8_;oc5e7%|pR0`(jcLp4D}Q>=?>Ilh#wCEHo$yWfh@3!0H|1b@_hZ8cNB_vuIrJ=!rLD7+039kkIJRmXZNo2O?n~1x{{2bOMwMcsEP2l-k=u-%jyPB!~yf4!~kGA;ELlFFuYKj18=vEc@tI<xtqU^tvF=YYc<i&~e@vv2W5jdV2$`yXF4uf<A$fZ0IeG5Ly&%S7N=0BO5yZz(inBIhqPv5<j~x2k32_|Cl?E1t4A*G=>I613j+#jB$@Jru=@n$X0jcOZ-8kZ~`3sEfQ%H;XNr>=SEQrAl+HYbgmeb95F8yu?f=!R(;aJ*f1QK42jtEOh0nDfn*y3{fFflyxhjM0cr&Q1o(n%NAf-!ZP}alLuuaXDANxW>Zb-F-`#B2dz4A(Ulc_D5)|Ipq5tz9Sx$5TFOREcv7Ns=!>RK#97lin^UpplH81W?EJ+X&Nlxepi@(h3mRmAF8YZCwjr127!9Oh6CDz8%Q*)<Z?V1+W#EpATda1rPzvSNM&PoTDQq$wor?ox9axFK|mr7C4TdlveRF;Ash#<I>+Z3O)+@JpS)AFa}@~787z5D6nKV6C=>Eh~O_JAJ#X9!CV%<AfRd~G`3=SGw&vE2PK!dn$yLG-(WC#ctomcjwcvMvKtHEyR0xxdl)AG)|*-x%t#=O+)nzp{;6a>|lf4SN5XZGOeWFQZGpQlRj1gSX#!si2#^IS*c){`&Uh^7K{FVgaOUG4HogI1WMagmY5k&`5JCZ-QoG{8^9BA#8X3k*o17#b^*4zY&>db;t~(*SHKV-eoULT~+Q%5pio001S+&oM-jWA_CR%JQ#A7jQx>)nY#&E^sePzuyDcyhVgPB5ylYFXaM=s#Xi3+D>8LoU%WoSJ5LQjvAP-%b)TOjnV+j;P9``WG!niF1j+hjlY;l52jQNiMiviboJqm`PBfE~Eu3%%Gzau9DVpSfcPljE0b8Z`5(l)fZkL_udItH!<G{9a^o?WgCSi4dJ|gCM#7Zp@{)J_f1Nz{8dZuI;wtIYf<|>lfL+eVg6JK|gt1vlpTnlR{(5O!C#G5unYnB8gpki&HjGjrlYX@$sH0`#vvM5X~&|i$!V$P0hO4>j#9`>@cHGwpS)qE^?NJUB=4h}N*pNNSSU`2l1-9nD*!Gk-u-2(x21@m)>THrHdJ?&5t+F2ZJFU58({7*+d!T8hQ^nW@3ZT^1$!c?oI_Ksg=tD84xr>J>X6f01H9T{Cm4O;FAVBIZHR2}%m9_ht;lV4FTqqT8P78;rBU!j63j?HLerfaD)*Zp`b+d22b{#=zXHWdtBHrE{p7%MnWoj_Z6EQ)8QYp)c(k*6mj4!6pfDMufWL}y)AD6e1nK{*YDxn7*%r<Y{;U9it#n1%m^48CjkQp}_DkGWHbC72cFJun7%^jr4Ap$5CPx{qVcfsQrNk-W6$d;ptvTav2(JNryyE560kihKWNMf)iG21R8n@Jga$gnv9ru)=UFE}W=4A&)RTRWV$=@S!HsTw^M)$qu0xt#01%ckACRlTFL&bgS8A?dhSCS_)-kME_#HrP2N|9qAAdeF7d&*3vvN%x<HnQ*)1HB?YRH)~pd3DTT~YQ6DPRP%YAZE9V3{cJR`T69FAJWC5xRGHtkSY>GT}Wd5JyO7b7d*K@3r_1FQa4V_r-&DB%zXuxR$*uzn%#e3-=S+tXcu?cQ{P%YUHf}71gxP>*Vq^Yv9!-T7SxCr<Q#Dir0g*^#49{B9nUpZ&Qhj*v1f*1b^K3<%Da2r>Tz~E$p<sObv1QZDUpthgQ_}M4N_o^&i+iblJ2#+L>D%QuH#-mUSlHp*I`hGN?j?!r~Oh^7Sio<j^!wid~csBG$gEaM{{x}*9lX&dU27~d)AEraM*6*t>4OWK`y-`X(BUxEX*plczQNSmGk3PlpcAU}xbOg>I=u#X)3$w!^4k}?;K9(I_H+6;1-QOGIT?*aSMyx=dsH>W&t_ECe+-!v{((8>7qyY`o(HBW(wv)iUagZ<9cj!94GTy;ZHMXd!e|vZ+Up5%Fq`lr~__KXTjIC<?Cb@;4RW5)UVym(V6torEH#L(qy-q|gUHK8&P|2;hiz8#_OygK7EYSQ|Co5<S3Z%9ww9@?a5nm=sYAD-ZDSgPjZgc7cx~wk0D-tHCZb@lwav<YdBV+behz;&chTfE|P^ARPN~*`50ty)!E9Ux&oNeD!R2w(s&|>xZgNWCkDZ>Ql&Fdw4PJytmuFX{92)u{`ZN*SwzJwU6x)a;&euKMR4`0JUL$XS1@m&h*<9=I&P^2dR$Zg5>VI)Gcg_=a$`TLv%jCYuNWC!9XXhy;?xXUR#SE7EChGdLf!9H%Dr$#f?_R70%ZvAAecmy-brt~H|%o7~z-1CxwM!G`L!GB&{dMFc{&h8TKfg5c(w@WCM>Uvk8rmb5*7p;0UbK7kUju=qXriv9;rX#XZBU*v@l{|Sr$qp$Y0RE}2n)HSzY`LZ^5?1pkL{jqo1|**>2SxWPzt4N@#1=*iWKZ3Fg{|}Hw%B?p5$E?rF91LuxCAIsj}q?5X|&YxVDR{2edD${doMH%^7+j3+W#iZW>#x*3rx_Z$nvpDUBfzuA#T`W2@_x`JC_M{yUxM0WCLy-Bk=pHvXpbJzJg!1ESK+cO>O(JrJ!!Aa>?}-JflE$36l+scB77dThP^-E}ksZSoB%}uD_seph+ylrb>vtkY2yvU@{NSnVul2=zE~@p@oX+48ep$Y`LpBAK#z+^W@FjlNWDK=|)u2l?IXa2?X=-;EZ6$=X9;DXdxMMpX(&shX&B>d?2G&1K{0d-z4Q`Ko$|*N}1wKeY<E|%X*t_z29N@BYZ5Fv@yQoan%BQ*ndjGt$6rdo}Hin`#h*h{!aBiPkM@vxgCA?!Fh_s)K>8*cLiVGSd0D(rg9@JvK%*_%XohWro@nn3v#^n!b_ICjGfw(C<N_MCwInl?`VNjso#4i+Z)Oqbxv<uiOAz^M!Pu=f@B>Bfq7NUY%fPI+av+49y8x9?gQ8l=o+n>)T(8ELM=ntOUu+(DbVn|P!{T&%hm0oq5H`nY38Y$cbr?X*v$33FFu~1e>lCkXqyXDfunimQ{lk#yjUdq2LWn(xAq~zhdm|17rtQ}szWcOpkco36nq??lkFcR_RuM$3#k?;62VpjWphPCIpQ-Ai`8eUO~CKnQtoSdSU8PfV2h&Qo9+5f5$_xV;mCZ#M5g6V<<6H3@H7%mrjohnw6Q6tvq$-dv+^3&ZgendRYz$W#Es*;W(n#u<$C1t!Z5F_Ulp<HXY3dj_w!Z%_*yYcBk^i3c5q}uWZQNwRdKU$A8CI6_QlD|-%sCDrqz?18>4-Ei)wojDXJI4KmucGG6$pavf~&SVcZW$S5}+x8Fz>D6`wZ=!n~Z4L;}X4sVu5yL4YJ92;Aek6mLvUEFapIW&j=McX3ZSO5P~-vt4BhHZ8o^twFNloSaXg`wdK37kKCqcqB=d&nevnUfcsqUji!cS{M;uTQP+zX;}HH(CV<eXE!M2lHWzW7S98c1D&lCz-&3BM5ZUwg*kq>G{<*T*w9~Vhfw_lgDA)_Zan*wydbTD8;446l#6au%5D@7_L1~=Hy_@czi`mTPcFt$SB`^dfvkez9^I~&;%M>psHiu=xz@Te*V+-DmwY3moj<LpGyveV=ZV2E_PxP$GU>;oWEdvHSt^F3Bu%IOH1_;~FUB6$!T7XNE`{*jbexPv@nGT&;{G&^)8Vu~NuuE_9*bcTr_;eK8l<ECY?O?rq449;XqH41;ro7+h-5Y)c<)Z$zxnm_;t~Ny<NkEyg=v(G)6h#q97luDAB5g)8qWG&7>da-iajxlynY;qz7XCRN1pW4v53=z0Lq&Jgy)Y2eSa_>&*1-sv(aEW@ZwN}>1a9;{Q*oHW)}B-e>O<Q-c$^xVmum!gK3<^o(HW69^S;w9pe%ei7(m((lFlTNA?n{3Lo_4Ki|AMJ@aPY`?WVaqO_S^7GjQk&mViUfp_VH_&u2XXLkg1fguiDorzuLV!gv8V>w>^8l?p&zpCkLP;>RKzETjilaKuDw$Wvp+#d)5sl9Y^*hcK-0CY)&%VRrPT!g*@l++pn@fYDTO2T8=l-x~Pcrz_8sDY$KZY7HBRte1n%l5>%E&E6F5}~2Zn2~IZ$DDE+<t<|ph0hDQX9FdhyZc?*ok~9xN(RjSS|s(5_je&il4ZF7S3K@nrs~IiOHk=2feAjvh33rc^pkACXpGn`H?@YEb{&I4fb40;)1&EQI;^}!bHF?!=;Ax~VL+=KX>pjxSvHqf1t74jzrZL~Dw83M;L|6R{3{Dp>Xg*fSePqPRYR!dennodsu)ADBE0UTkl34e;AR{&G(fp!XrO}5qefe&5@S5FdzRCk<uA-0!t$}{vipuFfL_YdZMW>NYvr6)J{<Dk>Cit)+TpTGn{vBug25$8Btycv-T*WI8D*Q&iWN+#$62CSdTL#{@CC|f-Ir20B$*6^Lj-aS{1mW;_XC{{vQ4q9B@O%7o)=L-!%O33xLU6$9r_~NfRID3<9UWwDiq%3<`?rclZYsoH=H}-bC-kx8g)0nU4Q8UJaWu*+sZ*h&cY09H9Fj`*E_ckc(etwd}~0Kp_&ZXwW+sDihzV}zj7p?M<;+OKN+~ZujisXoC;L>kvw<QMKKUi%R%Q>DDqzuId9s;sbj<?-6u^EX$q^7ZH_Ue+)*s%;e}%EB%pBP{ERU`ugiu}xK)32b@-GP14?v$M3I~M5pln+H&aPRyX>E!Yzs<KQ1e-T(RoWtj^m~w-3XfQn;Qf_Rh~529Y+zkrpu~yM>29TV!982*hBAd`|i(NUwMPaUn3C;!U)uo{Z7#gx7Tx3J(*HtvR4%Md3x>27kZ_Ytd7;D3g1djArilM3Ak=wKu0*EYhzw5l@3%5qHrcqoAFRESlfe5pMUEX^uV@HyFAja8ii%dGxAc-9s7cmvL8?`)xy{7x{bY1l^xVaj!wHQMftO%T-=}mqw9J7qXKGv!~#X&c|*_h&=<+B5M_Gt*#)TZDiUKeRKPtR0QNZq*^dto+h_~#Kiv5cQ!Zidm;@CVjB}GAYTU>;D@u+$P}<6_YE?KFN?<HJ<oh(uzP4OzVRh{sxkK{X(n(US3SVkGQcBcDjK;HOw4&(KEJ4e=%raJqc2Y}(dF#}q6}r@i7aj7X)RJp)lO*%qVB+ta@@<VoPg!qLeZxj0>E4Q-ueVquXc1x<EzpcxvBfAP3|DeEJksI2JzbJgYqV8W{w0s5?7AWcS*N+fuobeypGAbmmaE2)#IsxjE+Hqi<Z);&1@W&<f;dlYZb|~@^od@z_)9X!^NMl+u(REqVbD*p-Oi0KFD_r5eY`yA2l55_6Xe!Za}7~#Uf63AK}P{b+xrq?fIsi!ygzEVL5pCNcQrR4Y>zQUeaVo3B>;*yU1S4U-Q>0F?R3><Jn3HLwScyf2zPVFt@AgpoNqj@O_n@7cfQH4j(cAEkUz&woPUG=JUn){V;)pB+`kw2j;R}{M|8H~AP|&Aft&Y7-DS4I;9j~mM(g@|v*CgySuRn8<$=R@8<k|nI4hM-rzYn?-EKc+i+^Ab8SGtF&G}X8wE4<J=ZYa*oTIN+DwmX(2vj#blHafaY-BEk!yBiC<Kha|j)7Fw+MXD5XUEh%7KQiF5Q!wCA&Ia^=m%mVuS)`cc`6+eh;*EB5$S`EE%|7nS$7fD)2<hwpz$M86SgRcN=>d$8-tgclV+)@W5)>91zMG(tk4CEzA088J+Ls0jHB<WY8u-Og7XijFW+3eIeQ<R{qg?vgRT~0ym)=`{?*&l3;jyR(&~ICpavu$uE|s<7AN~%h>b%@EN{dPZ;RIv-bZN1l(!d`@1)@p&Ud&%OiEz8;(fAO^|mWaUC!_>kpgfKK(3&4B=6Hhkwuky&M8Iks;n+BWxV8|d>qmWaVxT|lWqonP<$+hNF{<ewE%~fNF~eJW6tcwJ<=nYd!=X2RbFuc-oimFuWZMuE~&PNRklH1Bhi8yC@UA5d_tGuZn<CRB6#vnKIhtb{IcHV1Y^HH*M0}9jqgy-Uagv1z^ez_e8#RHJT^QDa+EbHiB+4Rs3FQ9^z7XrZQnFol=q2|x#ZLSlx0`>RMnVsi}clPQx)ZFQvBGlT~;xSV&9q$jT(Y1nflF*$zdvBd^TKTdG3Kc00AEe%R^fgXhW+Ss2j}{m|!Q#en%GgfDax)YT9>&mU8G<geKR;pzuP7U3H6Rsy>I|!25kTgOxnyeAmQ{+T{|nS9br`vRa`kvLTam6lFH}G3MLiTdPA-VyOpRZAAA`EwKUMtI|p962?ZR?T&MHL4UNrw=4JJ?Bj=*r-9*c<Cd}=aM@Lbt_{hpECP)MtTchs8Kw4g;%eyjb0=AIEU4HO$ZTmDPe*oX2Pm3_n+)#{Z9-6vSa8wigOXfEiPLqGu%aL+y5=-B3d9P%_>xMjic4r7L;_T+Ww^a-ItRKz{r6VCO)#uRWjM@0g3=;XAeE?P2?#TQDiD2iCHuI31P)K?v9a*1o&pK^%Yji5YkJz4wsqsODN_l~gLQ|jD%tBx<QHq??+K?ILxW71Bp6bU#?vAN3I`qbKo-J8WLQXOy)~-e=#%hW_s9AMSTT=v-Bqx45!`(9TfLQ4<;*bq59;2ljlMfJZ`6wfJj9_pWPj=Ky;27Svra|i<~!e)Xa79?@UP(H{QSe&KTqBU7q3s~8h0#4VjA|n(NLs=D2*l~;fwKf=!yOm{uxKnWYSNPemEUYlO*!|nIDa&gV-O(gLsm<Mrr5Td_T78qUhP)G@Ib<5$_GhtxLyG<*TJKWHTG({Drx56K*VurXwLuVZD-(C^W0K*LdD%sq^Ehh-&P*KbAc39MBl;*@`)+GNjeCpNc;J{4;~=IQ2fJg>13`*nE3q7;aH4k&K@!iiV4zoG$|_RAB7Hy}V*JBVN>Wt&eMNyejYY)fcBg0S(n^LT12P+%40aU&Cxs#3VQiSUA=T$iP}46)j4a7DXoVQ_HD3R(N6$q_$wH3awF1uUwVnEYKk_X27vo%(VWqKi5$?Mx*-u<6t-@^jfF#&<R;1u~?(*uv-@)278VDjmmmW7k0Ex8Q-nL(~#Wn3F)w!Gk4YI#$tWnK5K>QPv@hOr%EnQ3iyM``yym8T+_;mzy2HOwm`B}F&2Y#z~4*Kck8F=P&)AZu4fBW#)0LiUP`}gktdlgu*>gNghF|K$6SD^G_M9CAk~j@*j}-{CCB+P*jnOIr7S?3EvWXJ>j&l;RL#OHuTbI<luCe)@5}R#4w+1_IL&@_#|D!-f$5hqYDFz8A|+b@WWpMYMIe(3sQO9}v~saXl|ML^p86{DY>xVVkqgHZ|M7#~0WMQ&^JI4o{T*A%0d~vm3U8#WEG)`Kf7pG3Kv1^P4*#t(hvMI~hCE;ENk(<$K{53B|Chk&MUo%MBt-I;N(Sv1%{ax1r_4jR0hH5JuD<jh5g=}curgNq6_q1#wdwIEvM~nHt7PZf1naoH<do<&DFsa9VA!KFEIkn8SWN<qJ)=K(NMzZfgz&-?CF_EqG`H)mXr2@(H?}e;ovnn*MvwK<bs8mT23ToagsW^mb^YYo%_@`yMGT*tD>W5Bn2tj+U!kl@fECpQCY%K7ozGit2e0Ul-F91Nj4Y>>th#6wy*4D5v^}MaI2Q>~65Ij<OFFd1^K9LFfiT~kmF?)T>6BVqT^@zXiD;9>cMDOQJ3?v%&RjW#fl^F3)09D^I2JI$1zlnh1PXjCBRn}a2o)nok_A!<*vumo)VAT5F1ExJujNn@SQ+!Upn(Pv9bJ6Nr?EVce^_{RTpL|fhTV`8)jG*4eOPS@Y28W_g<k-_54K@Z#7SK~O5JbP_s8C#l1;5{H~^uXU(Ke3r5-u)+V>P3Qb9uBm#6LE)Op`3#l>Lzwph?io(VK-tv2^=xk|IK7o@_GOE!M0HQaX1Cv>W=7&O%pbnyw6YoJFmln<|bDho!y;SG24l(ln-*dRYQ$&O0|2cBB==b>h>7JAPu{H|#&&tp@&LVT(&1kH{)2~Cjyd@z|0OZ<}iY=H0yoxEI^)#xzE&Mh~c^%q)Z)KtH!byXR^T>^wxX$(-wrW8rZuxm0d$Q(V#MYk^Z3y=7Zhg$hSru;Mb0SvA~*K@(0xL1-!lh0+n(9xr-;#T>kWa)hTi&bQ*fh|fY;hso}9w$L1s=#+zMn`PiEQU`&lDclp24>=8yk~5<RyDkPK{pj#*9!llS{0?cz_M^=B);qAn?<(64|91TYRSAG=yu}x5z(|k0<|Tuyd;ozK!|ISxl}lsZQGUvym-m!7))K%N;E+?I$8sjv_;;ivfQI~uY9kHyN92dQi7Vs0X)gIh5ay#C|xy?V@nMj7NX7RGGV+Xzn8Q=XCX)>RLItlTT*sC$BXt_0p+nngQ%5O)}hDtp31dcTE$~F31Ms<$Vidlz;9gL!5b_#!ZsG;d1O9k7J@?wOTVMz3jGbQ`1LhLm__;r#?4(7L|DBo9@rTl>9sKx)c6)O4-jibin&2BO+hItq4jT^S-U#5Z|)sXdA#9Jj%mSq8%Wn;6sK`y$PH1C#^J%=(L)h`b|pEZn!NhOYs0)$y_3HV4T3($5pY}Cf1tQ;(VZHOepU2cU#%CQXwI&OqM@i)#VF+*zjhpyYw&xin1yY*QSx_G*iyerkpX9;PP`lKv(!B_XOtBJh1l#Ot%mcpkU<{Q*;?$Bo*3n=ue~s*dr0`5!fD-&hv;*4U-P&KD9ET%!3zCXWO=i%ZRdYj5ANGK7cof!YOFNnh7p#fq_0v+viHM{GOWWVs|)dN$n!Ke4p-?T%5}`?YxIXy%1r6YvRDyr7iie{H%S9zcqorxg$00dnt@md^m9ca1pUxj|C^r!FWWC$mG4kLf<>1#;ZShFrESrcT=BzeRH9n!AH+|%d>VyV)|jqbH39;BvkUqnn5fY(WYmg8cY-@~aC9}0Ms<)zRY=Mf4nzwhyE$KkCSyjlDK;<!9j;DGU%}RuKclx239j(bwe)f|88Po`5QPb&B-T<cw$fm2;aB4OT%0f?PLdUcCBODoNw%bA&yd<+cgDvWJZ^^3XNMZj5q6s=2%XSY8I7zQ&`jR5Bo9;z<K$j*C+Q;Gt+?i63%+lIlurkuk__2Q*UQ-(HerU@lvBJ{l8$3UpKR$UM5Ylra~A82=52N~Ygr#v{r$s`ZW{2<D5BWxqHvkIc6qJ@n?-iRm4k8bId7;!34?2?%CRiwmBwVgsTo?DZ85pueL#vpNB+j^zv6F7>L!(#T$=6J+{XX?_D~nfMWOAmR){DdNVRz3$c-nHl~Kz=hpyg$m0^)O-Cc9>HZI-9Dv{QL4E3jDWMF-0@GsJ^8F*v?S8pHmhLFXN%V*V=Vpc`A=FZh!0Y|znVqxy^N@-aj>HRl4NVFp3QFA_?%o4SVPw0%_VZ34GS^K)0QlvC8HC`#&+*#Ek<Q^ZnRY3Hj8P*fwV9g>*JcTz*8pV`-s;-1>(P)l3xY{67cg434HU03A&93;#<GrcLHg-r2YWm~_sY?+^hvq80l1O`vbc!F+-W4w>?f5Tc>sNWr>kV^vO5(AP^Knz|II|I+gVKzD{#gmgPr^Q1gnp<Ix1=D;>FPCgGH(~#hK!5NOj2dHtC28?PU<RYC<8H}rM**lWHO6mlg`xuwD2LwL2JXb3$*~4I5xs4zR?S=Kh{*Y=4<{A<4b8$vPvj^Vwr&I%Ezfw7!U~yGGLd$*ux+W_c@4WAmFP?uE^~(;3_mcv<TYw^VRrQ62+?}!AK&n=!<4nB;Lqe7ac#YGDg{ixZ}W#ge&5fF!i@laK{M0VTWO$QuUr8n|9${F8jGIxf8%3w?AAIswv$TyPFqx!j*b_?wghNJuzlx2l+_{QUwFkpOpBlu5PLV;?!A2&dBxn(3~I=*SOTx1+CS!aK|}N_Pxn(SKg9ijnl%_B){^mRoPRXVYo#%JM;jMUguaYRYf$zX~4a+1TnJxgk<;PCF!u;DKZ|lEg?g@Mqx>~NES-Y3)J%!Q@RN^)ZaHu;07pf1sWK%rsEiS?W{)fEnhxszk<rubD5B9xl(T_3PyT>a_wvpNipndFZy2m`nuvSTx1Bo`1s=D^7K4-adP?cH6!yyfz5xsxqKa*pIyMa)0e*oznz>{5gZHjgiU3WWd@saKh2dt>yrytoRJUY8s3HZU6Ai%D#=MY3)glpyq+u2KwrR=!q^v*7W@gS1pcJD>b8qVfERBRVp;`c`HmC~DLb+ND2aaP_3(0*aGPq+z`r%%Oq9WP_z$ApViD+fR@|$uBaSVr%+PR}AulDF$1tzv?yh`EAy$+T+WqbPBYB7*rEtO|$({Uu72j^xtL!f>=8M7=hG&EebOr{6t8igm_6}W?xhFDhAoD<}=&%o!<!$nuT<_p#cb$@hTH)_s+@9kl%L-1*9vQ{qet)IzWE@B06nvRODYL&owF$LPjtYT&`SH~WkgDxUEXb@zq4*ARQgOJum&}9Fbz0~$u1<J!vlTahs$Jse4yVzg)!eVtE;~}*&??B`p5CvWcoAkxXPr9fez7S0P>(2$F!~TtxT|Au`{$f?7N`Z&enPy%{amgW0<#1NnYxTE+L2b*jU~;WJ3!N%Fh*Z@$}a!|xQDe+)m|T+5!M!<^W=yPaPY)P28>(d$m7qYp3xjp_;otkBU`ynX<BU7B<;wz?OK=lXAhlcScsln&z>EBdsgHuD%Cz<m*Kzt$I(9DQcgBZtR}O)gCxZ9yQ;zqIm<aOj-@P<LU*@CaBc^wkt6eyE&0X|$cV#^M;)EY^2eNti=<RC+12J=r=t>LgTISxRUzFT!B(q}b2M^)M>Z>J$nVkNLq~DTPh?#QD}B|YF3d7@RVi_mnW0MMr&8&}N-3%uU7;<v=P)PD^2a=)WgK4z)4*T{RDwfZnFWMSL15&>lj6pw73R1*&<+}K^$2vN9#1FJbU5$_gESh&-guY}N1o@8`cddbgTZJZM#4*`{mD#t(<n^_$!Hvk6n=;#9>$m_3lrL-2;_Qwg~-U7J>U8@(7!;!po$~tGUgsS3$9fsfYP#=kCd0a)l8Nu=BGdrnTV(?8`^1_N*IuIW3Y1bw0sXc|87qHh}7aDwUD$b01|tH`3x8i_YF)v&ojO>lJyiQi<S~>9ty?Ds(oVL7=J^_ljCcR;x6rKX_$wa!n<bnSd~qP>%ElRL5(!krR+jfVW?CahE2<yV2|EQmtfMNYAeP30Xudhi6MT{(2bD~q^JT*N#TmVxLypsE|-kq)HFx<xUejid~M4UvupGk{cs*>d?QQHyeb^`qJ45BcI0G*0Lp#nd`^|}OdYG>OsV-}rHDFK;hFFGSmvv-3w5dFOfX2pbwR3=25WDDRI^9-PL-&rBc+&mkp-=ECoDgxy*@3V(Cg!JFkh)IhoS76rdH}fSy>gxc3EXrB{f*+QsI9DE>%xSnSn<Y6CXB{MmFxwkwcLWRj<c&`Nc^L(bX%J6-ZX}WN>&JM}M$U4YXsVZW1Uu8OcA3*N=Ksxhh2&GBUbB<euuM!)l!Fv7CdwAwjpC8)dRdTFv+(tpN(W{Ap$ftC7M?SD9<NBjTaLDb#7QXxLZAqS8KAm$iO1AFiv&u4qL|Mg>z)Z(!!;`<rutv{n*8hY1zR4Ki=&u}sl|78j0<0~(Rx^-4w+rSV$fmv|4%s;2<H%24T#_M}~DX{jZ8H$BJi&3gSdny;nDt<XNatf%a5!yWhFym=)-QY0_4>pvG^M24XKpX+#C{MG#uZi&(3zg;-^<aDsj7v1fm3qqK_eKf9RTsdiq;Qje$$&cj2R+SmiFdnF|*EUuOBsg3!P#gp#orhXG<~|8^aG`8V@$7YtZ#H|~mSW!FoCHAaGGHN;mC$QT%^zj&Fe$=Av7zz2wrZI&nyHcn4WYDkUm6k$t{E?wk%t)1b7kEg_m(W>I6nW^<?FNeuTL&sU!1a;<>kj0Z%;0QKh8e<esRtw%=f>(`Hf$_ItkvMe0=}%b?^e~b7Bdpwu;Br5||PUq@-~`G~>u-c_Qa(lDfTgs50?FU${xvih^KZuQg<(xSi1S-Bjv|L)x*-U+>;ryhDqWYhJ6x>ZaM{ca5a9AR@+fPz@RBD1$x0Chu7Ns~I@q3{0{(`9{4xb)@9MX+H*rj=%AJ{W)ufQdrA%bEa#!iWNW|xgeNhX17*!@4!Op1EbOrNFfPtG)QPM416`Bw&F(lez^?E+u3~%{F^dV(RC4~J7E@Z-R88@YHv7Gzu3q|diSv10V705@{Q8x5+SbJW+oY9w`-Canz-P&GOg{Aino;4%bbB)^AF0)BIjBfB}|;mJTg!dgBKEVc9N-F40dIUb_#_Hjwf@Kqs%9dj(t7zH|L*Hci`&v9nI=s;4EpY#4QP$7c;b@ae5{pW3pgubpOyRh0f|eYjoj%PmSF_I6c1vgvPx}q1`&&L#AoB;vZ4qkRzvLFjf;q(MY9M(WkPBD#2EGtjeMRuv|NNL*cd`EBX9*C0N<rlZ&A0Fy-&7CD>54vueCA6&<WECuqlEK2l<^(#NCaJ0GjmS%LWYy+Wa~yYJ5~=~CnL;_~!W@ax%!zzh6~;MJRx-`<~HT)uhf*4XCR_`D4S9Qk-hYp$&9g@GMV747ecD9<_1nXE{YV=1|e%-twVP}uqk&N@m(ZjFVr9CxlRm=q3#7-)dj%DFO&Q*~dq#-O_wb!)oB+YUa;*oM5p(9n>Mk;sV>jy52^qjZE@%*vL0Se&jt*M6~GsP-T7#fQZ|MGP(f*rMjNt^EM^42vY(AQRR0Oq9HX?IMWNn~wakU#(!_<3(Nz1}%wsZ>>j*(GJ0VxLnk=4z&HN$60g(nMF5{>CqLG)jC*jfY`&HZW+se2wcF{dkJ)-N~+13Nsa%<AWIBKokN0e*;spda`Aic<`s|;-4cBs((I-q3s?sLiTo4zA>U@P>LXXZQ+Blv7lE`&@YH~>yc2lkc_z`Yqo}cT_bh&sdt&>?CR>DB>V4KHlFZ#5=Pp}`Wr!JK@JcA!(u9Acx2e5pXiMMj0yN+Ug&D8}xMUAi-@fFZ9Y-cE;t-^UZ>{20M!a+7)Rt#K2MIa;YW-g!5Z$4t<8UlHnZmvnReV(uKc;aG6U~|RH}of*B)Z0(t}aDZ>44^$x=nB{_<GCQ7$wvxr-i3#F3QMg6~{us4zo<5Lq&ybb&UQyHB_n+M*+Y<oxfct?4Y4+>v$bYiOit^sNdw6fr=<&BwyaRb){R;aHCRL6;)QqlEwYhJBYE(th`Rghm&<ylUc!E;SNh_FY=D;uiZB7F$yvpcn+(v?i*EN$jF$ZR!jL9<{4_DmS9}Cjuf)YRn_(7+R!A+HXVaQ%3R(=xGMG8nlh)Tl1vSDfO2KxCoyD}xZjY4NaT*b*~a|PNKlV~|KH=KDoq9rkczobtveJDU~o~E!0TQqE~(0jDiJ?s@?s)hMpp5;#cOIU*#(2BO9QJ@lxm3cvb1)-&6b=ELub-18PfGgQ_MSD;Yckv5ueL#i8~QFs4%M*Uuw1tX`v<g9V!q;wfGv4s=mi!eRG4E0Wgs_EdH90GFN;HNb{+UAlCwhff{JqTjqFw+`~7hf${)hVAX<oD7;mu6E>Zb)WepqIoZ&TJwi3iRY61I<Ok&-Vw3}`^sEIv!CEY`O%M_)=5v}qY`^iWQooWpzrveB`qB&B5=cwc#dC{bwv|J<8g(sD4vecqLBnkd(`;-4WI%+G4FGF|m6nuQ9G-U*4k)@{`PPviEfA+;n^kL~DaRi!V1L-a&(y+p;R`WCJS$-t=8UU4v`cy?$WLS^ui%_-br#fOeU6>FC97AXR9TNYSKB0{tAw{~+$KLDcH1i)HKTh~xu9l&pEI%AY}dD0L^a^+W5&WQ=HCj?+KoPTdg*%BtGaK=A`+m~!_-0Ujs_yL8wMJXZrThDla8ZB42I6GxPZ<D-0(}Ura`@1uaZGEnC+h8=Rp<39E#*xV~FLl6{S%^@$os%!Yq1PyL?-rCdrBJRUc<6snBM<$m07RNg)o`ZN*v8?^Z5EGN1One!CWUA;4<5QsBc(`AQMEMGqkFxw?^b1ZCirqjp#fssRoHEG1qG06}iHN)r#0&)8ps^IGYE@Z6aB<9?%9m93N)HovWA)2`h@1OjgY5I@V$y()mx^?$TadnEDKbyi;Kx+%y;<s2WZI$`pWswTzcni!#3f!4kRc5CaU7K})>mielcup%FqJ_do6%?Q1+_}Y{V1*fk(FlAI4=i7qk80Kd$BeocMoNo)GF0mDDMY40cPS5kP)_7lZ`+mK!sUQPa>?I=3mVB;?Bwh-rGDk_c2H*xUR$;wy!NbS3qn*~|6i}g%h>x-+5x|D9qHw`^2-xDjUh$Ud^#Ybj#Y|<w%{3;ID6#tL{;-OR-x;PJR(V@x#alYZeNUznl2PEe?$+^oF_*Yk;Vbi^Sd?K$oEvIm$x1l)zN<!hz3y(*?QB>Vb4e&{mrf9EL(Q>W(sJbiSJ&7g&DiXB=DMm!<&pwyOEXk*<G@)y|5`ZFhB#>f;E^f$gh@oF=r_XbMKXV-rR6k;F3qm~C{3V;Q8B=xZCAFftz=CSEII-QI^$D~JQV0=d826boifpPYFWne_QaBlT%n;DZ3g$QPR}y;I*tmJky0frviwU1*{B>SvDSQOOXl!@4IRq5x|O2UsHX9;E|}Gl5^2>1tQjOl2jccSPQTx7DpVr7S6|K5_lHdY+TUDpNFbRS`;p?yXDt^a%Xe;@W3KVK;5~=o+R_X)cH7_4X%L0^o!CX_6HolBhoZ&7ZVXSm_Tkmw-RHAZ1jHEoU%!`{TC@6t_gO4fG&faq;~;AJp^Va}veD$a*vER3?m0^^+z9CFlB-1JO$zbBS#j>hx=G>X5D<)f{!O=~9@%FGJA7#9h(6v5G+ZG|LN?WL0x1VzFKU8^7ektNTdH-(0?ZXf$Q)nz+F@;EIOk8sAR1eNET@t&BQ-;d92RF{AM58b>90<cCv)mm>eM|6X;p0*)IZGAg3nv-DqJxtTgN4Q?--vobysgn#!jDv6v{Q0?}QDs=vRrILX#(OVU0*~?<g!vqK8X+fKCyNC*&&2j5OG;zrYL4!!UsUs@uPQXk6=t6s~%frE8YDk1P3H4^zVb1KuKUb2)aiNgTFDP9b#l8V6;mEyUS6Wo2F$v!u$Il~0n^EUT%W%b}Up{>Wm5Sc!x7u#lvZj4!{jv^xq#${qlMHi^AWU9eNQO6m#X{<v?K!sMeUibM%zNQZ@mTDL`6H}3ldlli<N=crJ}cW4cAz6iYSXe8FwRh9vt(@?grT+>h+QhJ)0eTB71-Q31KNy4R6=$UqMX}_kWf7B$R%e=~tGfFDXNh(!_bk+3mgj-EE97-(BHbq^R!M(mJ*?_RmxAF;x{WNyhkO%#V7W@D6oa!&6YmY9Uq3$x0z04?~4q=1L4-(G$F0ap5$A_i%6%R>SrrEJ${<J9OgEZcvwsVVK>70i0CDH{XUGi2dpb|^^6Z!NoM*$JL^qGF5<m^C(OqJRxDdG@x2_W2T7Wd`P;aMgQf370=WU7I=NxC}DewF>#UdRw|&QCf=`pK;*gXCeI7ASilfpBIeP87}GtF+1#eNk)!{X7VjjnoBk3v?Ww(q4dQNJ3)2YFE>W)VnCv72QLJXn1~HzD+A5XhY6+kWR5I&YYBTfYY`vJd4t0_{vWDr!UYO(?Wi?9AQ-3t7b)bQkRb$ZGfTOd>SA$oP0GGl%?}q>9mDYA-Am`zKMhZX~uqFA)jUwQBlbjc)W6IhK!m-Wv*g@RWrgbkBq32ilULt)6poE>eAI$4yCScmupv%J0=Ps?CyvU6SJpsvT<sUm4Cwcw$<*%n?3xOO0rZlXR8x~Ky`ZUrkvS!<$3<zN~2wg3u8=Zl~Z$=l}zf<<|!2!o|Z(zthZ2(!m1SJElaWz8MMz}!?4O;h}47kgwha?SA%$Th*o;9rU6#(;;zg0<knD1{*6WBdKX8$VZw+is(@Vtb<~$K=01#4cOcVX4jk5nOHqW{YeC=g$XIQ(d@Ft0NN@5SL%8XjU>I_S5#{J|v)(6M<GiO5gY0f8QOBlI8&tXlH&+G4g(~qx#E}ghOe<#<&!{VN=4~*8-D9D5*TvzdLD4`PO093K+A~xpCB>Wo_**2>w!nK*D#ux>Om9}lp%L>^8KrTU!mCd@Pp;Z}s77nGs_R2(-q&qow^Wc@so?$1cD+Yok^b#)Nev9_>(Kvsk1QuTKbOZ<v)Inxo#E6fjqpJ(?oKRqSp-B!E+<^G+>$fVFu6<+j`SB8!9Oh6C7#36Q*)<Z?V1+W#Ettyda1tNSXCLY_qnsuVU^VMxb$gl&#+v}4OdvF;QZcd{iUVi^Lrp0;8Jc=eA04%`qxj(pOVX;UjOv&r;Gn|DT1PltAla#J^atGMa3Adj>p%g%Xx0ZXA&>mFC)C2;1xuVGs+Hu%uC^bWuY!teo7T`f1~j~baA`BG1OzvujKY8t9dz1vy7|!>UMr013yZwGCA^bLw(}%CTO;9rq^%5cGn-d8m~%>gDBPT{X3SGsX=BKy~brOJYy=6a#xC2Q<DH-NJ!;8tA`d5sE+5MU8$ce`!aXaDk#qbhVgO;O;J<1p<H;~*B7r(@a|9pP^_*7MBV4-NapA2n3E}s2i>l#XI~@-#U^GVvlTjzB{FYXE+Z@o?suY@oNS?sgGtdO2fULN=Qb2sfu)+dU3N<A8RQR-1KXg{$Bem~gw_4|h?wV*od(|4L)=f#lnlcLiBHd5MN-X#@%359aV_k;*{k~&$$y(RXPC7GBuZKM50Ie@(p@`nOK7xG#E~$yK))+mi#a=hDQN?}VXRU;QbbQ<Sk1@6W>k#OlglF!F|k66$b-3C$Z<V*aHs1cg)UJGe0F@OXe^Bub3w;<E&NYMKEcS#;Pihv|84$$|DvXrJ${v~Zr+?xrP$+|O7M=1E29Q2_e54)(;-jni(RZY`RC*^P8sK9p^>@%6)LDAxQsStqLMm0+K)$ZopUej7gY)1sw-tg)#aK)r3%hdC(za%3#0b1;UX)AZ{+ETh{LV2FUrveB+*$$f8{+ZKPV?)u<8X-Zkrw!a|`x4472cKkYQ)-UWyrn{xNq7u>|v>ya&brkABNuIMiUbR`*)0Inc4TvnT3elz0G}c3YCF|2z9^79NCe@wB!nb0F=b>>Cu7tw7U$#hCqgd|ZX$R$L)bcS0Uvczj}@bm5Ckq`AiAT$2|;FIwHa;qTVJTPB;9-EUjXu3t|N64X*CBP06h`Ynz2kLgH<fanwOc$$*tiD6zBJ)N5SAS)eCjkIQs$VjGRj*9xImijP|?prw<&au;{9Gg4Jc2J-HrY^j);o7b#=F*XwZ;mTzZzxmEu}V{8hhjE#Vz~)dPr;+1mJMJJ$6prjrJrHZP7=l@xb@+rWIqUQHv8Zf)~r6Qq%J0~!XgFy1>!-n{<11+RJv!s{>s@BKD;}96}<RY@bTjGgWI@z1cu6YEcck!S5&hten(`h%1^b;)?2Qw8mLm4YJJ>kJPO4i84f0???>b5D4j;bbmUK?I80}=!Awj?@oeai25IU?{c$uJCh^#x4F=<pKTL;it^ZVA0;moldZUznMzXS&uyxk?Ckgl@@X@E3*o}kvK}X=+c`o_$w=gdn;-C_i<zv~=byHX9B>lb7&847aZNv)XiMpzZXKKL31KCR%Tcr1xQPi}iZWkYYkz_lo_$fyij)Q!;zC#!6m61ZMX+xovw)F4K?W{_xRmI+<s<0L+*u5^>xeS*tNGP;#Y9>i{Q=wAh%8jUl%HG6X92pyA+Q3S2Z|2838ox2>joK=cxB2NKu0NEuPqx3(IfJi@s#Vo@MMePCZMk|?bx+LMS0Ofd{}}2~QEa(LfhnQbC6>$oYq%+>Hf~4~#ai+QF6xa@?BQ;`<cczmy0%i85%41HR+yT~mk?tycVfHUZ*aGplG;l+XfIZI8@@{~eca875Q?<nAGzeXK8!?^16@vn>Vo#kqhu>Ez0=CUk}ri_T~6V-lIWWhA!FnU_Hpw(HJYimSKf7V>nCHyBbZUPm^YbQp5TY(o|o(}(sPIo{`2B)Ls`;v`jT)DOlQl<OLVpJTF;rLt)nV4^=Rg{+ZanQpdd#T46Vz@qbXVydwueLlDSbrnCmp<jq1S@wp>#d39I=NA}RTP0|HK#gFbte-{(E{`NACU$gKAj?#-v$;@+i9li%a)4*+@K5}-sq$|xtN(NfEU(b0?b4c0S9#R{4R`F!Si?SB(yGpjYZ1!kZs635HjEHY($AywGY2NPf^B9{quyUri7Wa@1kBk=pHvM_V4zJgyhpFCN5uqC@xiFOy#r}P_ue|T<2t39XtRCPMZiuhdTHFg_>xx-lwj4~5tZxxl%id6BU7=>IpQ_(Bh)-r=;Bja~ioM<BowqcB~cw9C89rmb^a4SAvmuKha|306Zk_Sw^7mS|bV~!}_jbTqQ)!DiQr3>I&18dQL!R%RtMV80Fa~ZIWg*6#caWjkeEoj}(%h;)XOF~dvbUt0ou#Ki4m2JCsvb~{nJ?HdB*8#-~sUS$!aS)hyq)do&bdyXHP$x0j7?$1p8bjA;PNP;W^Al<r%3fNgqC}Bm=Y{xM-(0S47cHz${z#K^-Mr)6ip6HG=Y8?<{QSe|#YNj(m<k+C#h(fXp6A6P(LV?<Z<Ty1TKKT1WXi%@h+{42rDT2P%TB>r@HrWssl<p=@H;7}DiXo!sxtPWg%<G{h{ftNrGw}9ZYkX`JuJKxsN7A4DZy?qGWt&u?;HZ*$P}*RF(G#<cfMqRr;%_nl|RF%6G*xUa%4_~YM5Ekv!PWTrD=&Zj`Ny%`e#b7$K!=zURl2?jQ(fr7#8<4)~He1gpunq7dtpIA@Wo^eWkcrxQ{eHfBWL(<?pBODY@mz&5hAMzC}f{2=yz9CKgCwOiiY}GhTKa10xK>v~+H?Wmn_UioW9WCPA3DB9cgCSZ`TUSyUE+07*s=sKTVS3eM!j@~UiU2GDVS7x$E8;*C;>+Eu1t)54408hAaYL3|3`Z(zbYr+$yXBT2Hn5a|-?;vQJ~5>R>9!jR@?ooA(7@XwC9bip$#MG)%2yL<M(QB^*=sA=JOKysiHBm$T%#{|fXFuE4P4_8C@j-m$oYwaSapXlO4#xrDSpX?e3hpHTu>l{=n9TX4tk@SEzAKsh4aL~q2u76Nh0fA_NEIi;I-L9A7Xz}%^Nb%3<t-3P3)e)YTd?UkEKdmSmJK(hEiNP@Ty}@)c>BpmF7$(D6Du$yZO{e}e_WXe_#vYa)__R{KZ1CN5oQy{CVB!tp{xpr#;j}+VqTwtai(wL{)4?nnq@(_9l#HjL@Z-^FmP8Zb`+k&&WHupq?@r#o`StYT5&=fz{&eJpX_So9&`U%dM}yEGgx+i#&iY;$ipemFJu!>CejJCs5Z)L^p7hhPh|`1sN)t80^GAceKNyc^@c+WuXfPdkaVWxcG@Xe40HzHyi~GJm8zf_IDh5+A9*x4mG)`jAgH{9&uYl%`aeIfv7d6!sOT--6ORS12(3k&w^Xl}>n|<%s-t34n6?R#OIr2S!?9B$=r4QowVDg{c5zGaKIB<3HYnA@;4)Z$Yc;{FuhP^f~xohs6)mI9l_AZg1-8MSW(!Dz&Ahnmb>)VL0*MTmHa9JvN7U89}D*Dx0gv%%ik7ZM?la5!S@@5o^7P^%%pj#z06D->W<hJZF%S(i&A7e(cF&=YDN|U#YNfgc&<gg2rZ0_!NX?H3eASnMTI|`B1L;k>p7)iFy0$lO9XBkf)cWOX2lmsUD6c?J4Vbf2tPNI6eTW)Gi2kklrg#g*biKj=?$9zV4i{{XwVmzze+=l_Ja-_v!9%tEH+ON5IJr;<T$>U%IpFW}FUs<S9r(}?cg}EYCHH2F3SLAM~^4}LL!s|{7iM@#ju4F+&1C(2a1}f-8ShRI2v%w>~XL);Bj;idMDj%D^qVISD=pHNmMazz&R+?YsJRlFAj@_cH3oiRoDSzK47+eKHG9;Yq4KVYcQMM_qSiyvPoaTq6r`DAVU!aWEeJO=QlF2|gL?G9|PXTLqKk&9Y+Z4-M(y&WKB8dVT?g1~u)p|u4s2AY|gdFl&&NH-|pztm?znG_)1wz5R;o$$zT@nUp)ZP4c{iO@=$Rp5gE6)Hq3o|a!=n(6IxplxJteJIdK$fv~4A`})w`<nDv|l+A(4!N;EQ<_WUKDd36wbaV9T=WF>V^^ssO1@QD-`*!34J!LwA3-;lJ1kHh%|*&$u<WGQt}@b^YA7scd$|%&d(SF^tx;qg<ExmR)<e%F`z`}N0jwCKO*ke^=2y1W0x<+m2E*u3Ti&<FFI9Z$z$0x?HW<9eRG50r%H4syJPotyLu=4!5GQ(Fkb^e?4et-eH&%2ue?*@uaO90ZTa<Lzf&~B?e$z$!lTrf>=nfglU}>>6;o*?t7EmP!ncw)g2XRg0xp$imMY}cQW+T4Ac|XTwHXfugSB0!^!c~$<qmB7w96yys!>?RJR`B-+_A6hD14pLR~Al2*Hmbf3Y(xla&+2dDaxN6<>Cen7+ufn9~DsZBNiwE&l`H4hfXkdQ6AHM%&z%^mtz<dP?WkA(%B7UKR!Hs8&SBOaecy-_=3ql5>#L?@<|3QaU<ibC^_;#H6go@P~p`kfwAzA@6$B<+H$Rh)#Xa$4#{szCrPy`oOtm_DN!3S8qb>1isBJ7uAk0nqAFoFYKbuMn3}Xgm-@JtLoR$;axLyQWWo~6wR}^)Kal7tOZ=&C*k~mED$(Wg7R$0MLJX_~nsF<(82*BBG7bmKIegoss{m?^wyMg%<k6IUCgdRNG<O)bLU#DG2o2bBp$?LGmTSNz<fN874$Y+?{<TRE=c&z2N#LA5(W@4JN#=N7Q4Rohwwp5y`YE>Cx$))2<*T!gmj^u)zCeG1{Ap@#606M%drczfD8OiYUqawpav$gYQNx{11f#sGx!z`bj4|p<h6F4DP`v4C70Bu)uU&7at2U$l%hJ8dYXNN|5$@)UTjy_HIp26*n=E;F?tGJ79rwKSA%BjWIR6I!c{p^nQszEtxPLG39b@mQM|A(-=mV5Rft&Y7-DS4Ia8J6(MC<x`v*8*LSuRn8<$=R@8<lLdI4hM8r6vl#ZnvMZ#Xqoz4E8RoCT%Kp+I(fAlYI~_&e2IIl}pM?1gaYz$#2*IHZm8&;f>S6ad9bV2j8gzSx*f1vV*xE3;%j(2!WAdcveyD=3u<*l7L^HN{0j@H6bI!1Rq=S(L%HCBAldMFF--#N2DffQ4*D!oN+b=FEuC4Qd7qcPN|FVDTnu<QxJVqtUme#Vf+$@eO1-8!fw!z!@^=nSXsi6x<-+S0!cgIx`VNgfOvG>pvo$9O0hJ%;Vc!h6Pq7tNl<?lgqsX4$mwR_2l{E*S<)@)Lt}<Mo#&&d!iAT!O^pJvLeGMv;wFnZ4_B)|SNzq`^x&yMPdBLl-s-mrhSjJH2T>o3Zrh7cfn>ig16hW<<^FLXRNai*J_3+u_XKz_y{A9|^!WTlaM<)oFt<-0;kGi$2kVMK)t1$F3@@;lHx}o@8yl3@!z95_KPvyNL$f&;qUXF{7nQS(eDpt7Y*}9rsOy1s-A#dYPw;5@<7(MrN*92l*=nP&7M@zKnD3@238<NsvP45Da)h)J=jHiFhX@ts^xd!Sm`vRX%vy!sONut&7#&>rt&C5!2ndp|fRfe*K`R%FRC!@w#_q2&-^r-&7om@)f{Y*Zba;hHh-=$C*+{CtWBC-zCYoKL9%u#KDarV-(UL$=)|U?dtunaa-?WB2U+YOm1@s;|C?a@Pm@qZcizGjiF{kpE%Aw^L%{ZdYQ`J0h11PtQTzzS?6>zdcSQ#tYPUWgtZF>BPY>Y7zDn}gFsGqwpIg?yX1c}L#!yaX??ZN3-M|jB%9Pf<&;30v$NXH59wv7{7=M1O7hP@TdlN=_;R>17pN~k1$SQt)6%|bK4O2ho5f;a2>$+H`fKRFMpVd=h7Qvrkl<q&=ULYd4F6VY>-&h<8*x7-dI#*RTK(wRHT={>3!BSo(bNj~iR3B$`%LX;9tfq^BlR^xfL?!7>mZ_dinD{MMtBvUF|s7&{pEWTTa+E5`$Xq~ybin$Rx!kMOkvEo?31Q(<_5(IL9Eh9Xc6NU1XBpn+m1#IRK`ulIgFI{YjsdmZ^T(C0caY6eqB0AM<E}h2mK>lIj)p4zxc^P&?ZW`+(t7!4sR9?E3CJMg*ejjYZzUH7-@KNf1v%Wv}29@CXy5RtXayY$B2}?b4oc;IYX;!Xz-xrpDaO%8om7MIbeN-w^o(a6^T5ay#aycqvFNiaeBLse`1&4OcCv>W=7&Mh%aPbM1{=A{=8|2c7idJR82sl=#q)wi)b}kVc%*$$$9hV3WJhkZ0L(O0<^qyPzUDH~g$EJ2l2l)Bt>O#=$n3K>1`OgQF`EUv~$$d6J_=HX(r?ZiEm}KXcaeV!SmKkpszp8arnb%wbgbFMJRB3w1qn2UU1lY)s%;Tb4t|5wJ=RY25<pY`F!QcllxDF}B!JW8Ql19@_XT8wTBLhvV{8F-XKK{ijGS$Eqr3?U1Bt?&tpb}LmssI{30p{=tNK)60*}zOdh1z|CF~hi1Ut}v;8yWwi6rUvxyeym<c@BE{W|8gi!(5snESVRTWG(YHilz+`s4aozC4sa9LR^!~rJO+57HCPpi<cahV>$&}0Y?;+szZz<ZIN~imV4Chm8}`Ld-$0tC8%i}z>{3oaej<Nl%i$HoLmEkg=llSOc<}p+ML#>r~s*i>i-yWOUe!(Xx^$7P#!xph+1i79eP|}w-POuR`FOB_yro=aU#QQ29(s5%+CyOFh?sJp^WE|`J7n@4kax8j*ctzH@xE4*XY+7=^yB=dQ}i%^|p9mC-|V(##B(_ThKf}tQ9H7h+()&$rq^gZ=6}XI<;@^T}HLmU3DE4tha%@&_{6^M~2)G?-y}+@OSi3czs?;&Zs7@e(~BcFIDg4uS0{N&v69YR`wq#?pt)HhNE8<eb*N`0w|iZ>!D~U>QymHImfRZ2jv?4UMgl`TW*y69Th&XU!}-^vr#ABjrLjU9-1@C3P$~PoU|Iw*FpwS)!W5lr}V@qZ+-2BIeD7G?-Wk!Zal=Bj{BO&g&R*s86#I{HI`Z7zqXzKVLiAnsvVOgpvFp5ZWv)%O8P3LBzr&HD8upvGAT3fhCEMm<8YNeqSRI##;8A}4Dd={mN}kryFkOnzeyS(!$WxlD>S|0Gy|~^c)N^32;Tm*{x?4dUbbJhD&L`g1dA?H>7(F+OWUF?x#EY{s6@5cD$h?i>pVF&vwgsZsYXD6Z+1an1QRtHhKyQ~=uU8l4vww{(x?v7s0vBq77{Itr{MfUnv5Af-q^qpbhxMweFa-r{*2yAB)Gyy*Ya-JWW>C$K@=v8l2i;@Z>7Q7!mq^lxu!!#oFpsiaeVEql59!Ko*}it?u?H$c-#!5&ki-5BkVR$5IUi)G8$PaznHvdNgk*c#>u_rPSQoVTXCt37JT0ZDW48RCHIw?orz<nHerUzvw&FX>l&g@wsaIC(}<ioi*-ixHec_wtdFW*HepED{P<@SQA9+M?_Fiie69qWMRvmlmvHYnO^iYbgKMdRJ1pjv#$>*!8CsfcF_*)AK#D*|{zj{6@i)crOC=_kW;-^w@qfQP)HzU6KnbiBA_@pnEnYaQd|_GW(8)Tnpbk=}yKByQ&l&GoCDK}uq5gD?ggX!QCYS;?1CK1=>g|Kx5VH7j8uQu;Rw~)~+_}0d;K&O@tY;QpDJ=^mz5m9$sa9k>YL3gyEK#fYgwFUKdihqKwXdsze5H}8@k-I=&Pom{_xQ-I0-_hqu$~A9b4ODEGG;*1D5mUFb(MdLMsw7`)drcmE53cG>4%4GcEwK~?@dj%u|sN5(<d)TU8)E=G*{u3MA~blQ~Z$ju6RLN=6)$#zshS~Z<xDN5|4eHkDGD`n2qoplxF<%&q_dk68705^h1rfB?Vc|PN=Dq^6@4`%8+rfnMtbbc6Fs1kxuGt^(X@|p{2c3cw{n*W0TI+0JQKSpaP^f4FH%pHo_>r(F?9W)>OB)f+l{4Ufnb)SzHA_u}na9Y5dhG42XmUnXD}t#Nj>%(F_E9RR9OMeFj{GhKCkGnaAg$$?Kd%@oH*zlE^Fiq8W?K8=31`ZC3VeO2R_QE&GzUD+#y@j|>Uu+GecS#rB6Xa+4PZdr8nxI3b&gm^HIG+wQKil)L&20|86CPZw&bRP2T>tb)3fNXsg(VLVZFtz-f&lK&e^+Eo)?TB*0OeoIl*Bfs*lRY@0~a=1nR1N8FE%O%KBIHG;Bq$S#~uw4}Q>#%VsG9I-pNkCr<WB&X|7J<qO{P5d)YblGjo!#i-<BN;S)AQiP$>qz}2g-cZCoC!-l|%W9c42-O<oh^QwOM7|wWi{%tS4cbBJ2xuV5nM_=pu|Y&+L@y0p9MA_iz=Em3_2mNEw#|KuPpNuZK5)v^SK`7XGb*Bab!8?BxD-{*f-n5xJZ&NpdH@U&XiE^(y;I)|A6a$C$cifwgr&k}mu~mc2taAnvb7(^M^l#3=|Ldx&O#f$SK9;<#TWAznJLXYb38uTB8Xwkxq9y9GrUI7q6*;qG2Ck;@%)C|31KobcvmD{f$$x<oJ?jt-_-?N`cO+K~iZ%Xo)-dcSt!MVKv}HRzT5#iH<^IHG(#SZJ8McpZbBKIi-{fWdrjottozAMWRJy$~4uAY?=~n+zkZ4;f3EKX<V1a2TVLAB9c80PbNeR9TZpXN0->2$ysbSDiZpx5kmE&n1XZ-&-lFgV)y5%}5^7p0?PS#Wy<gZ8_mSKKrzK_Rx98Q5V3s$*2C=@zu9yXu^GV{LJgCOoB%$=Sx2w4dcX%j%0<AF0U&w2=tE84rb4e`~BI(het#c0b1(KhG9Gn$FZOI>39<PB8^7gFdPj>lXRN&XR|OI%%-C>6oY;^8w}y!k(h|#3}YTc80(Wp)=)mQG`itr=#R(7@WM2W$CGqC^ar!S%=1&<_xpn|@#1kh84bJ?BQeDko+q=J7>!|e{prL{CjH6G?|acxhlk+|UFjCzTW$U-jcYI-PY1@ZyfpHYbm+&UX_U-D7)df350ZE|7GfO6{h1$oLw`1lQ{koKvGDx<WE#csG@7Ne$n%~$thzGHrJ>D+BXhz0D20`P#SvmMfME^2VHAh`NjMYZ$#6Ix2tS%8<8TU%!o(kX{b(>AjK;HYnkIum`qYt8S(E$-OSIgF%SC0qX49cDs)0A^Lx270Nc3SdX9G-FIEev?#)Ij&KS`1#o%%40Ndzkzrb#j!ggAp~GDwHhCu5l*7?npf7)*=-&HPz3ijyIrzexL|M2K(z%ajh|bQDiVUg{<Bcp66OWa3W%rTPN)FdZkOzVN5;&65WtW01&kwDvsC2Ilr9aXJ~sgMmMurGtJd2E&2xg(0lT1Xe8aCc|`=rUNhXLtl(WgXt)SMGE`j*oPe!gQt#*(}+tq0nh8BTR?eaKpDpU{?tUSBnkWd$p{DnVDTu82I(~RM}9g9!|`|+PNL~F8u*Df7-N)i8b(3@J(<Kl;JrWcesFYt&FE(Raer)Kej;GM5-%L2aQM98U^p2}l72YoCvoUcLNV))M$_SNG8qhKqa^XCzBd|=fM~!u_9kQh`vzy;H<pGrnv7>-i;Bf!mW=&jn$G;;I2CackK);AHWtHaG>HdM)K4QoJ^?fE1d!EPGKBG^BJ#$e@cq#fhlV+>%9}VCd-g#T!%3QY{ul-_@L*3OZ|e7F$p8p_UyS2q5Dh&z1D*$D%LfV{r_g-ZPy3;WA0_t#;2&u`86G%@fES)Ipm7xX{x}JN%)|N&BjMBH4M)RCH1=Z=PsDUGoMLFbKNitUOeW!YFa%ulfEkVYPaII)(g2GvDZ&KEW9q{+$3Vtm(E*=kLpU06-X<_a5f6RXA>c!^(HPEUKO7}-GKvJy_^IbvE7Oq4xI($Dqd%TbrYRhEAi_YJJXnt52ylPYk3ugR04*0I;U!ZTweY4u<&out0vOCx0ON^ca{xN0T6xZsp}F`&ATnM*^}T-L%>ZZ!tQ08o#2ZADkr??S5%v9XJo1OL{uB<)0N8UM)YnlohUMv7+*PG-D)l%Sj?6PU@WTkU6zHxu6oYYp04N?!CrO%&M<eKW9KqgtK!m4*I8LU%fPEO_WPx)5lPGmqkk&3pvfa=9BNbw(%Z=5p!eG!hWejf&%q9XnPx}2KP~S;18O+ABSsHr-f0U-vehegdm`s6j`h#@RpY{FG2t*N)%z#2ZIw;zWg9Ea+r5_E41L5^YaleoJ5Ku1eM-gDzAO_w!n$Ca(55oZ{#OVO&Z!~~KOn_NU$ME-T`sjc_#|3q$%eU_6E?pYVaN4(IL^1UJG#Vl`h2;Sr8HX@$kjVP|7-WbctY-{|Wta{^Kc09%H(*>io?$wIc8`x|e|UXtBpOd*V8P>Y-<tq7jbRXe5>CCTmtfkF0W2CIZ3K8c3sc~U$s`S7DThEarXv)M9vzNc7d@iN0s%R&#KA$Dc)l0SqP~a+eZbd#Kba0^{$w_pjz&YE{!u)b#xXGL;VhZKaf4G03=t&6>3BTw9v_j6^7#MInCcIfMN*RpIO{MTg@77BJwT8H$vp0lML&)re}+^h@kKoACortY6A4h88Eo!I43cPIA1t(s7%6E=In`^HV>s-a)O7;r3+&gMC86IRMj()QqY+S>cnnXbAQgZV0E`)E{uuCE%wT2uQ;-jV9!w{*bW#}$D9@~KsyOqy1({BLQ|k1_prnrAgvOw<OyU`^GeApJBN8}F|8Hwo*4s7=ga3+==WOs1+X?!VW<eXQONu1jfFXsU_!tTpvC(!%QS`r$6eU^mwQSyAwuO@QSqDk!vq<LIG3*$iw`{b>EJS|_e3wZOgbB>vW3xllA&tl^*<8xL=bLK>)Nh!iGY0BA^g(HOSumR%GnRz*5f-asJFpY9S?0Jl7#L6m$t(o@15=6l;j9}$(bhaTAEwvMT-bYnsfkhP&RkGjj^ncsJaFi1-wtC`ZZWvsUJ6^%1jIHhb<al&5&E62t9u`{?hkF_<>LTUY7j6m-QZ!s0D$k4xbY-SU@MeDGyc&BYYtw=40;|7vgd-$WiWFm(-fTkdu?NmT0Q!IrgS`FW%*eAU^zVP70(3@BQvkSVT6eVNFu|*^V2^R97c4SyItPxmLH-otNnF!yV>m**UQDn=xTm@7>(sai!!Aun-kARL@#9Zc$Pf&99}dR$2;P#7+x?ura2E!l37wW$#V?Zp!}U8*;C7Sj~<5B+1sH5-Tw7mVEf?Q_-?Znh7ad;DtygCGGZ++SVymli(Z4>gcvwcK8E#(g9KZ7FehN_Lq7v=Bnd%%O}#9I%|q(Jf^d&1x?@@9gD*Twr@?W;66Q7GP!AZx)1KnCv#LrTfVH3Hnd)I=9DSqg^gYjotSp|MDGtjwR*DgpV<K+4T6hSP$dJdz!;ogrQKU``FX#d<r8TXtWuW-<1eh>df_i}x^O?(=yb(o@APiNwSG?*<Nt_*?FbXq4SHIZS<@}TWUrN+$$)wC!N&KX4%8xNqyd^&YrhJlB#?`TfxzNYRQPdNa<EiJ(Rz$ggD*Q^A=obEwsmoi#j^H<$3#)uMo$!2=LaAVV6%VgkTNE6F90+5M4Yfb22_C&Qg6)j)_@Vnc^?2<pGvbjo@cFOuMp#qdD4wBH<8aV?fK+4<d;pA_DHb%I)*dZcN)N)4mrDPUs4Vh__(df-Gx!1}RqMnMnmNP?P+I|-dP1zrF(0|!a9yS~Yk;aHX6ur7RF^5$J6>6e+uvDIEFz;77nAf&4TPR}+#D{?32jwOequ!Bnqu`Gn^90?sKMeYj|TK~j*x^emB^;K`ok%BqC^?^Z>(-$dakIZ!-&&}_|;TT5eg-cN4zZmKr13RvX|`luFd~ULhz=+Et$IG9vO-1?;_K~|04b)`iZXwFCO8CR}E08w2>N%q%!6ziI&*idj1*A#{BYT8LigQX1lstt>-t<?RK+R?sn07vj<DE*sOQ^?cHLp%!xMtB~6D4PR{g9%Qga|WwL=-pc=|L3#h7Nfw#Kt;#`H+q1rm;;l%h$wQv*)Hg8fnaD0w(zCx2~N^dB4AupuL`QHk+gShB{EJ%!Y%tHx{voP1$j2^JWY_ww?%5t2AwT<HF!OQ86cDO^SkF(&`c#qzmMGmAL>QFx9EU4NUY#&S{M%qyhB}vXh*<eWev;ndu?O=y8Cuf0eqfPqYN(!Z8!VaWU{spr82gs6F>FDHv9LrxI)(Ms#ypn8b$2*jGQF!sMdtvT64Q7wI)OKjx*o%z<D80(suM3p~2EPWvyUl7#O)<(ZR5TY6bCB0AC{n$MO(Sofr}&iwg>iYLtV=KWHW)AC?Kdpz7d`'


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def load(path):
    require(path.is_file() and not path.is_symlink(), 'Missing or linked file: ' + str(path))
    return path.read_bytes()


def optional(path):
    return load(path) if path.exists() or path.is_symlink() else None


def encode(value):
    return (json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n').encode()


def atomic(path, data):
    temporary = path.with_name(path.name + '.tmp-' + uuid.uuid4().hex)
    try:
        with temporary.open('xb') as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        if path.exists():
            temporary.chmod(path.stat().st_mode & 0o777)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def payload():
    data = zlib.decompress(base64.b85decode(PAYLOAD_B85))
    require(digest(data) == PAYLOAD_SHA, 'Embedded recovery payload checksum differs')
    result = json.loads(data)
    for name, text in result['new_files'].items():
        require(Path(name).name == name, 'Unsafe payload filename')
        require(digest(text.encode()) == result['new_tools'][name], 'Embedded source checksum differs')
        compile(text, name, 'exec')
    return result


def verify_rows(folder, rows):
    for row in rows:
        relative = Path(row['path'])
        require(not relative.is_absolute() and '..' not in relative.parts, 'Unsafe evidence path')
        data = load(folder / relative)
        require(len(data) == row['size_bytes'] and digest(data) == row['sha256'],
                'Reviewed evidence changed: ' + str(relative))


def check_no_new_attempts(folder):
    expected = ['attempt-1f54cd0b', 'attempt-1f54cd0b.log', 'attempt-37b2fc74', 'attempt-37b2fc74.log']
    require(sorted(p.name for p in (folder / '01-substep').iterdir()) == expected,
            'Another first-job attempt or receipt exists; preserve it for review')
    require(not (folder / '02-substep').exists() and not (folder / '03-substep').exists()
            and not (folder / 'comparison.json').exists(), 'Later jobs exist outside this reviewed recovery')


def repair(package, folder, data):
    require(package.is_dir() and not package.is_symlink() and folder.is_dir() and not folder.is_symlink(),
            'Missing or linked package/output directory')
    backup_dir = folder / 'rnn-reader-recovery-v1'
    journal_path = folder / 'rnn-reader-recovery-v1.json'
    require(not backup_dir.is_symlink() and not journal_path.is_symlink(), 'Linked recovery path')
    approval = json.loads(data['approval'])
    # The original ten raw evidence files and error log must remain byte-identical,
    # including when continuing a partially applied repair or re-running this tool.
    verify_rows(folder / ATTEMPT, approval['original_files'])
    require(digest(load(folder / (ATTEMPT + '.log'))) == approval['error_log_sha256'], 'Original failure log changed')
    for name, checksum in data['old_tools'].items():
        allowed = {checksum, data['new_tools'][name]}
        require(digest(load(package / name)) in allowed, 'Unknown diagnostic version: ' + name)
        require(digest(load(folder / 'tool' / name)) in allowed, 'Unknown source snapshot: ' + name)
    if journal_path.exists():
        journal = json.loads(load(journal_path))
        require(journal['payload_sha256'] == PAYLOAD_SHA and journal['package'] == str(package)
                and journal['folder'] == str(folder), 'Recovery journal identity differs')
        old_plan_bytes = load(backup_dir / 'plan.original.json')
        old_manifest_bytes = load(backup_dir / 'manifest.original.json')
        require(digest(old_plan_bytes) == data['reviewed_plan_sha256'], 'Original plan backup changed')
        require(digest(old_manifest_bytes) == journal['original_manifest_sha256'], 'Original manifest backup changed')
    else:
        verify_rows(folder, data['reviewed_files'])
        check_no_new_attempts(folder)
        require(all(digest(load(package / name)) == checksum for name, checksum in data['old_tools'].items()),
                'Package changed without a recorded recovery')
        old_plan_bytes, old_manifest_bytes = load(folder / 'plan.json'), load(package / 'manifest.json')
        require(digest(old_plan_bytes) == data['reviewed_plan_sha256'], 'Reviewed plan changed')
        journal = {'schema_version': 1, 'status': 'PREPARED', 'created_utc': datetime.now(timezone.utc).isoformat(),
                   'payload_sha256': PAYLOAD_SHA, 'package': str(package), 'folder': str(folder),
                   'original_manifest_sha256': digest(old_manifest_bytes),
                   'review_zip_sha256': approval['review_zip_sha256'],
                   'reason': 'Preserve live precision getter objects across cudnn.rnn module import; reuse reviewed trace with a visible postcheck gap',
                   'original_failed_result_preserved': True, 'backend_after_for_first_job': None,
                   'first_job_will_not_be_replayed': True, 'remaining_simulation_jobs': ['02-substep', '03-substep'],
                   'new_ppo_updates': 0, 'physics_reward_acceptance_changed': False,
                   'backend_settings_changed': False, 'cloud_contacted': False, 'stage10_complete': False}
    old_plan = json.loads(old_plan_bytes)
    require(old_plan['tools'] == data['old_tools'], 'Original tool map differs')
    new_plan = {**old_plan, 'tools': data['new_tools']}
    old_manifest = json.loads(old_manifest_bytes)
    for name, text in data['new_files'].items():
        rows = [row for row in old_manifest if row['path'] == name]
        require(len(rows) == 1 and rows[0]['sha256'] == data['old_tools'][name], 'Package manifest changed: ' + name)
    new_manifest = [{**row, 'sha256': data['new_tools'][row['path']],
                     'size_bytes': len(data['new_files'][row['path']].encode())}
                    if row['path'] in data['new_files'] else row for row in old_manifest]
    approval_bytes = data['approval'].encode()
    completed = {'attempt': 'attempt-1f54cd0b', 'receipt_kind': 'REVIEWED_EXISTING_TRACE_WITH_POSTCHECK_GAP',
                 'retained_utc': journal['created_utc'], 'original_worker_status': 'FAILED_REVIEW_REQUIRED',
                 'backend_after': None, 'first_job_replayed': False,
                 'files': approval['original_files'] + [{'path': 'recovery-review.json',
                           'size_bytes': len(approval_bytes), 'sha256': digest(approval_bytes)}]}
    changes = []
    for name, text in data['new_files'].items():
        old = data['old_files'][name].encode()
        require(digest(old) == data['old_tools'][name], 'Original embedded code differs')
        changes += [(package / name, old, text.encode(), 'package-' + name),
                    (folder / 'tool' / name, old, text.encode(), 'snapshot-' + name)]
    changes += [(folder / 'plan.json', old_plan_bytes, encode(new_plan), 'plan.original.json'),
                (package / 'manifest.json', old_manifest_bytes, encode(new_manifest), 'manifest.original.json'),
                (folder / ATTEMPT / 'recovery-review.json', None, approval_bytes, None),
                (folder / '01-substep/completed.json', None, encode(completed), None)]
    for path, before, after, _ in changes:
        require(optional(path) in (before, after), 'Target changed outside this transaction: ' + str(path))
    if not journal_path.exists():
        backup_dir.mkdir(exist_ok=True)
        for path, before, _, backup_name in changes:
            if before is not None:
                target = backup_dir / backup_name
                require(optional(target) in (None, before), 'Backup mismatch: ' + backup_name)
                if not target.exists():
                    atomic(target, before)
        for name in ('run.json', 'review-files.json'):
            atomic(backup_dir / (name + '.original'), load(folder / name))
        atomic(backup_dir / 'recovery-script.py', load(Path(__file__).resolve()))
        atomic(journal_path, encode(journal))
    for path, before, after, _ in changes:
        if optional(path) != after:
            atomic(path, after)
        require(load(path) == after, 'Recovery write verification failed')
    if journal['status'] != 'APPLIED':
        journal.update(status='APPLIED', applied_utc=datetime.now(timezone.utc).isoformat(),
                       new_plan_sha256=digest(encode(new_plan)))
        atomic(journal_path, encode(journal))
    print('Stage10Recovery=APPLIED; first 100-step trace retained; backend_after remains UNAVAILABLE', flush=True)
    print('Stage10RecoveryReceipt=' + str(journal_path), flush=True)
    return json.loads(load(folder / 'run.json'))['status']


def checked_core(data, package=PACKAGE):
    path = package / 'stage10_core.py'
    require(digest(load(path)) in (data['old_tools']['stage10_core.py'], data['new_tools']['stage10_core.py']),
            'Unknown source-check helper')
    spec = importlib.util.spec_from_file_location('stage10_reader_recovery_core', path)
    core = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(core)
    return core


def precision_probe(path, data):
    require(path.parent == FOLDER / 'rnn-reader-recovery-v1' and not path.exists(), 'Invalid probe output')
    require(Path(sys.prefix).resolve() == (REPO / '.venv').resolve(), 'Probe requires the existing project venv')
    os.environ.setdefault('MUJOCO_GL', 'egl')
    os.environ.setdefault('MPLBACKEND', 'Agg')
    core = checked_core(data)
    core.verify_source()
    os.chdir(REPO)
    result = {'status': 'RUNNING', 'environment_constructed': False, 'physics_steps': 0,
              'new_ppo_updates': 0, 'backend_settings_changed_by_reader': False}
    try:
        import torch
        from mjlab_microduck.double_balance_stage09_state import preflight
        preflight(cloud=False)
        expected = json.loads(load(PACKAGE / 'reference-runtime.json'))['backend']
        result['before_module_import'] = core.backend_snapshot(torch)
        # Separate process: exercise the exact namespace hazard without changing
        # lazy-import order in either of the two remaining simulation processes.
        importlib.import_module('torch.backends.cudnn.rnn')
        result['after_module_import'] = core.backend_snapshot(torch)
        result['reader_audit'] = core.precision_reader_audit(torch)
        require(result['before_module_import'] == expected == result['after_module_import'], 'Precision probe differs from frozen reference')
        result['status'] = 'PRECISION_READER_PROBE_PASS_ZERO_PHYSICS'
    except BaseException as exc:
        result.update(status='FAILED', error=type(exc).__name__ + ': ' + str(exc))
        raise
    finally:
        result['finished_utc'] = datetime.now(timezone.utc).isoformat()
        atomic(path, encode(result))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--precision-probe', type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    data = payload()
    if args.precision_probe:
        precision_probe(args.precision_probe.resolve(), data)
        return
    require(REPO.is_dir() and ART.is_dir(), 'Run on the original laptop')
    core = checked_core(data)
    probe_result = None
    with (ART / 'local-diagnostic.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('Another local controller is running; no changes made')
        core.verify_source()
        status = repair(PACKAGE, FOLDER, data)
        if status in ('SUBSTEP_BATCH_COMPLETE_REVIEW_REQUIRED', 'SUBSTEP_BATCH_TRACES_COMPLETE_WITH_POSTCHECK_GAP_REVIEW_REQUIRED'):
            print('Stage10AlreadyComplete=Traces await review; no replay relaunched', flush=True)
            return
        probe = FOLDER / 'rnn-reader-recovery-v1' / ('precision-probe-' + uuid.uuid4().hex[:8] + '.json')
        probe_log = probe.with_suffix('.log')
        command = [str(REPO / '.venv/bin/python'), '-u', str(Path(__file__).resolve()), '--precision-probe', str(probe)]
        print('Stage10PrecisionProbe=zero environment construction; zero physics; zero PPO', flush=True)
        try:
            with probe_log.open('xb') as log:
                child = subprocess.run(command, cwd=REPO, stdout=log, stderr=subprocess.STDOUT, timeout=60)
            probe_result = child.returncode == 0 and json.loads(load(probe))['status'] == 'PRECISION_READER_PROBE_PASS_ZERO_PHYSICS'
        except subprocess.TimeoutExpired:
            probe_result = False
    if not probe_result:
        subprocess.run([str(REPO / '.venv/bin/python'), str(PACKAGE / DRIVER), '--package-only', str(FOLDER)], check=True)
        raise RuntimeError('Precision-only probe failed; no additional simulation started; see ' + str(probe_log))
    print('Stage10PrecisionProbe=PASS; starting only remaining jobs 02 and 03', flush=True)
    command = [sys.executable, str(PACKAGE / DRIVER), '--resume', str(FOLDER)]
    print('Stage10Resume=' + shlex.join(command), flush=True)
    os.execv(sys.executable, command)


if __name__ == '__main__':
    try:
        main()
    except (Exception, KeyboardInterrupt) as exc:
        print('Stage10RecoveryStopped=' + type(exc).__name__ + ': ' + str(exc), flush=True)
        raise SystemExit(1)
