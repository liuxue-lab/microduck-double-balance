#!/usr/bin/env python3
"""Local RTX5060 visual rollout of the already selected Stage12 checkpoint."""
import argparse
import base64
from datetime import datetime,timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback
import zlib

PACKAGE_B85='c-oyjYjfPjk$+{QoD4w3g4iXOB6kT@DH3h-P8L-p<K#l}P*`BVg^2}F0G1EwlyY4XKjJv&ye^6DIAvcF`|_}_&bdUfoVe=#%88=<DSzR*du9OaE|yYGs^TtYdU|?#x_i2N8hhtL+|O8g$WkLYxzN3^OaC`STf>c$fO(Bk<PMx@AHMl%JZ$6x20g<hWH~fNP97Yx20L`Z5qym!gCp6EcB5dJ#A)6bIN2Z!`o;6m$p^a;;YwK&Iwv-u8^%#X^hb<&IQ*Hb+iP15tD!1+oY*dla+WH3gPsN~*dOHbDNA4oK7{cROYPYAGnSXn9M~QPk)rQL4f$`P*Ez-!Y21%<I}gIj_`VZ{<s75P$+94W#VZ<;`TW%zU!J1V9*$FLvowt>t%S`U+Sw%42vVVOlE`rq?bPw=$drf&aVUt~es*(h>)DMP*O5>!aC5avt7Vw2w%(W~7dp%(-ccZiw&gJr<-hS|02qv%+yHEEo0IL%F}1JMqOB~~WBXRyH01E{1zSd|qc?ELR*N;4YH_)7lreXNB5T7ACYjal05#B@r*UX|IZ)k3Bx!Ivj%<L9BQLXBra@fvnBVXNP=z!}g9v!YqAX4`t;^>cSVGi|Y~ZvzORC1({Wt|Z88V&2H-e}EA0R*yff3|vn5mjXg#S>Xn8g7MkL8X#P&LC%M)()#4faRzD3epr^u{RhoSdmRm%$=8_88#o742D`s`Q6&8(w~rFip=#pvs~N=!PauD@WI$=b+JOn4ADDdE7tASyq6iEFYy2Ko8il7woevSBZCS2x`-iwnZ$I+lnm*tx3c2K^V!r0Q>iY$O#)mmZyPB=4O<WdC?nv2K)r*=mh2hQJaUyBPaC=!-I%%r))cjg50*%jD^0QI!AgQr|v)>dP$Yh*lS=fec>SV72vF~Q4?b%l?Y^TN;8n<$N~8<ILWLWf}4`nf;ukABP@QHO-zyHPCsN|A47}A8v)4Rv8r&ZYrhbK6x#bN<|u6e?x%<{judSY-N<!WmOVK>0pH20s3P5Chk?szF(3vuiw-m7uff|VCgsbsh>J!Xfv5li6g|<@kK<5c<R4RHlcsX$+boSIF@$ln|2Pc9m?=bd7#uLQoMdvGJ4#cqk>?KS(RZW)Wc_bJ&Oi^&S=HnD;yXb&0v#Ab0$4&8sdRX1+TX~he#)GKSsaqnO%5V^P)h_q8Yigm!`R7}7N-a)6<6pE%{ry2QtPUgkF*pu)H+X@OZY@T)NKoz3{;tY)@s2~<aqW$5HavhKsi;wQJp~VleA}WVzM260xNrZr3ud>Wv?8;k5(%;LL_$IIAktRV@Gj1RFj>q-fi#cUNE$pEillPGuNWt_FfYXBF%96V-}vyx_X?45?cw)PvUxTI8vP4G13^0R_H#k6WHn_PFJ&kPUrMAoO){}#`6%iD;&cUp7!JjWGn}5#Rl95UGQ7$l?Q(Ps;1_FKlULD&mkV&1rmgQA7ufg4tL(b_KC9?BpmICd!VX(bM@-J=|q;!c`kZ%T!CH^ts_>Ah^W%A53^1oU*$eob&6H)jS;F<^1}FRtK@2N$SdSt)ww?9R;`k&g|w|&=!=8Cc!3jTYTMK*)UJ|4g(R<%Lp4pf3qA>KMctW3rq<?stH>{zVT+H0Etxpu_Q>X`T8}K_1XOu+Nw5;HV^vO?v26=bQi9A!Pe6Ds*^aXZniT?u##0TIxKEFKdC+TFoTd_R0%zkX@b-~203n!DB1CpzK;uMS!H#|Kh(#Mj2}ck;D10Jw_5@!?MCcR*5qUmi$gmxc?T8C4VKcxbs$#-8N+nhiPEA*C#u4X4gGPblD*`*=XAv*-MifVA#v6j-c^>aLkqaZpl`UqiJE-R@hfW$*oQ&18cg|Az<5P|%fM%R&0Qq$y=P3t700?J$Ge+<;tAtmbr_}1<eZ|zY$Dk;VdE;onQl>h({Kl!(t59<-=$$+pMIRh>V~-Q_6i$KxAx|(I4b^J%xkFJU{@HS#!q3|Ccdn#fu2$zgh8K@u5vn#$H}8~3vpklyS^-kW2nAWYf`OA}Sp{vSu7I3fBh`ga>5w&f4Also+#NvV>H93rcp?FP&xFCBD6a-)O_UU97)7}<<`$&!QO14w42`K&4e>RrdJ0aOIwzTR2c6t5=!=s>)I&qDMTSv%90nMk6agBLG;KOng+y1;m8vAequ>U7Lk7TylwJ7Bg%ZASE-TvIiZj6@q#X*waVZw92fSz-%vGRsFO@H0M@+uA6YS|@UToQ4A&S7pmmsLLYX{3E&^TNLxcWsWt*QoN$#|>$O7w9XJ&gc@H4m&x-WZ3kM+>GdBg7fPpibL@u{H@siND}C5t)u6o7g@HCibFcCA2vZ<qS(;v0n(x<s!gvJw-FAO6Rf&dNgofF6b9h5OIh<D=D<?qqxxrDm>7FFm@r5Jq)rDe57$0#-lvzfpTGr&EY99mkm)11OuBXEE?k6$Q=xw^q>fANo(9VIgyz%kFf-wMHf0h6lSwbLO2;pS&@3%cgw6?J`9To{(+ebI0+oYQ_}_78BQee#6z433KD1qP&xapDo{2ULy!gg;J=lRc{-{PnvMEN8Y45xrjzn}9t>Ie0O2m{gYQ(d+@^Hn><r^2(J8Z^vR!5;^h%PalK1g!o)$?}X=`x37GD?`-@qNHiti5-wl9yyz$chSkTuXGlOl|B)F8y*ak;7z4B>zeVq4<CVW>7Zv|`<govA`sq8lV?ypwze{?%s3Odfz_;3a)>BBBv@j(Rlko@_s$;{*JM20I3m3EXLLY@i|_<vkocz}6rFI}?T=_FUy*n}?&2lp1}GRAf!L8O%oD@s~SofCj5OIH|5Q2Z6`zz7s`cS(}3KpqRSrBsm%bP$P&Et?hKgbd(2pwpyL)%J>Whx3@SR{4JuZuK@Rt+e?cIDK`hpayTb*U0Iu0A!z48)(``Fk6RO&_Jrh~%S{6Q2Cd0Y@#yJR!rZ|qI#40z^mNoXtt&=zRWK5a29z(XD-=o-gf@rJbNGF33JQhZa8q}otrk$)Q|7TR{s|IAXmS*IKok0Z5eqYmh~w$M9`UZ|u@;DBm^S1<JZ)%M$_TV$88+w35jzI)#oK^G*tNJpaFQiSaExgBieg}TN$o4UyT_)f%n{Vi$zGX3Bzt>O+FMSl^hj_$7L8tBxpniI-Q9E-c)WYua^O>$<J{~k>(5@_RP+h%tbYmrDPq-v1<QI{H}OC4wHST<>b7KMJWN0|GL^5O3bc0fx}w911Z777Xb|qP5raRYD#v|LT?f3Jt`EU11|GB*g+zd6s%{u0S!qU6IInh;)h~Fli|i%qd}S5?g8W7M*6xMdgXXoHyBDskqJ>7sJ;mUD_rfUmo6Fdq2M{!0*`j<#yRq0Y&4+LQ_Tk%~oWAwW!yo=-b)NUHX88$x^keVj4mhDr;z+mEUN)2C9-KD&LDX$8CdUnDl*hfq@jOt4uGwfULq8hX{S-C6`^8qPwcNhkgLo`XyI*7-cA52iejMf9)@2x;K|l$<UmEy#Nht1!iA?ii6-^(t>n^x{s7-g|{4iI;yzAT%$X6EJj_=)Fo#(W#4q8`ec^|xc`p#>O|M_EsFZ9v-FFktyRk>UMT1~E;{_$6T{q)V#Z+-CSlixhJ_uWU|eDlG**UsMf#$P{u>-1kg#;#xd{`B8JhOP(qz76eXum0)urJw%qi~sWA-Y*E>_aFZ3mycfl_30bmd2sK=2lqcad;8}PfBc8jSMCGM>06(ie)#6uTlZjcK=|mjUp}~Z|G}SsaQ6QF)0aQts2+Xi8>esm{;!{Y=fVA7ax%_-c>nXCz4zeWC#NsJ1V~or6D0J({cpjP4?q0x)7O7=_T7(8-}}bdFFtwjm)CIQz1JW9<f8|leDCZxAHhcP-ba7_;L&G)I(ze*XaDt=vv)uI{N>kS)d>8jp94HT{@Z^(`taSe4}ZMeHkL0weC3VPZ~y%4yDy6wWvU7_4n+}_jzqFMp0T!21Pzvf(Pb(f3J3*m67DH@NeYU+0<J&#)x9LnFzYpA5Qd(C6WZK9$Uzmy*y3<32{N$O)pQzZh~Xp{FR03@5Eu}J_LZ~uUqAib8;y$@?0K8^{KI!X|LM2*UNtTnZNG8${`X;j9=`qJ=Rf-F(P!_R-T&R$2Oq=MjraZ*MCbI4S2}07>F++c{|69{iy3sCz4{+#Z@hGN|CNWodym@WwsCVOZjZ7Cm<tG^JW$HQi@MMdEZMzqF@vWpb;Z{!tMfD*DPDCNgOvHQ=i#eAIQ{fR5Wmmg``+rjbA{%bmupro3U*@DfJw+AM!1BRS83tZ+vA}D^(qhVzx(J9A1mmX;6UzhDtisEaIB#`eEZh|03d(zkpwH<_5$+sjbAA`UmL|#Jh0L#_N?5kd9r4T)`P<m91&~l#>V!%LRSw4b}7A_p-bbWCl&`W(4>N6?)$;9su=k&k+!yg<d)UW*>SGQvxVl>baEEJY{3zQ#t|lt!MWgmwpt+2`Ba5j9y)+QT4ZsqTM#;UDyLU8882jbv(odRCk!HHiS`{uw%DWGRrY#W9s>hBYv*YPIJVa#^|E$^P=n92kEe^pqd<sIN13Ip-(25bzj|Z+YwI_k0rn1@j9CiGlnuTSIys77T)o@!){=*GrY>>{6YXGe>l1>K0y|+T<{T~4s3Iq7D@f_Nt;2x;y|->{*!(sSVX6za<bzf^I>-qX{r%BUzhsYP^f`}q%sukssu}$Nq->KfdJVIms&b<ykOI2}5}1w>^gjA0BB}sm&_it`Z@$4D#Z0-#NvK@PLIWB;m5(yE)}L9wxviH^+iRQ8RUWV2*nCdwif8_TLv>t*z$!YO&`F0X@J?;%lwx&R5UP?jO<a_Le^gD=DvY9#5kwQjkeo1d$x9ZIA66N<D)H&{tPvwIWmjCNwmMDF<Pn6IVK5AGa9Al-qg03xx4f4w9UOszjHOki$jF;u4xzKqNivWMd0k$iu~}$`_`;P#>v6w`YgU1lR7;nSQsdK;3JGFnt;er{?I6OvwGPqKI1x{nLFVy-H6FF~N(Pr0rSodtQHC#sPG9LNEn~qj!2}LRFT`#P4GTugth6*?-3c`>8!c$X$0qt|G`7_y&@>C1Y1jl;5?VWkNsx3S1<!3`1)lFt<l62Wbf+V61J;&-;a3m1gM5G@D~Y~qS@Nrt0%0i{Vpt}VykptglguE872-9P=BkMfvwCZD<L}q5p{Yw<MbmoqP+My^0BG&{yplWFfl^Hv$^srN4L}DtN~ZdV0c9(%D54?<G=vgrn_8|@>S7F(cF4foiv)d+XT`q+rqbIC*P_GP5ouV!<MT#zD-Hwq1PlxWnEGZGZide@7e;?2#bYb2MS|ga#xE5mHWxGM`xuT=vyHAT9C}&p_``r5vDCm@8#iSvvQx}{rj`7I?Aa6@=b_;rHm5*hc%B-sQCr<$dl<NB?2X(519vYoif14_J7788|6nT8^$VoWNtiDcAe-Se|A{7$IL7W@Yr*M~&!Br)@P%OYk;~-h&9$f3HrH-mTeG*;x7Q{RiQ)yEu9(d^V7LOE(19ZwvI$k=chT<1b1=c-96CYhP?DUitkN-;OiB(^;4p{6Z>w!|c2rZ-mF+>0O&v)Y=_wYn@Z8_#_E!;KBaXt8sus=?9xzmVS}df(Tw^{!KO|GZbgPS$4giNHL&I%ja@(g%of9BPt7;=<FVV@Od^uSSK*lh+2ui$meXDBP@n2~jd#Q6|ffQIWLuQe^PAyCj^`zLMH`5E_ecno5J-7oCDOU7xxyPp}7_~K#nC3;Vc{GSa*2E1mhRGtF&d{!n4=(DKxOcR<c5P$xdet@RkAe{G`(cpA5vEWTy-cX1%VH4&=34rgQv~M(IW(3zG2!@GQ|yz!Z%K)Q1Dy!%!FK~=WSfeREOPRuhmx$}Ek1&axW<MJ+(-AeiFIYGE!Kp1U1rD;4zGX$cNBuVq&z<^TcR=z&-4<i&catpBdjSCiPXsY3k7An3kMe8ReOkeGTcx+L=UZL(>PHYQCzv~qYugR&8kdHtFuf8oGvt=C`~JN4de-EwG@6R^kXOW6n&JC6O6QloYk`^*ioEtADEO6hRUAA9x^A5ARw?&_N@+Uc68F7Q3&gY5CaQ4QO`iX0?@)7`Z8;=j+=B|ST$#e3e3`p0$hRZ1zcAqxm<%Ya}Mz)MO5fOxbfV^^|c$|HYPTj4~+R>crsG-Bg$K4HrkPyTp)=foEUKXG2Ue;yvn6*b5qIVcJ~0<>;SLdz<4NombI~o4zRs`ooD<WkAv1P0|CT<3a%^H2FyK3u$HVLotSe)lo)7WF$b-mCf4sL4i<JOT}tpOE^KMw%BMI17bm#Qpqu!hD`=gFSEA)E`YIrb6j#(t%4xU6+-TZiq5ROXd7nDDbv@p0kix68SdU690CR|9Q0xlj7VThEsb0kiaR~*Ou5f`8;JlxwnTMkceRT{`@yxr;qB=%$m)NRSsEQfsy51($)X2CS{6d76Y0!Nch&6_am)eWeZ)d@<w)axYthF*WtV*}37=$yZP-zIi`<Yd`bgG0VO@oPUGi#@{_k{eVY7^(!9yg|z1(HXMLt{&ISxCE=Qyo>s#+D8N*)hv%8B<1<+-Ev4Km^Z|T<%Fw{Vc3!M@a0RD5JG3I4qW0+@q*K2~i#cTke1}!LOhAj)%?(l5(Efpubd8mDbBJFA{jP8{|{*8>Rqr?8-kWbK}m`obq=d+O!gMEaYJ1jIKNj+-j8G*f7X45Ux6EkgsAo3R(zH5@*al+JU%1V1+1F?=I|3)-%gmVmdPF1RIVGYpU3;zNpOno3W!K4XzAnM;ONkPHyqtqLK*pn1@}*lqtYv&f)ahd>ad*cv%bxk*PbkgW(7yv9-`KAS7mdBdunuV`wugvKrM{#e_HqEJM>PkFD{?_TC)Ib%g?Bp`>7Of`GQ6?M)Mai^8Sjx`+kcdLpH_XH7*XT+f1`F?@j^&3ZNbP<hh9ruvd<hp|B&+t6-&Wn=Tk_1WBM(5->Bv}lS;8*6cya#A)s2JNIvc+1pTM2Sui)Q~4hGk!ekmoDMTw6aiBJAaKu(o!~Pg|lmj>@`@cIw7JfV)TGjZ3Z9dMz%}<)@?@_4@M<P-GTz)%}B1(rBaWhouu2~U&adftN-w_G)nMA0`M>xrgl}1NOz8d%xVdX0ASoofssPa<2{kSz%3f~@!}!GM3;eMcY&M&k$qEUGM=tH0fCd1Z#7Uc22Nfx$`8^kuMH@~1{n!hC>~Ap@bZ-g=wff`Q$S^{%)X@x)cp1r@Pbn!@oAL{zUk2-M5<h9-y-lGk$;-L)dtL<rYyt~BFu4OTNG!xhi@t=O&_yJ&Jn+Bf#)PRw*4U%9HsmH_M$RFZ<Hn$9^qCU!=B+lMBMJ+w*dYmLBhJ!?$9)D_YkX@#Em%4Lnkm6FR}E*cc{dv>840d+nhbCh*A-1oZ6Go;bPlNuyGh4`k}LrKy!$hvVyd1y0cYjts0g(0T0GlxqfR+=U4O=unoTg%kXQ>P83nAEa*_j-~Q;G^TtLI@=m|PQCq15A#gG{$pUaGxY6l_ur7YbMIk_qpx}Kmclb-Eb4c&obY~j}DvK-b##?&FY|R3n2<9DO6NO)ku2+`F;96whhsZjLS@FblF`%Ga7S#&uW%XZKWts?ocIs`Enhk^a6W`*=bPwMV0pA5|J2`R11W!&IucY3YAoNfc>C(xEtb}Pr^`xrp%}v4!g2br3x&_9i$2QNw&s!y~%px*rjd;e-1`d4M@r!LNS^<K+@fx7Ie-*(K=8mV?Q-jn=)FPBnm3Oy9_GZp%oj--3%EI!`E?U<*Lp%&TFJu)vte0~>$%;EqzGQJEigXY-ex#ZK3AV7C^_d@h($B}uQ*`yWZP{q3x|N&i;trPIs2A7i+T0UPG<6#X;npRUXRi>&JTMr2&U4xK1Pm)|70F={t=W~z%yY$Z+Wt}DY^8#>F{^I9`kot7TTpOrkvrWhBTakf6Gc^?GeR`vi>^G4j(|+oik$=hn}_BsqWeRPBE?QO_>RsB`yLl%TNV$;b>%s9jj3>cET)&W_ueSQ9Km=JdA5HcblD9dwkj3;$O&>)R>3Wp!hI8+V+C*+?PAq}a7r3jM9aKh0~EEWc#CU8JdY8I$%4JoyQ#P?hgFv*sGjY@uZcz>x;;&$=B&TI_WaE+-?(vZrQd=%YhyT>lZ)P@8r`Bq&5cJP-}pWg3!5F`m)2{lPLi9=D<;)O@`pw$rc${7S!W@Ayo`Nf11V&+^h`u0JoEG&sn2`*PE{@UP-SW|*=YJE>&^vs@u~BSJpu9Z)^h9Uw6N%bgdKuvEHi<SNat@nTks))#eD$((0zK>TIZ~`>AE%m3Me}_!#s|a3lSOQ%4<3g*y*n>a*+74O_@@&i=QmuLQ{_)HWI=}uu8Tl5{`X18{(12Q^0&^QH5Q+42wBr8~2#JLvN<zRJq|+6=Xcd5_H2Y_8i-EcI<|u3}28XR`4nj8O`AX2vJ!tI6aFv04EDKFqOlH=(Q(H**|rrkZgu^_aI=i%ZlkrW%3hw)3OAi7A-eaQ`#l$7wj>*ZK8&H#sOD<bu%>wH=jjDcG{+T-8wsaPKKh!0JqE;17NpVk~h}q1N5}}SiA=TuD$ofv%o^Ze2;ApJkMUnblL&y23VO%$h0y6F%t{S3hw&YtmwYagl8cZmN`dLVpKZ<k58`WI?vbZ{E7gtTrjYcGNu<akB{GXK^#L&4HY-5bfL?JNq%C}dxlf*%}o@P$kSXj=*otysL5E!^Bq_upW!hN3ikr5PaKPy_pzF<8ynZI-msrvzrMC%U)y-@){V98HAUxYOn-Gz<1$Wx)F<(LzRCN4I<M!Il<|Wn1GCuYo^`^D5fBa_CJ3l28{@VXhD0)HXX<!~Z#yok2c|W_nmHit!pJ_UG1e!=hwgFCzsXNd@<9w<$Vn#aZFs&w46|*~YTG(F<#ZlR6*6l0=17nYc%4(V5UL^smB_qW1tnfX;5AVEVd`|4xDc-w9T5ykN|f3Yz^GOORg2b4ZR$-map5D37hWS51`MlwpuPJ9$j+gCZF6&D)4sL2wzanTe6fO}EDW|wRj3UZ%-@oS6FXI5VofjJ{;`U0hL4^A9`3HH?KDwb&F-E)`;648C2r`@Pg7?pbLE^qY|21viQrM2e((r?bOa^Jwk=DsZM?6uZKcbr8MjVAQHE>BV8VH=sV3eYtGuDaUm~0F!$~L_Px#K&ZX}AiEPOTwG0Nb)QdHYC`8%P_KI{|d+5AzQ9%KpdPuY!Rf!7lbb<#ZW!Hde~$2*$h!g!uv7&oP%Y@#QO(}2`laxNor@U7Q*@k=C4&J+S|9<~6QF2${~_RKBrn6r5<S+;2+xRK-%+}?Wjs-<|oxzIM-j_aE(b7|S^FD-Zr?aP;$yL@@YUs_;oZ*kdNW*x89US3-2EUdH_oQ2k+-)eUpcgX>4+gG>#fg|fLt(YtQ71#Hdmli$OciD2=Trii~&gD*fx!t+ka-Ft&nfV=WX{q0KoX)aqcHG6K_L9@-I6fe|wSIGpBQuxX%gZfud6~KW#Xiuu;C5CP0JZNsu#}}%--l6`eaGvpm~G$nFR%1FW~<Zj9A{-|#RX)=l|8J}Tk=}0#g<ks_g%+XSX}n{p10EWJea1xU@};g!&cmdc4uW}VPP4W(OGfW<)seLzJTkz^^e=nZrrqrIzTXm(cyeQh~~L@B;i=QzGbOE>{82Yw{-CZoyg~|e%0PuyS8x?09H&e8qWfCUjuv{eBeda{8r&Whmarx>5vIbP#kmv;)8BY;je_1I55VbSJNKzvrCj_t-Q#_bN6ZSa?W_AR3DE$bh04a8}>b?>l;-34U1d6bg8Y;`##;`N09oD`MD!{VHwABN(C5dFcmpuYK1U_v&&ym;)D6oKyQvwin4MR@Q9=O3Sx<@N3Th#G?6zNCYdTmM;V6FPUZ%II3@8nsO$rFl5vj#oc{O-tXCxP&nfs`Rrym;d^&VI`5S|Nqo9bt6F{Y+RJc!vDm~)2IV#|9D)`D87OsIgPXbRZYSn0;@HXk(x2Jl(L1tH_YJNXEiCh($!0C$O;&>HuR1ns!6#nDOeZ(TkVOjd30yUDG<52{}*%sd@Wge3giZ+jHpp*}c(deasToF4+uOh(rED6EHE1IIWw4K(T!Jns(RUTsFE8ha2PPZWz%(~NBq1U}3fWjwpPsjJA5`x5r_MTj^IFz3TKHQA+r_r?_t&tMPZy@-BAvivSeF-AD+6dNH;X<^d02#`jMz5`6y&!IDd50lTQr-@RiXiGp)OKV258)6`W8FEpnY1-{iLQK2(Uq?%rL0dl+uh1LoMgz$cTkJKt!U3qhIsogx_I+=Zo)!;XJH8|FsZHy%m;yiQ(qhty0Rtx_(jSAmL*K$Hw%gvP{2dA#%rMGD;214fGa&<LpBMM#UpUOn%KpCyvibxBm@GA<WW<QiUXiso*KIHG}ZqQ9T;tJytrMNAXQSzd-y9Xgr$7(A!Q+01Q2%qTXYB~e}k?n{w$=0aq|LCBCUx0mwi+{+t1<<f3Qgu&l-jz_qqV1S95e_J>tJ_Qc_K3qZ%V6#rY!#PlRU4gg7Kq`%O#RQOKkUK_pgcMHfFQls#EAFX$0_-EE+^7mk~FQrA*0Y?1LQMTlY-(JHr5=ZJ0Ozw9fvQuI;aSf<3N3Ws{gLA?8auYMI%'
PACKAGE_SHA='2888e2766f2dbe5357fad4a557525d3996c483c3cad30440e6981ab5a5f111fe'
PACKAGE_ID='a2e2e5f471ee51a183c9926a80bc9520db3a68df8ed413f7cbb1fb2a32c7bd51'
payload=zlib.decompress(base64.b85decode(PACKAGE_B85))
if hashlib.sha256(payload).hexdigest()!=PACKAGE_SHA:raise RuntimeError('Embedded package differs')
PACKAGE=json.loads(payload)
exec(compile(PACKAGE['replay_common.py'],'<embedded replay_common>','exec'),globals())

def install():
    folder=SLOT/'package';folder.mkdir(exist_ok=True)
    for name,source in PACKAGE.items():
        need(Path(name).name==name and name.endswith('.py'),'Invalid package member')
        target=folder/name;data=source.encode()
        if target.exists():need(not target.is_symlink() and target.read_bytes()==data,'Existing replay package differs: '+name)
        else:
            with target.open('xb') as f:f.write(data)
    return folder

def snapshot():
    value=read(SLOT/'progress.json') if (SLOT/'progress.json').exists() else {'status':'PREPARED'}
    if (SLOT/'started.json').exists():value['started']=read(SLOT/'started.json')
    if (SLOT/'worker.json').exists():
        pin=read(SLOT/'worker.json');value['worker']=pin;value['worker_alive']=alive(pin)
    if (SLOT/'result.json').exists():value['result']=read(SLOT/'result.json')
    if (SLOT/'finished.json').exists():
        value['finished']=read(SLOT/'finished.json')
        if value['finished']['worker_exit_code']!=0 or value.get('result',{}).get('status')!='LOCAL_VIDEO_COMPLETE':
            value['status']='LOCAL_VIDEO_ERROR_PRESERVE'
    elif 'worker_alive' in value and not value['worker_alive'] and not value.get('result'):
        value['status']='LOCAL_VIDEO_ERROR_PRESERVE'
        value.setdefault('error','Local worker is no longer alive; partial files preserved')
    return value

def show(value):
    print('本机录像 %s；完成 %s/2 段；当前种子 %s；步数 %s/500；阶段 %s；新增PPO=0。'%(
        value.get('status','UNKNOWN'),value.get('completed_clips',0),value.get('seed','—'),
        value.get('control_steps_in_clip',0),value.get('phase','—')),flush=True)
    if value.get('started'):
        print('原15分钟上限剩 %.1f 分钟。'%max(0,(value['started']['deadline_utc']-time.time())/60),flush=True)

def save_report(value):
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
    folder=SLOT/'reports';folder.mkdir(exist_ok=True)
    report=dict(value,controller_package_identity=PACKAGE_ID,local_return_utc=stamp,
                added_PPO_updates=0,cloud_connections=0,power_action='NONE')
    logs={}
    for path in [SLOT/'worker.log',*sorted(SLOT.glob('ffmpeg-*.log'))]:
        if path.is_file():
            with path.open('rb') as f:f.seek(max(0,path.stat().st_size-9000));tail=f.read().decode('utf-8',errors='replace')
            logs[path.name]=tail
    if logs:report['log_tails']=logs
    out=folder/('microduck-stage12-replay-4500-'+stamp+'-'+str(os.getpid())+'.json')
    write(out,report,exclusive=True)
    print('LOCAL_VIDEO_REPORT_SAVED='+str(out),flush=True)
    print('报告 %.1f KiB；SHA256=%s'%(out.stat().st_size/1024,sha(out)),flush=True)
    result=value.get('result',{})
    if value.get('status')=='LOCAL_VIDEO_COMPLETE' and result.get('status')=='LOCAL_VIDEO_COMPLETE':
        print('视频总览：'+str(SLOT/'index.html'),flush=True)
        for clip in result['clips']:print('视频：'+clip['video']['path'],flush=True)
        print('请回传小JSON及两段MP4；不用回传模型。5090无需开启。',flush=True)
    else:
        error=value.get('error')
        if not error:error=value.get('finished',{}).get('error')
        if error:print('错误：'+str(error),flush=True)
        print('现场已保留；请回传上述小JSON，不要删除started.json或重开录像预算。',flush=True)
    return out

def halt(process,pin):
    # Never signal an unrelated process or a reused PID.
    if process.poll() is not None:return process.returncode
    need(alive(pin) and pin['group']==pin['pid'],'Worker process identity changed; refuse unrelated signal')
    try:os.killpg(pin['group'],signal.SIGTERM)
    except ProcessLookupError:pass
    try:return process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        need(alive(pin),'Worker identity changed during shutdown')
        os.killpg(pin['group'],signal.SIGKILL)
        return process.wait(timeout=10)

def monitor(process,pin):
    stop_reason=None;last=0
    try:
        while process.poll() is None:
            value=snapshot();now=time.time()
            if now-last>=20:show(value);last=now
            if now>=value['started']['deadline_utc']:
                stop_reason='Original 15-minute wall-clock limit reached';halt(process,pin);break
            time.sleep(1)
    except KeyboardInterrupt:
        stop_reason='Interrupted by user; original budget and partial files retained'
        print('正在停止本机录像并保留现场……',flush=True);halt(process,pin)
    except BaseException:
        if process.poll() is None:halt(process,pin)
        raise
    code=process.wait();value=snapshot()
    finished=dict(worker_exit_code=code,worker_alive=alive(pin),finished_utc=time.time())
    if stop_reason:finished['error']=stop_reason
    write(SLOT/'finished.json',finished,exclusive=True);value['finished']=finished
    if code!=0 or value.get('result',{}).get('status')!='LOCAL_VIDEO_COMPLETE':
        value['status']='LOCAL_VIDEO_ERROR_PRESERVE'
        value.setdefault('error',stop_reason or 'Local worker exited without complete videos; see log_tails')
    save_report(value)
    return 0 if value['status']=='LOCAL_VIDEO_COMPLETE' and code==0 else 2

def watch():
    need((SLOT/'started.json').exists(),'No local replay has been started')
    try:
        while True:
            value=snapshot();show(value)
            if not value.get('worker_alive',False):save_report(value);return 0 if value.get('result') else 2
            time.sleep(20)
    except KeyboardInterrupt:
        print('只退出只读监看；本机录像由原启动终端管理。',flush=True);return 0

def launch():
    SLOT.mkdir(parents=True,exist_ok=True)
    with (SLOT/'launch.lock').open('a+b') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            print('已有本机录像入口运行中；不重复启动。可用本脚本 --watch 只读查看。',flush=True);show(snapshot());return 2
        if (SLOT/'started.json').exists():
            value=snapshot();show(value);save_report(value)
            print('本次录像已经启动过；不重录、不重置原截止时间。',flush=True)
            return 0 if value.get('result',{}).get('status')=='LOCAL_VIDEO_COMPLETE' else 2
        process=None;pin=None
        try:
            need(PYTHON.is_file(),'Existing project Python is missing; no environment installation')
            verify_inputs();folder=install()
            env=dict(os.environ)
            env.update(PYTHONDONTWRITEBYTECODE='1',MUJOCO_GL='egl',PYOPENGL_PLATFORM='egl',CUDA_VISIBLE_DEVICES='0',
                OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='4',HF_HUB_OFFLINE='1',WANDB_MODE='disabled',
                PYTHONUNBUFFERED='1')
            env.pop('PYTHONPATH',None);env.pop('PYTHONHOME',None)
            now=time.time()
            write(SLOT/'started.json',dict(started_utc=now,deadline_utc=now+MAX_SECONDS,maximum_control_steps=1000,
                maximum_clips=2,seeds=list(SEEDS),package_identity=PACKAGE_ID,added_PPO_updates=0),exclusive=True)
            print('本机RTX5060：固定两段，每段最多10秒；15分钟墙钟上限含首次初始化。',flush=True)
            print('这是前台录像；保持本机开机及此终端。Ctrl+C会停止本次录像，结果与预算保留。无需SSH或网络。',flush=True)
            with (SLOT/'worker.log').open('xb') as log:
                process=subprocess.Popen([str(PYTHON),'-B',str(folder/'record.py')],cwd=str(PROJECT),env=env,
                    stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            pin=identity(process.pid);need(pin and pin['group']==process.pid,'Cannot identify local worker')
            write(SLOT/'worker.json',pin,exclusive=True)
            return monitor(process,pin)
        except BaseException as exc:
            if process is not None and process.poll() is None and pin is not None:halt(process,pin)
            value=snapshot();value.update(status='LOCAL_VIDEO_ERROR_PRESERVE',error=repr(exc),traceback=traceback.format_exc()[-10000:])
            save_report(value);return 2

def main():
    parser=argparse.ArgumentParser(description='4500模型本机有限录像；无新增PPO，无云连接，无自动重录。')
    mode=parser.add_mutually_exclusive_group(required=True);mode.add_argument('--record',action='store_true');mode.add_argument('--watch',action='store_true')
    args=parser.parse_args()
    try:return launch() if args.record else watch()
    except Exception as exc:print('已停止：'+repr(exc),flush=True);return 2

if __name__=='__main__':raise SystemExit(main())
