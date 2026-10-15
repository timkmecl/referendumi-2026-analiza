# Koda in podatki za poročilo »Odkrivanje potencialnih napak v preliminarnih rezultatih referendumov po voliščih z modelom štirih sočasnih referendumov«

[Prenesi poročilo](report.pdf)


Analiziral sem preliminarne (trenutne) rezultate štirih sočasnih
referendumov (parlamentarna preiskava, RTV, sociala, tujci) na 3190
voliščih. Za vsak referendum delež glasov ZA na volišču napovem iz
rezultatov ostalih treh referendumov na istem volišču. Ker gre za iste
volivce, je napoved zelo natančna (bolj, kot to dopušča binomski šum med
neodvisnimi vzorci), zato izstopajo tudi razmeroma majhna odstopanja.
Med 12 760 kombinacijami volišče $\times$ referendum jih je 53 (na 50
voliščih) izstopajočih ($|z|>5$, po Gaussu pričakovano $<0{,}01$). Okoli
polovica jih kaže vzorec, ki bi ga pričakovali pri zamenjavi rubrik ZA
in PROTI: objavljeni rezultat je zrcalna vrednost napovedi,
$100-\hat p$, do natančnosti binomskega šuma.

**Najpomembnejša ugotovitev je, da se outlierji zgoščajo prav na zrcalni
vrednosti napovedi, ki je za vsako volišče drugačna: tudi če bi
outlierje pripisali resničnim lokalnim posebnostim volišč ali
porazdelitvi z debelejšimi repi od predpostavljene, bi to pojasnilo
njihovo število, težko pa zgoščevanje na območju, ki pripada
zamenjavam.** Mešani model za vse outlierje (izbrane samo po $|z|>5$)
oceni okoli 22 domnevnih zamenjav (95-% IZ 15--30) in je s hipotezo
"samo resnične lokalne posebnosti, brez napak" slabo skladen
($p<10^{-10}$); v isto smer kažejo test gladkosti brez modela ozadja in
ničelni model z enako porazdelitvijo velikosti volišč. Nazoren primer:
pri sedmih od osmih domnevnih zamenjav, kjer so v isti stavbi še druga
volišča, ta glasujejo povsem v skladu z napovedjo ($|z|<2$), zgolj
lokalna posebnost pa bi se verjetno pokazala tudi pri njih. Podoben
primer je znan iz Danske, kjer so leta 2014 s primerjavo sočasnega
referenduma in volitev po voliščih odkrili volišče z zamenjanima ZA in
PROTI, kar je volilni organ potrdil.

Domnevne zamenjave so pogostejše na večjih voliščih (okoli 9 na milijon
glasov), geografskega vzorca ni opaziti. Tudi če bi šlo za dejanske
napake, na izid nobenega referenduma ne bi vplivale; največji relativni
učinek bi bil pri RTV, kjer bi popravek označenih volišč povečal
prednost PROTI s 546 na okoli 1400 glasov. Smer odstopanj vseh
outlierjev (32 od 53 v škodo zmagovalca, pričakovano 27 brez zamenjav in
35--36 s samimi zamenjavami) je s tem skladna, a sama nima moči za
ločevanje hipotez.

**Rezultati utemeljujejo sum, da gre pri približno 22 od 53 outlierjev
za zamenjavo števil ZA in PROTI pri vpisu ali prepisu, vendar napak
zgolj s statistično analizo ne moremo dokončno potrditi**. Da bi
hipotezo z veliko zanesljivostjo potrdili ali ovrgli, zadošča že pregled
zapisnikov in glasovnic na treh označenih voliščih z najmočnejšim
signalom. Podatki ne kažejo znakov sistematičnega prirejanja.



## Vsebina

```
podatki/       preiskava.xlsx, rtv.xlsx, sociala.xlsx, tujci.xlsx
               (izvoz DVK po voliščih, prenesen 13. 10. 2026 ob 19:30; oznaka osvežitve 12. 10. 2026 ob 18:00;
                uporabljen je list »Podatki po voliščih«)
koda/          skripte za obdelavo (01–21) in za slike (s01–s07)
run_all.sh     požene vse skripte po vrsti
requirements.txt
```

Ob zagonu nastanejo mape `rezultati/` (vmesni rezultati, .pkl in .csv), `slike/` (vse slike iz poročila) in `izpisi/` (besedilni izpis vsake skripte, iz katerega so številke v poročilu).

## Zagon

```
pip install -r requirements.txt
bash run_all.sh
```

Skripte je treba poganjati iz korenske mape paketa (poti so relativne), v številčnem vrstnem redu; skripte za slike (`s01`–`s07`) na koncu. Celoten zagon traja nekaj minut. Simulacije uporabljajo fiksna semena, zato so rezultati ponovljivi; nekatere simulirane številke se od poročila razlikujejo v zadnji decimalki (glej spodaj).

Preizkušeno s Python 3.13, numpy 2.5, scipy 1.18, pandas 3.0, matplotlib 3.11, openpyxl 3.1.

## Kje je kaj iz poročila

| Razdelek poročila | Skripta | Izpis / slika |
|---|---|---|
| 1 Podatki (nacionalni deleži, 3190 volišč) | 01_podatki.py, 18_vpliv.py | izpisi/01, izpisi/18 |
| 2.1 Korelacije | s02_korelacije.py | 6_korelacije_matrika.png, 7_korelacije_heatmap.png |
| 2.2 Model, dvostopenjsko čiščenje | 02_model.py | rezultati/R.pkl |
| 2.2 Preverjanje krožnosti | 05_kroznost.py | izpisi/05 |
| 2.3 Faktor c_k | 02_model.py | izpisi/02 |
| 2.3 R², RMSE, binomski šum | 06_R2_RMSE.py | izpisi/06 |
| 3.1 Tabela \|z\|>3/4/5, kurtoza | 02_model.py | izpisi/02 |
| 3.1 Q–Q in lijakasti prikaz | s09_qq_lijak.py | 4_qq_lijak.png |
| 3.2 Razvrstitev outlierjev, Priloga A | 03_outlierji_po_modelih.py, 04_razvrstitev.py | rezultati/final_table.csv |
| 3.2 Število glasovnic, Sostro 25 (384 prerazporeditev) | 20_druge_kontrole.py | izpisi/20 |
| 3.3 Večkratne zamenjave | 07_veckratne_zamenjave.py | izpisi/07 |
| 3.4 Velikost volišč | 08_velikost.py | 13_velikost.png |
| 4.1 Diagnostični diagram | s01_slike_1do5.py | 2_diagnostika_zamenjave.png |
| 4.2–4.3 Ničelne hipoteze, razporeditev v pasu | 09_nicelne_hipoteze_pas.py, s03, s04 | 9_test_pasu.png, 11_z_po_zamenjavi_zoom.png |
| 4.4 Zrcalna črta | s03_test_pasu_zrcalna_crta.py | 10_zrcalna_crta.png |
| 4.5 Volišča v isti stavbi | 17_sestrska_volisca.py | rezultati/sestrska_volisca.csv |
| 4.6 Zrcalna os čez vse celice | 10_zrcalna_os.py | 12_zrcalna_os_vsi.png |
| 4.7 λ, mešani model, LR test | 11_lambda_mesani_model.py, s05 | 14_lambda_mesanica.png |
| 4.7 Robustnost na velikost | 12_lambda_velikost.py | izpisi/12 |
| 4.7 Test gladkosti | 11b_test_gladkosti.py | izpisi/11b |
| 6 Oddaljenost od 50:50 (omemba) | 13_razmik.py, 14_robustnost_smer_razmik.py, 15_neodvisnost_testov.py | izpisi/13–15 |
| 4.8 Ocena skupnega števila zamenjav | 16_skupno_zaznavnost.py | izpisi/16 |
| 4.9 Druge kontrole | 16_skupno_zaznavnost.py, 20_druge_kontrole.py | izpisi/16, izpisi/20 |
| 5 Vpliv na rezultate | 18_vpliv.py, s01 | rezultati/vpliv.csv, 5_vpliv.png |
| 5.1 Prispevki volišč | s07_prispevki.py | 16_prispevki_volisc.png |
| 5.2 Smer napak (pogojna) | 18_vpliv.py | izpisi/18 |
| 5.2 Smer napak (nepogojna) | 14_robustnost_smer_razmik.py, 19_smer_vsi_outlierji.py, s06 | 15_smer_vsi_outlierji.png |
| 6 Preverljivost | 21_preverjanje.py | rezultati/posteriorne_verjetnosti.csv |

Kode `PP`, `RTV`, `R42`, `R43` v skriptah pomenijo preiskavo, RTV, socialo in tujce.

## Ročni koraki

- **Sostro 25** (Moste-Polje 1/25) je samodejno razvrščen med »več referendumov«; preverjanje vseh prerazporeditev in napovedi iz sociale je v `20_druge_kontrole.py`.
- **Geografija** (razdelek 3.4, mesto/podeželje): ročna ocena po imenih krajev, ni v kodi. Razporeditev po volilnih enotah se razbere iz `rezultati/final_table.csv` (prva števka okraja).
- **Sestrska volišča** so poiskana samodejno po imenu stavbe in naslovu (`17_sestrska_volisca.py`).

## Ujemanje s poročilom

Vse številke v poročilu so izračunane s to kodo. Simulirane vrednosti so odvisne od semena in vrstnega reda vrstic, zato se lahko pri ponovnem zagonu v zadnji decimalki nekoliko razlikujejo; slike se razlikujejo le v posameznih pikslih.
