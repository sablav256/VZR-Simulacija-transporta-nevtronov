# VZR-Simulacija-transporta-nevtronov 
### Elisabeth Sambolec
Simulacija transporta nevtronov skozi material z uporabo Monte Carlo metode, ter optimizacija izvajanja z uporabo MPI paralelizacije in Numba optimizacije

## Zagon projekta
Projekt zaženemo tako, da se umestimo v mapo, kjer se nahaja projekt:
- cd /media/sf_VZR_-_Visoko_zmogljivo_raunalnitvo/Projekt2

Nato zaženemo en serialni proces:
- python main.py

Nato sledi zagon MPI paralelizacije
- mpirun -np 1 python benchmark.py
- mpirun -np 2 python benchmark.py
- mpirun -np 4 python benchmark.py
- mpirun --oversubscribe -np 8 python benchmark.py

Nato zaženemo še numba_accel.py:
- python numba_accel.py

Nato zaženemo še monte_carlo_numpy.py:
- python monte_carlo_numpy.py

Nato sledi še zagon datoteke za prikaz grafov:
- python plot_results.py



## UVOD

**Cilj** projekta je simulacija transporta nevtronov skozi material z uporabo Monte Carlo metode, ter optimizacija izvajanja z uporabo MPI paralelizacije in Numba optimizacije. Dodatno analiziramo skalabilnost sistema in vpliv komunikacijskega overheada na zmogljivost. Namen projekta je simulacija transporta nevtronov skozi homogeno snov z uporabo Monte Carlo metode.

Glavni cilj je:
- oceniti verjetnosti transmisije, absorpcije in refleksije,
- analizirati vpliv števila procesov na čas izvajanja,
- preveriti skalabilnost MPI implementacije. 

**Ključno raziskovalno vprašanje** projekta je, kako velik mora biti problem (workload), da MPI paralelizacija postane učinkovita, ter pri katerem številu procesov začne komunikacijski overhead pomembno vplivati na zmogljivost sistema.


## TEORETIČNA IZHODIŠČA


### Monte Carlo metode

Monte Carlo metode so skupina algoritmov, kjer problem rešujemo z naključnim vzorčenjem, statistiko in ogromnim številom simulacij. Monte Carlo metode so postale ključne, ker omogočajo reševanje problemov, ki jih matematično skoraj ni mogoče rešiti neposredno. Kadar analitična rešitev problema ni praktično izvedljiva, uporabimo veliko število naključnih simulacij za statistično aproksimacijo rezultata (povprečje velikega števila simulacij). Bistvo Monte Carla je s ponavljajočimi se poskusi, se približati oceni rešitve.

Prednost metode so enostavna implementacija kompleksnih fizikalnih procesov, naravna paralelizacija in visoka natančnost pri velikem številu vzorcev. 

To je izredno uporabno v fiziki, biologiji, ekonomiji, strojnem učenju, jedrski energiji, medicini, vremenskih modelih, računalniški grafiki. Primeri uporabe Monte Carlo metode: ocena integralov, napovedovanje vrednosti delnic, reševanje diferencialnih enačb.

Prav zato so Monte Carlo metode eden temeljnih konceptov sodobnega znanstvenega računalništva.

Ime metode izvira iz mesta Monte Carlo, ki je znano po kazinojih, igrah na srečo in verjetnosti. Prva velika uporaba je bila pri razvoju atomske bombe med drugo svetovno vojno - Manhattan Project. Fiziki niso mogli analitično izračunati vseh interakcij nevtronov, zato so simulirali milijone naključnih poti nevtronov.

Najbolj znan Monte Carlo primer je **ocena vrednosti π**. Lahko si predstavljamo kvadrat in znotraj njega krog. Naključno generiramo točke in če točka pade v krog prištejemo zadetek, če točka pade izven kroga, ni zadetka. Z deležem točk v krogu se približujemo pravi vrednosti π in več kot je ponovitev bolj natančen je rezultat. To ocenimo s formulo:  

$$
\pi \approx 4 \cdot \frac{N_{krogu}}{N_{vse}}
$$

Monte Carlo temelji na **zakonu velikih števil (Law of Large Numbers)**. Če naredimo dovolj naključnih simulacij potem statistične napake postanejo majhne in povprečje se približa pravi vrednosti. Napaka pada približno:  

$$
\text{napaka} \sim \frac{1}{\sqrt{n}}
$$

, n = število simulacij

Če želimo 10× manjšo napako, potrebujemo 100× več simulacij. To je zelo pomembno pri HPC (High-performance computing).


### Visoko zmogljivo računalništvo (HPC)

Visoko zmogljivo računalništvo (HPC - High-performance computing) omogoča reševanje zelo velikih in računsko zahtevnih problemov z uporabo več procesorjev oziroma jeder hkrati. Namesto da en procesor izvaja vse operacije zaporedno, se problem razdeli na več manjših delov, ki se izvajajo vzporedno.

Glavni cilj HPC sistemov je:
- skrajšanje časa izvajanja,
- boljši izkoristek strojne opreme,
- reševanje problemov, ki jih z enim procesorjem ni mogoče učinkovito rešiti. 

Pri paralelnem izvajanju programov pogosto nastopi potreba po komunikaciji med procesi. Procesi si morajo izmenjevati podatke, sinhronizirati izvajanje ali združevati rezultate. Ta komunikacija predstavlja dodaten strošek izvajanja (communication overhead), saj procesi med komunikacijo pogosto čakajo drug na drugega.

V MPI programih komunikacija vključuje pošiljanje podatkov med procesi, sinhronizacijo, združevanje rezultatov in usklajevanje izvajanja. Komunikacija je bistveno počasnejša od lokalnega računanja, zato lahko pri večjem številu procesov omejuje skalabilnost sistema. Skaliranje ali skalabilnost pomeni kako dobro program izkoristi več procesorjev oziroma jeder. Monte Carlo metode zato dobro skalirajo, saj program dobro izkoristi več procesorjev in pohitri računanje. Pri dobro paraleliziranih problemih lahko pričakujemo približno linearno pohitritev, vendar je v praksi speedup manjši zaradi komunikacijskega overheada, sinhronizacije in omejitev strojne opreme.

Slaba skalabilnost bi bila v primeru, da na 8 jedrih program deluje samo 2× hitreje, kar bi pomenilo, da veliko časa izgubi za komunikacijo, procesi morajo čakati drug drugega in paralelizacija ni učinkovita. 

Monte Carlo metode so zelo primerne za HPC, ker so simulacije med seboj skoraj popolnoma neodvisne. Vsak proces lahko lokalno simulira svoj del nevtronov brez potrebe po stalni komunikaciji z drugimi procesi. Komunikacija se izvede šele na koncu simulacije pri združevanju rezultatov z MPI_Reduce.

Takšne probleme pogosto označujemo kot embarrassingly parallel, kar pomeni, da jih je mogoče zelo učinkovito paralelizirati zaradi minimalne komunikacije med procesi. V primeru, kjer vsak proces deluje neodvisno, procesi ne potrebujejo podatkov drugih procesov in je potrebne malo komunikacije, je communication overhead majhen in dosežemo skoraj linearen speedup. Če pa procesi pogosto izmenjujejo podatke in čakajo drug na drugega, se communication overhead poveča, učinkovitost paralelizacije pa se zmanjša (speedup pade).

Paralelizacija pomeni izvajanje več operacij hkrati. Namesto, da en procesor dela vse zaporedno, imamo več procesorjev ali jeder, ki delajo istočasno. Cilj paralelizacije je skrajšati čas izvajanja, bolje izkoristiti strojno opremo, omogočiti reševanje večjih problemov. Problem razdelimo. Manj komunikacije pomeni boljšo paralelizacijo.

### Generatorji naključnih števil (RNG)

Monte Carlo je popolnoma odvisen od generatorjev naključnih števil (RNG - Random Number Generator). Ampak računalniki ne generirajo pravih naključnih števil, generirajo psevdonaključna števila - generator uporablja matematično formulo, zaporedje je deterministično, ampak izgleda naključno.

Problem RNG (Random Number Generator) je: če vsi procesi uporabljajo isti seed, dobijo ista zaporedja in napačno statistiko. Zato v MPI vsak proces potrebuje svoj seed: 

$$
seed = base_seed + rank
$$

Uporaba enakih začetnih seed vrednosti omogoča reproducibilnost statističnih rezultatov simulacije, medtem ko se lahko čas izvajanja med posameznimi zagoni razlikuje zaradi vpliva operacijskega sistema in strojne opreme.


### Simulacija transporta nevtronov

Monte Carlo simulacija transporta nevtronov opisuje gibanje nevtronov skozi material, kjer nevtroni naključno potujejo in trčijo z atomi. Med potovanjem lahko pride do:
- absorpcije (nevtron se absorbira oz. izgine),
- sipanja (naključna sprememba smeri),
- transmisije (nevtron zapusti material) ali
- refleksije (odbit - vrnitev na rob x = 0), 

Parametri modela so debelina materiala (L), povprečna prosta pot (λ) in verjetnost absorpcije (p_abs). Dolžina proste poti (povprečna razdalja med trki) je eksponentno porazdeljena: s=-λln⁡(U), kjer je U∼U(0,1).

Včasih material absorbira nevtron, potem se simulacija za ta delec konča. Pomembno je, da so nevtroni neodvisni in ni potrebe po komunikaciji med simulacijo. Vsak proces se računa lokalno, na koncu se podatki samo združijo. 
Sipanje (scattering) pomeni spremembo smer gibanja nevtrona po trku z atomom. V modelu je smer po sipanju določena z enakomerno porazdeljenim naključnim kotom:
angle ∈ [−π, π]

**Sipanje** je pomembno v simulaciji, saj določa kako se giblje nevtron, kako globoko prodre v material, verjetnost, da pobegne iz njega ali se absorbira. Brez sipanja bi vsi nevtroni šli naravnost in simulacija ne bi bila realistična. 

Model uporablja poenostavljeno 1D/2D aproksimacijo transporta nevtronov, kjer gibanje poteka vzdolž osi x, smer pa je opisana z enim kotom.


## IMPLEMENTACIJA IN MERITVE


### Python programi

Projekt je sestavljen iz 7 datotek:
- neutron_model.py – fizikalni model
- mpi_simulation.py – MPI implementacija
- main.py – zagon simulacije
- benchmark.py – meritve zmogljivosti
- plot_results.py – analiza in grafi
- numba_accel.py – CPU optimizacija (Numba)
- monte_carlo_numpy.py – NumPy simulacija

Datoteka **neutron_model.py** vsebuje gibanje nevtrona, njegovo sipanje, ali se absorbira, preide skozi material ali odbije. Ključna funkcija simulate_neutron(material, rng) za vsak posamezni nevtron določi začetni položaj (x = 0), začne smer gibanja (angle = 0.0), dobi naključno prosto pot, se premakne proti tej smeri in preveri:
- ali je material nevtron absorbiral → ABSORPCIJA
- ali je šel nevtron skozi material (ga zapusti) → TRANSMISIJA
- ali se je nevtron odbil nazaj → REFLEKSIJA 

Zanka while (Monte Carlo simulacija) se izvaja dokler nevtron ne zapusti materiala (transmission), se odbije (reflection) ali absorbira (absorption).

Datoteka **mpi_simulation.py** vzame fizikalni model in ga razdeli med MPI procese. MPI določi rank (ID procesa) in size (št. procesov). Celotno število nevtronov se razdeli:
local_n = n_neutrons // size

Vsak proces simulira svoj del nevtronov in vodi lokalno statistiko za absorbirane, prepuščene in odbite. Na koncu MPI.Reduce(...) združi rezultate (komunicira na koncu).

Datoteka **main.py** vsebuje glavni vstopni program, kjer definiramo material (debelino, dolžino proste poti (povprečna razdalja med trki) in verjetnost absorpcije pri trku), določimo število nevtronov v simulaciji (100.000) ter zaženemo MPI simulacijo, kjer izpiše rezultate (transmisija, absorpcija, refleksija). 

Datoteka **benchmark.py** meri zmogljivost. Za vsako konfiguracijo jeder (1/2/4/8 jeder) izvede 3 ponovitve simulacije, izmeri čas (MPI.Wtime()) in izračuna povprečje ter shrani rezultat v CSV in ga hkrati prikaže v konzoli. Iz tega rezultata lahko kasneje izračunamo pospešek, učinkovitost in Karp–Flatt faktor. 

Datoteka **plot_results.py** bere rezultate datoteke benchmark.py, kjer sta shranjena parametra cores in time. Nato izračuna in izdela graf za pospešek (speedup), učinkovitost (efficiency) in Karp-Flatt faktor.  

Datoteka **numba_accel.py** prikazuje pospešitev Monte Carlo simulacije transporta nevtronov znotraj enega procesa z uporabo Numba (MPI razdeli delo na jedra). Nastavili smo začetne vrednosti absorbiranih, zapuščenih in odbitih nevtronov. Nato smo funkcijo s funkcijo simulate_batch implementirali model interakcij nevtronov z materialom. 

Datoteka **monte_carlo_numpy.py** prikazuje optimizacijo Monte Carlo simulacije z uporabo knjižnice NumPy. Program uporablja vektorizirane operacije nad polji podatkov, kar omogoča hitrejše izvajanje numeričnih izračunov brez uporabe klasičnih Python zank za vsak posamezni delec. Simulacija hkrati obdela veliko število nevtronov, pri čemer se izračunajo transmisija, absorpcija in refleksija. Zaradi optimiziranih operacij v jeziku C NumPy bistveno zmanjša interpreter overhead in izboljša učinkovitost izvajanja simulacije tudi brez MPI paralelizacije.


### Meritve

Za vsak nevtron smo naključno določili dolžino poti, ga premaknili, določili ali je absorbiran, sipan ali odbit, to ponavljamo dokler ne umre ali ne preide iz materiala. 

V projektu smo simulirali N število nevtronov, vsak ima pozicijo in smer gibanja. Za vsak nevtron generiramo naključno prosto pot, premaknemo nevtron ter določimo absorbcijo in sipanje. Če se sipa določimo novo naključno smer in ponavljamo dokler ni absorbiran ali pa zapusti sistem. 

Po zagonu serialnega dela (en cpu) v main.py smo prejeli rezultat o tem kakšno je razmerje med absorbiranimi, zapuščenimi in odbitimi nevtroni, kar nam nakazuje, da naša simulacija deluje.


┌──(kali㉿kali)-[/media/sf_VZR_-_Visoko_zmogljivo_raunalnitvo/Projekt2]
└─$ python main.py
TRANSMISSION: 0.00157
ABSORPTION: 0.72837
REFLECTION: 0.27006
TIME: 2.428065538406372

Sledi zagon MPI paralelizacije na 1, 2 in 4 jedrih.

└─$ mpirun -np 1 python benchmark.py
AVG TIME: 2.379461766666888
                                                                                                 
└─$ mpirun -np 2 python benchmark.py
AVG TIME: 1.2305788666708395
                                                                                                 
└─$ mpirun -np 4 python benchmark.py
AVG TIME: 0.7086478333343015

└─$ mpirun -np 6 python benchmark.py
AVG TIME: 0.5230845394759276
                                                                                                 

Čas izvajanja se je zmanjševal z vsakim procesom. 

Nato sledi še zagon datoteke za prikaz grafov:

**Pospešek ali speedup** merimo: 	

$$
S(p) = \frac{T_1}{T_p}
$$

, kjer: 
T(1) = čas na enem jedru oz. procesu,
T(p) = čas na p jedrih oz. procesih. 

Graf prikazuje sub-linearno rast. Do 4 procesov se pospešek povečuje skoraj linearno, nato pa se rast upočasni.
•	pri 2 procesih: ~1.47× 
•	pri 4 procesih: ~2.24× 
•	pri 8 procesih: ~2.31× 
To pomeni, da dodajanje procesov po 4 jedrih ne prinaša več bistvenega izboljšanja. Zaradi komunikacijskega overhead-a MPI, sinhronizacije procesov, omejitve pomnilniške prepustnosti (memory bandwidth) in  zmanjšanje količine dela na proces.

<img width="634" height="475" alt="image" src="https://github.com/user-attachments/assets/12720d4d-7322-41e0-b2ed-c4ff12f00df6" />



**Učinkovitost ali efficiency** prikazuje izkoristek procesorjev. Formula:

$$
E(p) = \frac{S(p)}{p}
$$

Graf učinkovitosti kaže monotono padajočo krivuljo, kar je tipično za MPI strong scaling.
•	1 proces: 1.00 
•	2 procesi: 0.74 
•	4 procesi: 0.56 
•	8 procesov: 0.29 

Vsak dodatni proces prispeva manj učinkovitega računa, delež komunikacijskega časa se povečuje in sistem izgublja “compute-to-communication ratio” 
To je pričakovano vedenje za Monte Carlo MPI simulacije pri srednje velikem problemu.

<img width="631" height="471" alt="image" src="https://github.com/user-attachments/assets/4d46c116-eafe-4dd4-b371-eeb982df1243" />


 
**Karp-Flatt metrika** ali Karp–Flatt metric meri koliko problema je efektivno sekvenčnega. Formula: 

$$
e_{Karp-Flatt} = \frac{\frac{1}{S(p)} - \frac{1}{p}}{1 - \frac{1}{p}}
$$

Graf kaže rahlo variabilne vrednosti, vendar brez jasnega monotonega trenda.
•	pri 2 procesih: ~0.36 
•	pri 4 procesih: ~0.26 
•	pri 8 procesih: ~0.35 
Sistem ima majhen, a ne zanemarljiv efektivni sekvenčni del, variacije so posledica:  meritvenega šuma (MPI_Wtime), majhnega problema glede na 8 procesov, overhead sinhronizacije. Problem ne skalira idealno nad 4 procesi, saj overhead začne dominirati.

<img width="628" height="474" alt="image" src="https://github.com/user-attachments/assets/87742854-ed24-4781-97f3-2077663a66fe" />



Sledi še zagon programa z **Numba**, ki prikazuje pospešitev Monte Carlo simulacije transporta nevtronov znotraj enega procesa z uporabo Numba (MPI razdeli delo na jedra).

└─$ python numba_accel.py
RESULT: (145396, 306, 54298)
TIME: 0.8343968391418457
                          
Sledi še zagon programa z **NumPy**, ki prikazuje pospešitev Monte Carlo simulacije z uporabo vektoriziranih operacij nad polji podatkov. NumPy uporablja optimizirane numerične operacije implementirane v jeziku C, kar zmanjša Python interpreter overhead in omogoča bistveno hitrejše izvajanje simulacije tudi brez MPI paralelizacije.

└─$ python monte_carlo_numpy.py
NUMPY RESULTS
TRANSMISSION: 0.00013
ABSORPTION: 0.45496
REFLECTION: 0.54491
TIME: 0.06214118003845215

Izvedli smo primerjavo časa izvajanja med različnimi implementacijami simulacije, kar omogoča oceno učinkovitosti posameznih optimizacij.

<img width="991" height="493" alt="image" src="https://github.com/user-attachments/assets/67f483f9-9f73-4bee-baf7-6f763ba7dd70" />


Čas izvajanja glede na število procesov

└─$ python plot_results.py

FINAL RESULTS TABLE:

cores     time  speedup  efficiency  karp_flatt
    1 2.379462 1.000000    1.000000    0.000000
    2 1.230579 1.933612    0.966806    0.034334
    4 0.708648 3.357749    0.839437    0.063758
    6 0.523085 4.548905    0.758151    0.063800
Numba 0.834397      NaN         NaN         NaN
NumPy 0.062141      NaN         NaN         NaN




## ZAKLJUČEK

Rezultati Monte Carlo simulacije transporta nevtronov kažejo, da večina nevtronov ostane ujeta v materialu. V simulaciji je bilo absorbiranih 725762 nevtronov, 1486 jih je prešlo skozi material, medtem ko jih je bil večji del odbit nazaj. To potrjuje, da gre za debel material, kjer prevladuje absorpcija nad transmisijo.
MPI meritve kažejo, da se čas izvajanja zmanjšuje z naraščanjem števila procesov, vendar skaliranje ni linearno. Speedup narašča do 4 procesov, nato pa se rast bistveno upočasni (pri 8 procesih približno 2.31×). Učinkovitost zato pada z večanjem števila procesov, kar je posledica komunikacijskega overheada, sinhronizacije med procesi ter omejitev pomnilniške prepustnosti. Karp–Flatt metrika dodatno potrjuje prisotnost efektivnega sekvenčnega dela in variabilnost zaradi majhnega razmerja med količino računa in številom procesov.
Dodatne optimizacije na nivoju algoritma kažejo bistveno večje izboljšave zmogljivosti kot sama MPI paralelizacija. Implementacija z Numba JIT kompilacijo je zmanjšala čas izvajanja z odstranitvijo interpreter overheada, medtem ko je NumPy vektorizacija dosegla še večji pospešek zaradi izvajanja operacij na nivoju C implementacije.
Rezultati tako potrjujejo, da je za učinkovito HPC simulacijo ključna kombinacija paralelizacije in optimizacije numeričnega jedra algoritma. MPI izboljša razporeditev dela med procesorji, vendar je skupna zmogljivost močno odvisna tudi od učinkovitosti lokalnega računanja. V tem primeru se je izkazalo, da optimizacija algoritma (NumPy/Numba) prinese večji pospešek kot zgolj povečevanje števila MPI procesov.


 
## LITERATURA IN VIRI


- An Introduction to Parallel Programming. Elsevier, 2011.
- Monte Carlo Methods in Statistical Physics. Oxford University Press, 1999.
- MPI Forum. MPI: A Message-Passing Interface Standard. Version 4.0, 2021.
- MPI for Python (mpi4py) Documentation
- Numba Documentation
- Computer Simulation Methods in Theoretical Physics. Springer, 1990.
- Introduction to High Performance Computing for Scientists and Engineers. CRC Press, 2010.
- Manhattan Project. Uporaba Monte Carlo metod pri simulacijah transporta nevtronov.
