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
- mpirun --oversubscribe -np 4 python benchmark.py
- mpirun --oversubscribe -np 8 python benchmark.py

Nato sledi še zagon datoteke za prikaz grafov:
- python plot_results.py

Nato zaženemo še numba_accel.py:
- python numba_accel.py

 
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

Projekt je sestavljen iz 6 datotek:
- neutron_model.py – fizikalni model
- mpi_simulation.py – MPI implementacija
- main.py – zagon simulacije
- benchmark.py – meritve zmogljivosti
- plot_results.py – analiza in grafi
- numba_accel.py – CPU optimizacija (Numba)

Datoteka neutron_model.py vsebuje gibanje nevtrona, njegovo sipanje, ali se absorbira, preide skozi material ali odbije. Ključna funkcija simulate_neutron(material, rng) za vsak posamezni nevtron določi začetni položaj (x = 0), začne smer gibanja (angle = 0.0), dobi naključno prosto pot, se premakne proti tej smeri in preveri:
- ali je material nevtron absorbiral → ABSORPCIJA
- ali je šel nevtron skozi material (ga zapusti) → TRANSMISIJA
- ali se je nevtron odbil nazaj → REFLEKSIJA 

Zanka while (Monte Carlo simulacija) se izvaja dokler nevtron ne zapusti materiala (transmission), se odbije (reflection) ali absorbira (absorption).

Datoteka mpi_simulation.py vzame fizikalni model in ga razdeli med MPI procese. MPI določi rank (ID procesa) in size (št. procesov). Celotno število nevtronov se razdeli:
local_n = n_neutrons // size

Vsak proces simulira svoj del nevtronov in vodi lokalno statistiko za absorbirane, prepuščene in odbite. Na koncu MPI.Reduce(...) združi rezultate (komunicira na koncu).

Datoteka **main.py** vsebuje glavni vstopni program, kjer definiramo material (debelino, dolžino proste poti (povprečna razdalja med trki) in verjetnost absorpcije pri trku), določimo število nevtronov v simulaciji (100.000) ter zaženemo MPI simulacijo, kjer izpiše rezultate (transmisija, absorpcija, refleksija). 

Datoteka **benchmark.py** meri zmogljivost. Za vsako konfiguracijo jeder (1/2/4/8 jeder) izvede 3 ponovitve simulacije, izmeri čas (MPI.Wtime()) in izračuna povprečje ter shrani rezultat v CSV in ga hkrati prikaže v konzoli. Iz tega rezultata lahko kasneje izračunamo pospešek, učinkovitost in Karp–Flatt faktor. 

Datoteka **plot_results.py** bere rezultate datoteke benchmark.py, kjer sta shranjena parametra cores in time. Nato izračuna: pospešek (speedup), učinkovitost (efficiency) in Karp-Flatt faktor. 

Po izračunu program še generira graf za pospešek in učinkovitost.

Datoteka **numba_accel.py** prikazuje pospešitev Monte Carlo simulacije transporta nevtronov znotraj enega procesa z uporabo Numba (MPI razdeli delo na jedra). Nastavili smo začetne vrednosti absorbiranih, zapuščenih in odbitih nevtronov. Nato smo funkcijo s funkcijo simulate_batch implementirali model interakcij nevtronov z materialom. 

### Meritve

Za vsak nevtron smo naključno določili dolžino poti, ga premaknili, določili ali je absorbiran, sipan ali odbit, to ponavljamo dokler ne umre ali ne preide iz materiala. 

V projektu smo simulirali N število nevtronov, vsak ima pozicijo in smer gibanja. Za vsak nevtron generiramo naključno prosto pot, premaknemo nevtron ter določimo absorbcijo in sipanje. Če se sipa določimo novo naključno smer in ponavljamo dokler ni absorbiran ali pa zapusti sistem. 

Po zagonu serialnega dela (en cpu) v main.py smo prejeli rezultat o tem kakšno je razmerje med absorbiranimi, zapuščenimi in odbitimi nevtroni, kar nam nakazuje, da naša simulacija deluje.
 
<img width="605" height="127" alt="image" src="https://github.com/user-attachments/assets/a57075d6-82ab-4d23-bc17-e34358560441" />


Sledi zagon MPI paralelizacije na 1, 2, 4 in 8 jedrih.
 <img width="605" height="240" alt="image" src="https://github.com/user-attachments/assets/a724023a-7b84-419d-91c7-a295522ee9f6" />

Čas izvajanja se je zmanjšal pri uporabi 2 procesov, pri večjem številu procesov pa se je zaradi overheada ponovno povečal.

Nato sledi še zagon datoteke za prikaz grafov:
 <img width="605" height="159" alt="image" src="https://github.com/user-attachments/assets/4f7b5493-5216-4dfc-b5fa-fa6f30dc962b" />


**Pospešek ali speedup** merimo: 	

$$
S(p) = \frac{T_1}{T_p}
$$

, kjer: 
T(1) = čas na enem jedru oz. procesu,
T(p) = čas na p jedrih oz. procesih. 

Idealni speedup bi bil, če bi 2 jedri opravili nalogo 2x hitreje, kot eno jedro, 4 jedra 4x hitreje in 8 jeder 8x hitreje kot 1 jedro. V praksi pospešek nikoli ni popoln, zaradi MPI overhead, process startup, synchronization, cache misses, memory bandwidth, OS scheduling, reduction cost.
 
<img width="306" height="230" alt="image" src="https://github.com/user-attachments/assets/b7a8ff85-1f6b-4bc3-b057-1ecf01f4a960" />

Graf prikazuje rast speedup-a glede na število procesov. Največji speedup je dosežen pri 2 jedrih oz. procesih. Po tej točki dodatno povečevanje števila procesov ne izboljša več izvajanja.  Glavni razlog je komunikacijski overhead MPI, neenakomerna delitev dela (remainder) in omejitve Amdahlovega zakona. 

To kaže, da sistem doseže mejo učinkovite paralelizacije, komunikacijski overhead postane pomemben, procesi začnejo tekmovati za iste CPU vire. 

**Učinkovitost ali efficiency** prikazuje izkoristek procesorjev. Formula:

$$
E(p) = \frac{S(p)}{p}
$$

Največja učinkovitost je pri majhnem številu procesov, z večanjem števila procesov oz. jeder učinkovitost pada. To je tipično za Monte Carlo MPI sisteme, kjer je računanje sicer paralelno, vendar komunikacija in sinhronizacija nista brez stroška. To pomeni, da dodatni procesi niso več učinkovito izkoriščeni, velik del časa se porabi za MPI sinhronizacijo, scheduling in komunikacijski overhead. 

<img width="280" height="208" alt="image" src="https://github.com/user-attachments/assets/9b09da13-0609-4265-8334-70cfb8d5353e" />

**Karp-Flatt metrika** ali Karp–Flatt metric meri koliko problema je efektivno sekvenčnega. Formula: 

$$
e_{Karp-Flatt} = \frac{\frac{1}{S(p)} - \frac{1}{p}}{1 - \frac{1}{p}}
$$

Pri svoji nalogi pričakujemo majhen e, ker komunikacije skoraj ni. Iz grafa je razvidno, da se z večanjem jeder veča komunikacijski overhead (sekvenčni del) ter zmanjšuje učinkovitost paralelizma. 
<img width="312" height="233" alt="image" src="https://github.com/user-attachments/assets/9b99cd31-d5f2-4e13-9c01-4e252ae7883a" />


Sledi še zagon programa z **Numba**, ki prikazuje pospešitev Monte Carlo simulacije transporta nevtronov znotraj enega procesa z uporabo Numba (MPI razdeli delo na jedra).
<img width="580" height="78" alt="image" src="https://github.com/user-attachments/assets/1427a932-6f05-40f2-b301-89c574db8b2f" />


**Rezultat** prikazuje, da je bilo v simulaciji absorbiranih (absorbed) 725762 nevtronov, prešlo je (transmitted) 1486 nevtronov in odbitih (reflected) je bilo 272.752 nevtronov. 

Rezultati nam povedo, da gre za debel material, kjer se večina nevtronov ujame v material (absorpcija), del se odbije nazaj (refleksija), zelo malo jih preide skozi (transmitira). Numba je zdaj pospešil notranjo zanko brez MPI in realni physics model. Rezultati kažejo znatno zmanjšanje časa izvajanja v primerjavi s standardno Python implementacijo, saj je bil čas izvajanja z uporabo Numba približno 2,5× hitrejši (S=7.392791/2.921173≈2.53) od osnovne Python implementacije.


## ZAKLJUČEK


Projekt uspešno prikazuje uporabo Monte Carlo metode za simulacijo transporta nevtronov skozi material ter uporabo MPI paralelizacije in Numba optimizacije za pohitritev izvajanja. Simulacijo smo izvajali v okolju Oracle VM. Rezultati kažejo, da so Monte Carlo simulacije zelo primerne za paralelizacijo, saj so posamezne simulacije med seboj skoraj neodvisne in zahtevajo zelo malo komunikacije med procesi.

Pri manjšem številu MPI procesov smo dosegli opazno pohitritev izvajanja, pri večjem številu procesov pa se je učinkovitost zmanjšala zaradi komunikacijskega overheada, sinhronizacije procesov, omejitev virtualnega okolja in oversubscribe izvajanja. Pri 8 procesih je število MPI procesov preseglo število fizičnih jeder, zato je prišlo do preklapljanja procesov (context switching), kar je dodatno zmanjšalo učinkovitost paralelizacije.

Čas izvajanja glede na število procesov
<img width="750" height="198" alt="image" src="https://github.com/user-attachments/assets/a53c65de-c758-48b7-86db-403fb38f70c7" />

Rezultati kažejo, da največjo pohitritev dosežemo pri 2 procesih, nato pa se zaradi overheada in omejitev sistema skalabilnost začne zmanjševati. Speedup zato ni linearen, učinkovitost pa z večanjem števila procesov pada.
Meritve speedupa, učinkovitosti (efficiency) in Karp–Flatt metrike potrjujejo praktične omejitve paralelnega izvajanja:
- komunikacijski overhead omejuje skaliranje,
- sinhronizacija procesov povzroča dodatne zakasnitve,
- stohastična narava Monte Carlo simulacije povzroča variabilnost izvajanja,
- del kode ostaja serijski (združevanje rezultatov). 

Kljub temu rezultati potrjujejo, da Monte Carlo metode predstavljajo zelo primeren tip problema za paralelno računalništvo, saj večino časa porabijo za lokalno računanje in potrebujejo minimalno MPI komunikacijo.
Dodatno smo z uporabo Numba optimizacije uspešno pospešili izvajanje simulacije znotraj posameznega procesa. Čas izvajanja z Numba je bil bistveno krajši od standardne Python implementacije, kar kaže prednosti JIT kompilacije pri numerično zahtevnih simulacijah.

Projekt tako uspešno demonstrira osnovne principe visoko zmogljivega računalništva (HPC), MPI paralelizacije, Monte Carlo metod ter vpliv komunikacijskega overheada na skalabilnost sistema.

Monte Carlo simulacije transporta nevtronov so zaradi neodvisnosti posameznih simulacij zelo primerne za paralelizacijo, vendar v praksi popolno linearno skaliranje omejujejo komunikacijski overhead, sinhronizacija procesov in omejitve strojne opreme
 
## LITERATURA IN VIRI


- An Introduction to Parallel Programming. Elsevier, 2011.
- Monte Carlo Methods in Statistical Physics. Oxford University Press, 1999.
- MPI Forum. MPI: A Message-Passing Interface Standard. Version 4.0, 2021.
- MPI for Python (mpi4py) Documentation
- Numba Documentation
- Computer Simulation Methods in Theoretical Physics. Springer, 1990.
- Introduction to High Performance Computing for Scientists and Engineers. CRC Press, 2010.
- Manhattan Project. Uporaba Monte Carlo metod pri simulacijah transporta nevtronov.
