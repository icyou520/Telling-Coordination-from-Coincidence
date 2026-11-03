# Data

No third-party data are redistributed here. Each set below is public; download it into `data/` as shown. The
simulation experiments (e0–e30 and the simulated checks inside later experiments) need no data. Loaders cache parsed
arrays as `.npz` next to the raw files on first use.

| Folder | Source | License | Used by |
| --- | --- | --- | --- |
| `data/puy2024/` | Puy et al. (2024), schooling fish | CC BY 4.0 | e31–e33, e39–e52, e56, e58, e68, e70, e72, e77 |
| `data/zebrafish/trajectories/` | Heras et al. (2019), zebrafish groups of 60–100 | see source | e38, e53, e58, e68, e70, e72, e77 |
| `data/pigeons/D2/` | Nagy et al. (2013), pigeon flocks | free with citation | e35–e37, e53, e70 |
| `data/steinmetz/` | Steinmetz et al. (2019), mouse Neuropixels | CC BY 4.0 | e61–e63, e67 |
| `data/inagaki/` | Inagaki, Chen et al. (2022), ALM with midbrain silencing, DANDI 000221 | CC BY 4.0 | e64, e65, e70, e71 |
| `data/dandi000006/` | Economo et al. (2018), ALM, DANDI 000006 | CC BY 4.0 | e73 |
| `data/finance/` | Kenneth R. French Data Library; S&P 500 daily prices 2013–2018 | see source | e66, e67, e69, e70, e74–e76, e79 |
| `data/stocks/` | Index members (Wikipedia) and Yahoo Finance prices | see source | e79, e81, e82 |
| `data/stocks_fresh/` | As above, 13 further indices plus S&P 500 outside the S&P 100 | see source | e82 |
| `data/intervention/` | Liu & Burbano (2025); Escobedo et al. (2026) | CC BY 4.0 | e80, e83 (simulated checks) |
| `data/lafoux/` | Lafoux et al. (2023), schools at fixed light levels | CC BY 4.0 | e83 |

## Fish and birds

**Puy et al. (2024)**, *PNAS* 121, e2309733121; Zenodo [10.5281/zenodo.10890112](https://doi.org/10.5281/zenodo.10890112).

```bash
mkdir -p data/puy2024
for f in N=8_recording_{0..5} N=39_recording_{0..1}; do   # the eight recordings used
  curl -L -o "data/puy2024/Experimental_school_$f.csv" "https://zenodo.org/records/10890112/files/Experimental_school_$f.csv?download=1"
done
```

**Heras et al. (2019)**, *PLoS Comput. Biol.* 15, e1007354, tracked with idtracker.ai (Romero-Ferrero et al. 2019).
The trajectory archive is linked from the idtracker.ai data page (https://idtracker.ai, "Data"); unpack it so that
`data/zebrafish/trajectories/{60,80,100}/` hold the `.npy` trajectory files.

**Nagy et al. (2013)**, *PNAS* 110, 13049, https://coordinatedflock.elte.hu/.

```bash
mkdir -p data/pigeons && curl -L -o data/pigeons/trajectoriesD2.zip https://coordinatedflock.elte.hu/data/trajectoriesD2.zip
unzip data/pigeons/trajectoriesD2.zip -d data/pigeons/D2      # flights end up in data/pigeons/D2/flight/ff*/
```

**Liu & Burbano (2025)**, Zenodo [10.5281/zenodo.15351451](https://doi.org/10.5281/zenodo.15351451).

```bash
mkdir -p data/intervention/liu
curl -L -o data/intervention/liu.zip "https://zenodo.org/api/records/15351451/files/swarmintelligencelab%2FIntermittentExploration-v1.0.0.zip/content"
unzip data/intervention/liu.zip -d data/intervention/liu      # gives liu/swarmintelligencelab-IntermittentExploration-2eb7fdd/
```

**Escobedo et al. (2026)**, figshare 30539666: the files listed with their URLs in `data/intervention/vr_urls.txt`
(included; the loader reads the four `.dat` recordings), saved under `data/intervention/vr/`.

```bash
mkdir -p data/intervention/vr && while read f u; do curl -L -o "data/intervention/vr/$f" "$u"; done < data/intervention/vr_urls.txt
```

**Lafoux et al. (2023)**, *Commun. Biol.* 6, 585; figshare 22434919 (fixed-light set).

```bash
mkdir -p data/lafoux && curl -L -o data/lafoux/discrete_light_variation.zip https://ndownloader.figshare.com/files/39883996
unzip data/lafoux/discrete_light_variation.zip -d data/lafoux/discrete
```

## Brains

**Steinmetz et al. (2019)**, *Nature* 576, 266, in Neuromatch Academy's binned form (OSF).

```bash
mkdir -p data/steinmetz
curl -L -o data/steinmetz/steinmetz_part0.npz https://osf.io/agvxh/download
curl -L -o data/steinmetz/steinmetz_part1.npz https://osf.io/uv3mw/download
curl -L -o data/steinmetz/steinmetz_part2.npz https://osf.io/ehmw2/download
```

**Inagaki, Chen et al. (2022)**, *Cell* 185, 1065; [DANDI 000221](https://dandiarchive.org/dandiset/000221). Save each
session as `data/inagaki/<Session_name>.nwb` (for example `HI192_061219.nwb`), with the authors' session table:

```bash
curl -L -o data/inagaki/Session_spreadsheet.csv https://raw.githubusercontent.com/hidehikoinagaki/InagakiAndChenEtAl2022/HEAD/Session_spreadsheet.csv
```

**Economo et al. (2018)**, *Nature* 563, 79; [DANDI 000006](https://dandiarchive.org/dandiset/000006): every `.nwb`
file into `data/dandi000006/`.

## Markets

**Kenneth R. French Data Library** (https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html).

```bash
mkdir -p data/finance && cd data/finance
B=https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp
curl -L -o ind49.zip $B/49_Industry_Portfolios_daily_CSV.zip && unzip ind49.zip     # 49_Industry_Portfolios_Daily.csv
for r in Europe Japan Asia_Pacific_ex_Japan; do curl -L -o ${r}_25_Daily.zip $B/${r}_25_Portfolios_ME_BE-ME_Daily_CSV.zip; done
curl -L -o all_stocks_5yr.csv https://raw.githubusercontent.com/plotly/datasets/master/all_stocks_5yr.csv
```

The French library is updated monthly; the experiments used the files current in October 2026.

**Individual stocks (e79, e81, e82).** Index members were scraped from Wikipedia on the fetch date and are included
(`data/stocks/constituents.json`, `data/stocks_fresh/constituents.json`); prices come from Yahoo Finance:

```bash
uv run python -m experiments.stock_data prices
uv run python -m experiments.stock_data_fresh prices
```

Yahoo revises adjusted closes, so re-downloaded prices will differ slightly from the ones hashed in the
pre-registration log.
