# Simulare OJIA 2

Link: https://judge.nitro-ai.org/roai-2025/simulare-ojia-2

## Predicția statusului pacientului: 100p

- Subtask 1: np.select + filtrare
- Subtask 2: np.quantile + filtrare si np.select
- Subtask 3: .median() si o comparatie
- Subtask 4: se face un df mic cu value_counts() ale T Stage, apoi cu .map() se coreleaza cu cele din test df
- Subtask 5:

1. Feature engineering

- se inlocuiesc valorile missing cu mean-ul (Tumor Size)
- se scot featurile logic irelevante (culoarea pielii sidaca e casatorit)
- pentru T si N stages, se face un mapping cu valori ordonate (! relevant pentru o precizie buna)
- pentru restul, dummy encoding
- ulterior, feature-urile cu corelatia in [-0.003, 0.003] se scot

2. Model

- un model random forest cu 350 estimatori intra

## Predicția prețului unei case
