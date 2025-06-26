# RoAI 2025

Lot Baraj 1 (CV): https://judge.nitro-ai.org/roai-2025/lot-baraj-1/

Lot Baraj 2 (NLP): https://judge.nitro-ai.org/roai-2025/lot-baraj-2/

In timpul contest-urilor, dataset-urile s-au putut descarca de pe http://roai-docs.olimpiada-ai.ro/. In caz de orice, am salvat zip-urile intr-o arhiva pe Google Drive, sa nu se piarda: https://drive.google.com/drive/folders/1CscTLaJADKRQASn3s7UeTFpSbrkNt7TQ?usp=sharing

## Status

| Task | Score  | Type |
| ---- | ------ | ---- |
| 1    | **96** | CV   |
| 2    | **-**  | CV   |
| 3    | **-**  | NLP  |
| 4    | **-**  | NLP  |

## Explanations

### Task 1: [HotSpot](https://judge.nitro-ai.org/roai-2025/lot-baraj-1/problems/1/task)

Summary: Se dau 4 liste de imagini de tip fundal negru + forme geometrice colorate, treptat se adauga si elemente de noise. Trebuie calculate binary segmentation masks doar pentru formele geometrice. 

Solutie 96/100: Nu folosim niciun model, se foloseste doar preprocesare de imagini cu OpenCV. Pentru fiecare imagine facem grayscale, median blur (astfel ajuta la noise + stripes) apoi thresholding (valoare diferita pentru subtask 4). Se proceseaza fiecare imagine, apoi cream RLE si se scrie solutia.

Solutie 100/100: Explicatie coming soon!

### Task 2: [Angry Birds](https://judge.nitro-ai.org/roai-2025/lot-baraj-1/problems/2/task)

Summary: Se da un resnet50 si trebuie finetunat pentru a clasifica 2 tipuri de imagini: pasare de apa sau pasare de pamant. S-au adaugat elemente la dataset pentru a ingreuna: un patrat rosu intr-o pozitie random s-a adaugat la imagini cu pasare de pamant (doar in traning set). Exista imagini cu pasare de apa, pe fundal de pamant/apa si pasare de pamant pe fundal de apa/pamant. Datasetul train este imbalanced pentru aceste tipuri de pasare/fundal. Evaluarea se face pe baza celei mai slabe acuratete dintre cele 4 clase: pasare de apa/pamant pe fundal apa/pamant.

Explicatie coming soon!

### Task 3: [How Toxic Are You Online? ](https://judge.nitro-ai.org/roai-2025/lot-baraj-2/problems/1/task)

Summary: se da un dataset cu comentarii online, trebuie clasificat fiecare comentariu daca este toxic/severe_toxic/obscene/insult.

Explicatie coming soon!

### Task 4: [Skeletons Don’t Lie: Can AI Decode Your Moves?](https://judge.nitro-ai.org/roai-2025/lot-baraj-2/problems/2/task)

Summary: trebuie clasificata pozitia camerei si actiunea executata intr-o serie de video-uri. Un video este reprezentat printr-o serie de coordonate XYZ ale joint-urilor unui om, cate 25 x 3 de coordonate per frame, cu numar variabil de frames per video. PDF-ul [dat](./nlp/skeletons/explicatie.pdf) explica coordonatele date. 

Explicatie coming soon!
