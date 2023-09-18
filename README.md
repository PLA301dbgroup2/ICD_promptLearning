# Pre-trained Language Model + Prompt Project Code

This paper utilizes the code from the project ClinicalPrompt and adapts it for a multi-center dataset from the Chinese People's Liberation Army General Hospital.
The original project directory can be found at https://github.com/NtaylorOX/Public_Clinical_Prompt.


It mainly uses a pre-trained BERT model combined with the prompt method to predict 13 diagnosis and then form their corresponding diagnostic codes.

We also utilized another open-source prompt library, OpenPrompt, which can be found at https://github.com/thunlp/OpenPrompt.

## 1. Data Processing
### 1.1. Data Source
Data acquisition can be requested by contacting the email provided. Due to the sensitivity of the hospital data, it cannot be made publicly available.


### 1.2 Data Processing Flow
I. Raw data table,marked as formatted.csv, which includes discharge summary text and corresponding primary diagnosis label, is processed using pre_process.py to obtain final.csv. final.csv filters out data for 17 types of diseases, reducing the number of original data from 580,000 to a smaller number. Some unstandardized diagnosis names are manually merged.

```
python pre_process.py
```

II. Using process.py, the keyword extraction process of key-bert is executed, and the maximum length is further limited to obtain else.csv.
```
python process.py
```

III. Else.csv is randomly shuffled and split into train, test, and valid datasets. There is no specific method for this, as the samples were selected based on their distribution at the time.

## 3. Training Process
### 3.1 Execute BERT Pre-training
Execute pre.py according to the script. Each line in corpus.txt represents a single input.

```
cd ./src/pretrain

python pre.py
```

### 3.2 Execute Prompt Code Training
```
cd ./src/based-models 

MPLBACKEND=Agg python main.py --model bert --model_name_or_path /home/sr/pretrain/bert --num_epochs 5 --template_id 2 --template_type manual --verbalizer_type soft --verbalizer_id 0 --max_steps 25000 --tune_plm  --run_evaluation
```
After training, the model results are printed in the terminal.
