# Overview
This repository contains the source code used for finetuning the LLM phi-2 for programming-tasks. The methodologies used are SFT (supervised fine tuning) and DPO (direct preference optimization). To execute both script it's necessary to install the dependecies specified in the file requirements.txt. The repository also contains two notebooks for testing the performance of the fine-tuned or base model, on the  HumanEval and HumanEval-X benchmarks. Regarding the latter, for computational purposes, only Java, C++ and JavaScript were selected as the languages for generating the samples; furthermore, for each language,  were sampled 80 programming tasks from the corresponding json file of the HumanEval-X dataset (out of 164). Lastly, i tried to implement a simple memory for the LLM, which leverages ChromDB for storing and retrieving the embeddings to insert in the context of the model. Since not all informations retrieved are relevant to the information need explicited through the query by the user to the LLM, an hyperparameter, threshold, was added, in order to filter the least relevant informations.  

