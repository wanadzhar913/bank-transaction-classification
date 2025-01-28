This repo details code for building a **text classifier for predicting Bank Transaction categories**. We finetune a [base version of a DeBERTaV3 model](https://huggingface.co/microsoft/deberta-v3-base) purely on text data, as well as another version using a combination of text and non-text (e.g., categorical, datetime, etc.) data. My approach and methodology are detailed below.

# Model Files and Usage  

The finetuned models (and how to use them) are available in the below links.

- DeBERTaV3 model finetuned on **text only**: https://huggingface.co/wanadzhar913/debertav3-finetuned-banking-transaction-classification-text-only
- DeBERTaV3 model finetuned on **text and non-text features**: https://huggingface.co/wanadzhar913/debertav3-finetuned-banking-transaction-classification

# Methodology
### 1.0 Feature Engineering
**Text-related pre-processing**  

1. We first removed rows with null values in the `category` column from the *bank_transaction.csv*
2. We then pre-processed our dataset using the `stem_text()` function which performs: **Non-Letter Removal:**, **Case Normalization:**,**Tokenization:**, **Stopword Removal:**, **Stemming:**. We did this to identify transactions comprised of only stop words, non-alphanumerics and numbers-only (which we then remove as they are not meaningful).

**Transaction-level features**  

1. We then created day related features to identify if the transaction happened on Friday, Monday or the weekend based on our EDA.
2. We also chose to keep the amount spent column on the transaction. Sizeable amounts may be correlated to specific transaction categories.

**User-level features**  

1. From *bank_transaction.csv*, we created aggregate features to determine, on a monthly basis, how likely a consumer is to make a transaciton of a particular category based on their count. This can potentially capture a transaction category a user has a procilivity for. We acknowledge that this may result in a sparse dataset, but it's a good first step for experimentation.
2. From *user_profile.csv*, we included the intent columns in spite of their sparsity for experimentation.

**Train-test Split**  

1. 20% of the original dataset was set aside and used as a test set. This was useful in evaluating our model's performance on unseen data.
2. We also used stratification to preserve the class representation in our train and test set. However, due to the imbalanced nature of the categories in our dataset, we've had cases where there's only 1 instance of a category in our test set e.g., *Tax Refund*.

### 2.0 Model Architecture
**Why DeBERTaV3?**   
We chose the latest version of DeBERTa due to it's [SOTA (State-of-the-Art) performance on NLU (Natural Language Understanding) tasks and benchmarks](
https://huggingface.co/microsoft/deberta-v3-base#fine-tuning-on-nlu-tasks). Moreover, with a **vocabulary of 128k and only having 86 million backbone parameters**, it is relatively efficient to finetune which is good due to our compute constraints.

**Training Loop**  
We trained 2 models using **Google Colab Pro's A100 GPU (40GB VRAM)**. One model takes in text primarily as input, and the other takes text and additional non-text features. Both were trained using **PyTorch** and had the following training procedures:

1. Learning Rate Scheduler
    - `ReduceLROnPlateau`: To dynamically reduce the learning rate when the F1 score plateaus, helping the model converge without overfitting.
2. Gradient Clipping
    - `torch.nn.utils.clip_grad_norm_`: To limit the gradient norm to 5.0, preventing exploding gradients and stabilizing training.
3. Early Stopping
    - `current_patient` and `patient`: To stop training if validation F1 score doesn’t improve for 3 consecutive epochs, reducing overfitting from excessive training.
4. Validation Monitoring
    - Regular evaluation on validation data (test_X and test_Y) was done to ensure overfitting was detected early. 
5. Weight Decay via AdamW
    - Optimizer includes weight decay, indirectly acting as L2 regularization to reduce overfitting.

For Early Stopping, we based it on the **weighted F1-Score**. We do this for several reasons:

1. F1 score is the harmonic mean of precision and recall, meaning it considers both the ability to correctly identify positive cases (precision) and the ability to identify all relevant cases (recall). In our case, both false positives and false negatives impact our ability to give sound financial advice which is why we opt for a metric that balances both of them.
    - A false positive, for instance, could occur if a transaction is incorrectly labeled as Payroll when it actually belongs to the Loans category. This misclassification might inflate a user’s perceived income, leading to poor financial advice, such as suggesting they have enough income to cover their expenses when they actually do not.
    - Similarly, a false negative could occur if a transaction belonging to the Payroll category is mislabeled as something else, such as Miscellaneous. This would result in underestimating the user’s income, potentially causing unnecessary warnings about their financial health or debt levels.
    - Hence, the F1-score balances precision and recall, addressing these risks by considering both types of errors. Precision ensures that when we classify a transaction as a particular category, it is highly likely to belong there (minimizing false positives), while recall ensures we correctly identify as many relevant transactions as possible (minimizing false negatives). This balance is crucial for delivering accurate financial insights and avoiding advice that could mislead or harm the user.
2. As for why we opted for **weighted** instead of the default **macro** F1 (average of individual F1 scores without weights), a macro-average could overly penalize the model for poor performance on very small classes, even if it performs well overall (especially across more common transaction categories). **However, an argument can be made for the macro version if we need to ensure our classifier accounts for the smaller classes** just as well. 

For handling **non-text data**, we also created a custom class, `CustomSequenceClassification()`, with it's own `ClassificationHead()`. We employ the following techniques:

1. Combining Text & Non-text Features
    - Concatenates [CLS] embeddings with `extra_data` to utilize both textual and auxiliary features effectively.
2. Classification Head
    - We include a simple head with a dense layer, non-linearity (*tanh*), dropout, and projection for classification.

# Results
| Model                        | Epoch | Learning Rate | Grad Norm   | Training Loss | Validation Loss | Accuracy  | F1 Score (Weighted) | F1 Score (Macro) | Precision (Weighted) | Recall (Weighted) |
|------------------------------|-------|---------------|-------------|---------------|-----------------|-----------|---------------------|------------------|----------------------|-------------------|
| DeBERTaV3 (Text Only)        | 10    | 0.0000050     | 3.755       | 0.102         | 0.309           | 0.913     | 0.914               | 0.858            | 0.918                | 0.913             |
| DeBERTaV3 (Text & Non-text)  | 15    | 0.0000100     | 264.891     | 0.441         | 0.494           | 0.883     | 0.908               | 0.604            | 0.948                | 0.883             |

From the above results, **our text-only model dominates**, especially when comparing Macro F1. When looking at categories where there was 0 were correctly identified, the text-only model only has 1 (*Tax Refund*) while the text and non-text model has 4 (*Bank Fee*, *Interest*, *Payment*, *Tax Refund*). Additionally, all but 2 categories for the text-only model have less than 70% F1 Score for every class.

The difference in results may be due to the **sparse nature of our additional features**. For example, the user-level features (`IS_INTERESTED_{category}`) are all highly imbalanced with less than 10% are actually interested in each category. Hence, more data exploration and feature engineering work needs to be done to improve the text and non-text model.

# Next Steps & Improvements  

1. For improving the DeBERTaV3 (Text & Non-text) model
    - Add batch normalization after our dense layer extra stability. This is especially the case as our gradients for the Text & Non-text model were quite big.
    - Normalizing & scaling `extra_data` (especially the `amounts` & `monthly_transaction_count_{category}` column which have a lot of outliers) before concatenation for smoother training.
    - We can train different models that either **discard/adopt the user-level features** e.g., `monthly_transaction_count_{category}` & `IS_INTERESTED_{category}` or create features based on monthly transaction amount.
2. However, it may be **wise to focus on iterating on the text-only model** due to its success.
3. The data we have at the moment **only encompasses 3 months**. More data (1 month to 3 months on) can potentially improve our model (though it can also introduce additional complexity due to transactions that are popular during different times of the year e.g., holidays, festivals, etc.) as this will also allow us to do **resampling**. For example we can downsample overrepresented classes such as *Uncategorized* and upsample smaller classes like *Gyms and Fitness Centers*, *Tax Refund*, etc.
4. In the training loop, we **ought to also monitor performance metrics on the Training data to identify overfitting**. Additionally, **StratifiedKFold can also be used to provide a more reliable estimate of a model's performance** by training and evaluating it on multiple different subsets of the data, effectively reducing the bias introduced by a single train-test split and preventing overfitting. MLOps tools like Weights & Biases can help with this.
5. Explore **different architectures** e.g., [ModernBERT](https://huggingface.co/blog/modernbert), Classical Bag-of-Words models, etc.

# Resources  

- https://datascience.stackexchange.com/questions/40900/whats-the-difference-between-sklearn-f1-score-micro-and-weighted-for-a-mult
- https://huggingface.co/microsoft/deberta-v3-base
- https://arxiv.org/abs/2111.09543
- https://huggingface.co/docs/datasets/en/process
- https://discuss.huggingface.co/t/combine-bertforsequenceclassificaion-with-additional-features/6523
- https://colab.research.google.com/drive/1ZLfcB16Et9U2V-udrw8zwrfChFCIhomz?usp=sharing
- https://www.pinecone.io/learn/batch-layer-normalization/
- https://www.geeksforgeeks.org/stratified-k-fold-cross-validation/

