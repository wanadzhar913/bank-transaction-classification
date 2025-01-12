This repo details code for building a **text classifier for predicting Bank Transaction categories**. We finetune a [base version of a DeBERTaV3 model](https://huggingface.co/microsoft/deberta-v3-base) purely on text data, as well as another version using a combination of text and non-text (e.g., categorical, datetime, etc.) data. My approach and methodology are detailed below.

# Methodology
### 1.0 Feature Engineering
Text-related pre-processing
1. We first remove rows with null values in the `category` column from the *bank_transaction.csv*
2. We then pre-process our dataset using the `stem_text()` function which performs: **Non-Letter Removal:**, **Case Normalization:**,**Tokenization:**, **Stopword Removal:**, **Stemming:**. We do this to identify transactions comprised of only stop words and non-alphanumerics (which we then remove as they are not meaningful).

Transaction-level features
1. We create day related features to identify if the transaction happened on Friday, Monday or the weekend based on our analysis above.
2. We also choose to keep the amount spent on the transaction. Sizeable amounts may be correlated to specific transaction categories.

User-level features  
1. From `bank_transaction.csv`, we create aggregate features to determine, on a monthly basis, how likely is a consumer make a transaciton of a particular category (and overall to determine activeness) based on their count (in the future, perhaps amount maybe better). This could potentially be an important feature if to determine if, for a particular user, a transacition category is more likely (in addition to the *description* column). We acknowledge that this may result in a sparse dataset, but it's a good first step for experimentation.
2. From `user_profile.csv`, we adopt the intent columns in spite of their sparsity for experimentation.

Train-test Split  
1. 20% of the original dataset will be set aside and used as a test set. This will be useful in evaluating our model's performance on unseen data.
2. We will also use stratification to preserve the class representation in our train and test set.

### 2.0 Model Architecture

# Results



# Next Steps & Optimizations

# Finetuned Models and Usage
The finetuned models (and how to use them) are available in the below links.

- DeBERTaV3 model finetuned on **text only**: https://huggingface.co/wanadzhar913/debertav3-finetuned-banking-transaction-classification-text-only
- DeBERTaV3 model finetuned on **text and non-text features**: https://huggingface.co/wanadzhar913/debertav3-finetuned-banking-transaction-classification

# Resources Used
- https://huggingface.co/microsoft/deberta-v3-base
- https://huggingface.co/docs/datasets/en/process
- https://discuss.huggingface.co/t/combine-bertforsequenceclassificaion-with-additional-features/6523
- https://colab.research.google.com/drive/1ZLfcB16Et9U2V-udrw8zwrfChFCIhomz?usp=sharing