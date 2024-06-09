# Deciphering Financial Documents: A Classification Journey

## Overview
This project aims to classify tables from financial statements into five categories: Income Statements, Balance Sheets, Cash Flows, Notes, and Others. It utilizes machine learning techniques and advanced feature engineering to analyze and classify the text data extracted from HTML files containing financial statement tables.

## Package Used
- numpy
- pandas
- scikit-learn
- nltk
- wordcloud
- matplotlib

## Folder Structure
- **Folder: Classified Word Clouds**
  - Contains Word Cloud images of Income Statements, Balance Sheets, Cash Flows, Notes and Others.
- **Folder: Data:** Contains the dataset and processed files.
  - *Income Statements, Balance Sheets, Cash Flows, Notes, Others:* Each folder contains HTML files with financial statement tables.
- **Folder: Models:** Contains pickled models used for classification.
  - *Vectorization_model:* Model for text vectorization.
  - *ClassifyingFinancialStatements_model:* Model for label encoding.
  - *Financial_Statements_model:* Pre-trained classification model.
- **Folder: Report**
  - *file: Financial_Document_Classification_Report.pdf* (Report summarizing the approach, model selection, results, and conclusion)
  - *image: ROC Curves for each class.png* (ROC curve image)
- **Classification_Model_For_Financial_Statements.ipynb:** Jupyter Notebook containing the code for the classification model.
- **README.md:** Project overview, approach, results, and conclusion.

## Approach
1. **Data Extraction and Preprocessing:**
   - HTML files containing tabular data were processed to extract relevant information.
   - Initial preprocessing steps included cleaning the extracted text to remove unnecessary characters and formatting.

2. **Feature Engineering:**
   - Text Vectorization: Text data underwent vectorization using TF-IDF representation.
   - Label Encoding: Document labels were encoded into numerical format for model training.
   - Word Count: A feature representing the total number of words in the document was added.
   - Word Cloud: A visual representation of the most common words in the documents was created to understand prevalent terms.

3. **Model Selection and Training:**
   - Various classification models were evaluated using the LazyClassifier library to identify the most suitable model.
   - The Support Vector Classifier (SVC) exhibited the best performance based on accuracy and generalization to new data.

4. **Model Evaluation:**
   - The trained SVC model was evaluated using standard performance metrics such as accuracy, precision, recall, and F1-score.
   - Additionally, the ROC AUC score was calculated to assess the model's ability to distinguish between classes.

## Model Selection
We chose the Support Vector Classifier (SVC) due to its superior performance, particularly in accuracy and generalization. While models like LGBMClassifier and ExtraTreesClassifier showed higher training accuracies, their test accuracies were slightly lower. For instance, **LGBMClassifier** achieved **99.39%** training accuracy but only **92.05%** test accuracy, while **ExtraTreesClassifier** had **99.39%** training accuracy and **92.72%** test accuracy. Conversely, **SVC** had a slightly lower training accuracy at **96.85%** but maintained comparable test accuracy at **92.05%**. What distinguished SVC was its ability to generalize well to unseen data, evident in the narrow margin between its training and test accuracies. With a training accuracy of **96.85%** and test accuracy of **92.05%**, SVC showed **minimal risk of overfitting** and superior generalization compared to other models. Hence, owing to its balanced performance metrics and lower risk of overfitting, SVC emerged as the optimal choice for financial document classification.

## Results
1. **Accuracy:** The SVC model achieved an overall accuracy of **92.05%** on the test set, indicating precise classification of **92.05%** of test samples.
2. **Precision, Recall, and F1-Score:** The average precision, recall, and F1-score were **93.41%**, **93.78%**, and **93.56%**, respectively, demonstrating strong classification capabilities across all document types.
3. **Classification Report (in %):**
   | Metric          | Balance Sheets | Cash Flow | Income Statement | Notes | Others |
   |-----------------|----------------|-----------|------------------|-------|--------|
   | Precision       | **98.48**      | **93.75** | **93.67**        | **90.07** | **91.10**  |
   | Recall          | **97.01**      | **100.00**| **91.36**        | **87.18** | **93.33**  |
   | F1-Score        | **97.74**      | **96.77** | **92.50**        | **88.60** | **92.20**  |
4. **ROC AUC Score:** The ROC AUC score of **98.7%** indicated excellent ability to distinguish between different document types.
![ROC Curve](https://github.com/GDharan10/Project8_ClassificationModelForFinancialStatements/blob/main/Report/ROC%20Curves%20for%20each%20class.png)


## Conclusion
The Support Vector Classifier (SVC) model achieved accurate classification of financial statement tables into their respective categories. With a balanced trade-off between accuracy, precision, recall, and generalization, the SVC model proves to be effective and reliable for this classification task. Future enhancements could involve fine-tuning the model parameters or exploring ensemble techniques to further improve performance.

## Word Cloud
### Income Statements
![Income Statements Word Cloud](https://github.com/GDharan10/Project8_ClassificationModelForFinancialStatements/blob/main/Classified%20Word%20Clouds/Balance%20Sheets.png)

### Balance Sheets
![Balance Sheets Word Cloud](https://github.com/GDharan10/Project8_ClassificationModelForFinancialStatements/blob/main/Classified%20Word%20Clouds/Cash%20Flow.png
)

### Cash Flows
![Cash Flows Word Cloud](https://github.com/GDharan10/Project8_ClassificationModelForFinancialStatements/blob/main/Classified%20Word%20Clouds/Income%20Statement.png)

### Notes
![Notes Word Cloud](https://github.com/GDharan10/Project8_ClassificationModelForFinancialStatements/blob/main/Classified%20Word%20Clouds/Notes.png)

### Others
![Others Word Cloud](https://github.com/GDharan10/Project8_ClassificationModelForFinancialStatements/blob/main/Classified%20Word%20Clouds/Others.png)

In summary, This project demonstrates a robust approach to classifying financial statement documents, utilizing advanced feature engineering and machine learning techniques to achieve high accuracy and reliability in predictions.
