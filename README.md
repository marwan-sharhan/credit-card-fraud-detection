\# Credit Card Fraud Detection

\### كشف الاحتيال في معاملات بطاقات الائتمان



A machine learning project for detecting fraudulent credit card transactions using supervised classification, anomaly detection, feature engineering, and evaluation under severe class imbalance.



مشروع تعلم آلي لاكتشاف معاملات بطاقات الائتمان الاحتيالية باستخدام التصنيف الخاضع للإشراف، واكتشاف الشذوذ، وهندسة الخصائص، وتقييم النماذج في ظل عدم توازن شديد بين الفئات.



\---



\## Table of Contents | فهرس المحتويات



\- \[English](#english)

\- \[العربية](#العربية)

\- \[Project Structure](#project-structure--هيكل-المشروع)

\- \[Limitations](#limitations--القيود)



\---



<a id="english"></a>

\## English



\### 1. Project Overview



Credit card fraud causes financial losses and requires reliable detection methods. This project builds and evaluates machine learning models that distinguish fraudulent transactions from legitimate ones.



The dataset is highly imbalanced: fraudulent transactions represent only a small fraction of all observations. Therefore, accuracy alone is insufficient for evaluating model quality.



\### 2. Objectives



\- Prepare and validate transaction data.

\- Explore class imbalance and transaction characteristics.

\- Engineer additional numerical features.

\- Train supervised classification models.

\- Evaluate an Isolation Forest anomaly-detection baseline.

\- Select a classification threshold using validation data.

\- Evaluate final performance on a separate chronological test set.

\- Save the selected model and its metadata.



\### 3. Dataset



The project uses the `creditcard.csv` dataset, commonly distributed through Kaggle and originally associated with European cardholder transactions.



The original dataset contains 284,807 transactions and 31 columns, including `Time`, `Amount`, anonymized features `V1`–`V28`, and the target `Class`.



\- `Class = 0`: legitimate transaction.

\- `Class = 1`: fraudulent transaction.



\*\*Dataset access:\*\* \[Kaggle — Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)



Check the dataset's current license and terms before redistributing the data. The raw dataset is not included in this repository.



\### 4. Methodology



The implemented workflow includes:



1\. Data quality checks.

2\. Duplicate-row handling.

3\. Chronological train, validation, and test splitting.

4\. Feature engineering.

5\. Training supervised models:

&#x20;  - Logistic Regression

&#x20;  - Random Forest

&#x20;  - HistGradientBoosting

6\. Training an Isolation Forest baseline.

7\. Selecting the Random Forest classification threshold on validation data.

8\. Evaluating the selected threshold on the held-out test set.

9\. Saving the trained model and metadata.



The chronological split is intended to reduce temporal leakage by evaluating later transactions after training on earlier transactions.



\### 5. Feature Engineering



Nine additional features were created:



\- `Amount\_Log1p`

\- `Time\_Days`

\- `V\_Mean`

\- `V\_Std`

\- `V\_Mean\_Abs`

\- `V\_Max\_Abs`

\- `V\_L1\_Norm`

\- `V\_L2\_Norm`

\- `Amount\_Robust\_Score`



The exact feature transformations and model pipeline are documented in the notebook.



\### 6. Test Results



The following results were obtained on the held-out chronological test set using the recorded experiment configuration.



| Model | Precision | Recall | F1-score | Average Precision | ROC-AUC |

|---|---:|---:|---:|---:|---:|

| Logistic Regression | 1.0000 | 0.6346 | 0.7765 | 0.7703 | 0.9716 |

| Random Forest | 0.8261 | 0.7308 | 0.7755 | 0.7802 | 0.9291 |

| HistGradientBoosting | 0.7872 | 0.7115 | 0.7475 | 0.7578 | 0.9749 |

| Isolation Forest | 0.0665 | 0.5385 | 0.1184 | 0.0480 | 0.9297 |



\*\*Selected Random Forest threshold:\*\* `0.193001`



The threshold was selected using validation data, not the test set. At this threshold, the test results were:



\- True Positives (TP): 38

\- False Positives (FP): 8

\- False Negatives (FN): 14

\- True Negatives (TN): 42,501



The Random Forest detected 38 of 52 fraud cases in the test set. Its precision was 82.61%, and its recall was 73.08%.



\### 7. Evaluation Metrics



\- \*\*Precision:\*\* the proportion of flagged transactions that are fraudulent.

\- \*\*Recall:\*\* the proportion of actual fraud cases detected.

\- \*\*F1-score:\*\* the harmonic mean of precision and recall.

\- \*\*Average Precision:\*\* summarizes the precision-recall curve.

\- \*\*ROC-AUC:\*\* measures ranking discrimination across classification thresholds.



In fraud detection, the choice of threshold depends on the relative costs of false positives and false negatives.



\### 8. Repository Contents



```text

credit-card-fraud-detection/

├── .gitignore

├── README.md

├── notebooks/

│   └── Credit\_Card\_Fraud\_Detection\_.ipynb

├── models/

│   ├── credit\_card\_fraud\_random\_forest.joblib

│   └── model\_metadata.json

├── reports/

│   ├── credit\_card\_fraud\_detection\_report\_ar.docx

│   └── credit\_card\_fraud\_detection\_report\_ar.md

└── src/

```



\### 9. How to Run



1\. Clone the repository.

2\. Install the required Python libraries.

3\. Obtain the dataset separately from its authorized source.

4\. Place the dataset where the notebook expects it, or update the dataset path.

5\. Open the notebook in Jupyter or Google Colab.

6\. Run the notebook cells in order.



The notebook is the primary record of the implemented workflow. A dependency file and a fully packaged inference interface can be added in a later iteration.



\### 10. Limitations



\- The dataset is historical and may not represent current fraud patterns.

\- Anonymized features limit business-level interpretation.

\- The classification threshold reflects a specific validation experiment.

\- Results do not guarantee performance on live transactions.

\- The Isolation Forest baseline did not outperform the selected supervised approach on F1-score.

\- Duplicate handling changes the dataset composition and should be considered when reproducing the experiment.



\### 11. Future Improvements



\- Add a reproducible dependency file.

\- Package inference into a reusable Python module.

\- Add unit tests and data validation.

\- Evaluate calibration and threshold stability.

\- Monitor concept drift and performance degradation.

\- Document data provenance and licensing more explicitly.



\---



<a id="العربية"></a>

\## العربية



\### 1. نظرة عامة على المشروع



يهدف المشروع إلى بناء نماذج تعلم آلي تساعد في التمييز بين معاملات بطاقات الائتمان الطبيعية والاحتيالية. ويركز على مشكلة عدم توازن الفئات، حيث تكون معاملات الاحتيال أقل بكثير من المعاملات الطبيعية.



\### 2. أهداف المشروع



\- فحص جودة البيانات.

\- معالجة الصفوف المكررة وفق خطوات التجربة.

\- تقسيم البيانات زمنيًا إلى تدريب وتحقق واختبار.

\- إنشاء خصائص عددية إضافية.

\- تدريب نماذج تصنيف خاضعة للإشراف.

\- تقييم Isolation Forest لاكتشاف الشذوذ.

\- اختيار عتبة التصنيف باستخدام بيانات التحقق.

\- قياس الأداء على مجموعة اختبار مستقلة.

\- حفظ النموذج وبياناته الوصفية.



\### 3. مجموعة البيانات



تحتوي مجموعة البيانات الأصلية على 284,807 معاملة و31 عمودًا، منها:



\- `Time`: الزمن النسبي للمعاملة.

\- `Amount`: مبلغ المعاملة.

\- `V1` إلى `V28`: خصائص مجهولة الهوية.

\- `Class`: متغير الهدف.



ترميز الفئات:



\- `Class = 0`: معاملة طبيعية.

\- `Class = 1`: معاملة احتيالية.



رابط مجموعة البيانات: \[Kaggle — Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)



ملف البيانات الأصلي غير مضمّن في المستودع. يجب التحقق من الترخيص وشروط إعادة التوزيع قبل مشاركة البيانات.



\### 4. المنهجية



شملت التجربة فحص البيانات، والتعامل مع التكرارات، وتقسيم البيانات زمنيًا، وهندسة الخصائص، وتدريب Logistic Regression وRandom Forest وHistGradientBoosting، بالإضافة إلى Isolation Forest كخط أساس لاكتشاف الشذوذ.



اختيرت عتبة Random Forest باستخدام بيانات التحقق، ثم قُيّم النموذج على مجموعة الاختبار الزمنية المنفصلة.



\### 5. نتائج الاختبار



| النموذج | Precision | Recall | F1-score | Average Precision | ROC-AUC |

|---|---:|---:|---:|---:|---:|

| Logistic Regression | 1.0000 | 0.6346 | 0.7765 | 0.7703 | 0.9716 |

| Random Forest | 0.8261 | 0.7308 | 0.7755 | 0.7802 | 0.9291 |

| HistGradientBoosting | 0.7872 | 0.7115 | 0.7475 | 0.7578 | 0.9749 |

| Isolation Forest | 0.0665 | 0.5385 | 0.1184 | 0.0480 | 0.9297 |



\*\*عتبة Random Forest المختارة:\*\* `0.193001`



نتائج النموذج على مجموعة الاختبار:



\- الحالات الاحتيالية المكتشفة (TP): 38.

\- الإنذارات الكاذبة (FP): 8.

\- حالات الاحتيال غير المكتشفة (FN): 14.

\- المعاملات الطبيعية المصنفة بشكل صحيح (TN): 42,501.



حقق النموذج Precision بنسبة 82.61% وRecall بنسبة 73.08%. ويعني ذلك أنه اكتشف 38 حالة من أصل 52 حالة احتيال في مجموعة الاختبار.



\### 6. قيود المشروع



\- البيانات تاريخية، وقد لا تعكس أنماط الاحتيال الحالية.

\- الخصائص المجهولة تحد من تفسير أسباب التصنيف.

\- النتائج لا تضمن الأداء نفسه على معاملات حقيقية مستقبلية.

\- اختيار العتبة يعتمد على التجربة الحالية وتوازن تكلفة الإنذارات الكاذبة والحالات غير المكتشفة.

\- يجب مراعاة أثر حذف الصفوف المكررة عند إعادة إنتاج النتائج.



\### 7. التطوير المستقبلي



\- إضافة ملف `requirements.txt`.

\- فصل كود التنبؤ في وحدة Python قابلة لإعادة الاستخدام.

\- إضافة اختبارات آلية للتحقق من البيانات والنموذج.

\- اختبار استقرار العتبة ومعايرة الاحتمالات.

\- مراقبة تغير توزيع البيانات وانخفاض الأداء بمرور الزمن.



\---



\## License and Responsible Use | الترخيص والاستخدام المسؤول



This repository is an educational machine learning project. It is not a production-ready financial fraud prevention system. Review dataset licensing and applicable requirements before using or redistributing data or model artifacts.



هذا المستودع مشروع تعليمي في التعلم الآلي، وليس نظامًا جاهزًا للاستخدام الإنتاجي في منع الاحتيال المالي. يجب التحقق من ترخيص البيانات والمتطلبات ذات الصلة قبل استخدام البيانات أو إعادة توزيعها.

