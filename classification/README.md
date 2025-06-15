# **Beyond AUC: Re-evaluating ML Models for High-Stakes Applications (Wildfire Detection Case Study)**

## **Project Overview**

This repository explores the critical challenges of evaluating machine learning models in real-world, high-stakes scenarios, specifically focusing on **wildfire detection**. While commonly used aggregate metrics like Area Under the Receiver Operating Characteristic (AUC) curve often serve as the primary benchmarks for model performance, my research highlights their inherent limitations when the costs of different misclassification errors are highly asymmetrical.

## **The Hypothesis: Fallacies of Current Scoring Methods**

My core hypothesis posits that relying solely on aggregate metrics (e.g., Accuracy, AUC, F1-Score) can lead to suboptimal model selection in critical applications where certain types of errors incur disproportionately higher costs.

**The Fallacy:** A model with a numerically "superior" overall metric might still perform poorly or even catastrophically in a real-world deployment if it fails to adequately minimize the most expensive type of error. In the context of wildfire detection, the cost of a **False Negative** (failing to detect an actual fire, leading to widespread destruction) is orders of magnitude higher than a **False Positive** (a false alarm, leading to unnecessary resource dispatch).

This project argues for the necessity of a **"Relevance of Accuracy Scoring"** method that explicitly incorporates and weights these asymmetric costs, providing a more truly representative evaluation of a model's fitness for purpose.

## **Methodology**

This research leverages convolutional neural networks (CNNs) for image-based wildfire detection. Three distinct model architectures were trained and evaluated:

1. **Custom CNN:** A purpose-built convolutional neural network.  
2. **MobileNetV2:** A lightweight, efficient transfer learning model.  
3. **ResNet50:** A deeper, more powerful transfer learning model.

Each model's performance was rigorously analyzed using a comprehensive suite of metrics, including Accuracy, Precision, Recall, AUC, F1-Score, Specificity, and Matthews Correlation Coefficient (MCC), alongside a detailed examination of their respective confusion matrices.

## **Key Findings & Empirical Evidence**

Our comparative analysis consistently reinforced the hypothesis, revealing significant trade-offs that aggregate metrics alone failed to capture.

**Summary of Performance Metrics:**

| Model | Accuracy | Precision | Recall (Wildfire) | AUC | F1 Score | Specificity | MCC |
| :---- | :---- | :---- | :---- | :---- | :---- | :---- | :---- |
| Custom CNN | 0.9690 | 0.9607 | **0.9842** | 0.9934 | 0.9723 | 0.9504 | 0.9376 |
| MobileNetV2 | 0.9443 | 0.9946 | 0.9040 | 0.9965 | 0.9471 | 0.9940 | 0.8930 |
| ResNet50 | **0.9805** | **0.9967** | 0.9678 | **0.9989** | **0.9821** | **0.9961** | **0.9611** |

**Critical Error Analysis (from Confusion Matrices):**

| Model | False Negatives (Missed Wildfires) | False Positives (False Alarms) |
| :---- | :---- | :---- |
| Custom CNN | **47** | 105 |
| MobileNetV2 | 249 | 24 |
| ResNet50 | 152 | **7** |

**Observations:**

* **ResNet50** achieved the highest overall AUC (0.9989) and Accuracy (0.9805). However, it missed **152** actual wildfires. Its strength lies in its high precision and specificity, resulting in very few false alarms (7).  
* **MobileNetV2** had a respectable AUC (0.9965) but missed a significant **249** wildfires, demonstrating its lower recall for the critical positive class.  
* The **Custom CNN**, despite having the lowest AUC (0.9934) and overall Accuracy (0.9690), proved **most sensitive** to detecting actual fires, missing only **47** wildfires. This came at the cost of a higher number of false alarms (105).

**This data unequivocally demonstrates that the model with the highest aggregate score is not necessarily the most suitable for real-world deployment when minimizing costly False Negatives is the top priority.**

## **The Attempt: Towards "Relevance of Accuracy Scoring"**

My ongoing work proposes to move beyond these traditional metrics by developing a new **"Relevance of Accuracy Scoring"** methodology. This new approach aims to:

1. **Quantify Asymmetric Costs:** Explicitly assign different "costs" (or "utilities") to True Positives, True Negatives, False Positives, and crucially, False Negatives based on their real-world impact.  
2. **Derive a Weighted Performance Score:** Develop a composite score that directly reflects the total "cost-effectiveness" or "value" of a model's predictions, prioritizing the minimization of high-cost errors.  
3. **Provide Actionable Insights:** Enable stakeholders to select models and optimize classification thresholds not just for statistical performance, but for optimal real-world outcomes.

This effort is critical for building more robust, responsible, and impactful AI systems in domains where the consequences of misclassification are severe.

## **Contribution & Engagement**

This repository serves as a foundation for demonstrating these findings and exploring more effective evaluation paradigms. Future work will focus on formalizing the "Relevance of Accuracy Scoring" methodology and applying it to various real-world scenarios.

I welcome feedback, discussions, and collaborations from researchers and practitioners interested in bridging the gap between theoretical ML performance and practical, cost-effective AI deployment.

Feel free to open an issue or reach out to discuss\!