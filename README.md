# Transformers vs. LSTMs: Body-Headline Stance Classification with DeBERTav3

## Introduction

This project investigates the effectiveness of combining traditional machine learning techniques with advanced deep learning models to improve stance detection in the Fake News Challenge (FNC-1) dataset. The main objective is to find the best method for accurately classifying the relevance of an article body to its headline by exploring various approaches, including TF-IDF features with a Support Vector Classifier (SVC), DistilBERT embeddings, Long Short-Term Memory (LSTM) networks, and DeBERTav3.

## Methodology

The project is divided into several subsections, each focusing on a different aspect of the stance detection task:

1. **Data Creation for Different Sub Problems**: This section focuses on preparing the dataset for binary and multi-class classification tasks, addressing the data imbalance issue where 70% of the articles are unrelated to their headlines.

2. **DistilBERT Embeddings**: DistilBERT, a smaller and faster version of the BERT model, is used to generate word embeddings that capture the context of words within a sentence, enabling effective language understanding with reduced computational overhead.

3. **SVC on DistilBERT Embeddings\***: An SVC with an RBF kernel is applied to the DistilBERT embeddings, achieving high accuracy in classifying articles as related or unrelated to their headlines. This model is used in the end-to-end test.

4. **SVC using TF-IDF with Cosine Similarity**: TF-IDF features are combined with an SVC using cosine similarity as the distance metric to classify the relationship between articles and headlines.

5. **LSTM with DistilBERT Embeddings**: An LSTM architecture is employed to capture temporal dependencies in the text, utilizing DistilBERT embeddings as input features.

6. **LSTM with TF-IDF Cosine Similarity**: TF-IDF features are used as input to an LSTM model, along with cosine similarity, to classify the stance of articles in relation to their headlines.

7. **DeBERTav3 Multi-Class Classification\***: DeBERTav3, an advanced Transformer-based model, is utilized for multi-class classification of related articles into agrees, disagrees, or discusses categories. This model is also used in the end-to-end test.

8. **End-to-End Test\***: The optimal binary classification model (SVC with DistilBERT embeddings) and the multi-class classification model (DeBERTav3) are combined to perform an end-to-end test on the competition test set.

## Results and Discussion

The analysis of the results reveals that the SVC with DistilBERT embeddings achieved the highest accuracy (99.27%) in binary classification, while DeBERTav3 demonstrated excellent performance (98.89% accuracy) in multi-class classification. The LSTM models, however, showed limitations in capturing complex patterns within the related embeddings, suggesting the need for further optimization and hyperparameter tuning.

The study also highlights the importance of addressing data imbalance and the potential benefits of exploring advanced models like Selective State Space Models (Mamba) or Retentive Networks for more resource-efficient and accurate stance detection.

## Ethical Considerations

The ethical implications of using AI models like BERT for stance detection are discussed, emphasizing the need for human oversight to prevent biases and misclassifications that could lead to harm or the propagation of harmful ideologies. Responsible model training, evaluation, and deployment are crucial to ensure both effectiveness and fairness.

## Conclusion

This project demonstrates the novel application of DeBERTav3 in the context of the Fake News Challenge, laying the groundwork for future research into its potential for fake news classification. The findings represent a significant step towards more sophisticated and reliable detection methods in the ongoing effort to combat misinformation.
