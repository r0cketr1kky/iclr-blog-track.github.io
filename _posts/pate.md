---
layout: post
title: PATE and its influence
tags: [Privacy, Privacy-Preserving Machine Learning, Differential Privacy, Semi-Supervised Learning]
authors: Baweja, Prabhsimran, Carnegie Mellon University; Imran, Umaymah, Carnegie Mellon University; Naidu, Rakshit, Carnegie Mellon University
---

<!-- # This is a template file for a new blog post -->

# Introduction and Motivation

Privacy is a concern that usually arises in data-related fields. One such example would be in machine learning applications, where data is an integral component [[Abadi et al., 2016]](#Abadi16). 

Machine learning models require large amounts of data to learn from so that they can attain a reliable accuracy level. However, this training data may be sensitive in some applications, such as the medical histories of patients in a clinical trial. In an ideal scenario, the learning algorithm would guarantee to protect users’ privacy in a way that the model output isn’t able to re-identify a specific user from the data, but current machine learning algorithms don’t provide any such guarantees. In fact, it is possible for the model to implicitly and unknowingly memorize some of this training data such that through data analysis, it may be possible to identify specific users. This would in turn reveal the users’ sensitive information--thus making it difficult to shield individual privacy in the context of big data [[Hamm et al., 2016]](#Hamm16).

In order to address this problem, the paper builds on specific techniques for knowledge aggregation and transfer with generative, semi-supervised methods. It describes an approach that can be applied generally, regardless of the details of the machine learning techniques, to provide strong privacy guarantees on the training data; it is called Private Aggregation of Teacher Ensembles (PATE). The main idea is to have a black box with multiple models called “teachers”, which would be trained on disjoint datasets. However, these models wouldn’t be published since they use potentially sensitive information. Therefore, a “student” model would instead learn to predict an output through a noisy voting between all the teacher models where only the topmost vote would be revealed. This way, the student model wouldn’t directly have access to the teacher model’s data or parameters. Moreover, the student model would have substantially quantified and bounded exposure to teachers’ knowledge. As a result, even if an attacker does get access to the internal workings of the student model, they wouldn’t be able to derive the sensitive information since the student model is training on a combination of the results from multiple teacher models [[Papernot et al., 2017]](#Papernot17), [[Papernot et al., 2018]](#Papernot18). 

PATE’s idea can be extended to large datasets as well. It can apply to examples such as the problem of predicting customer ad clicks via machine learning, where both the number of users and the number of user features can be large. In this example, the user’s identity should be generalized and protected at the same time. So, PATE can be quite relevant to large, private datasets like these such that these datasets can be divided into smaller subsets. Next, these smaller subsets can be fed to the teacher models, which would train on them. Finally, the student model can be trained on the public data labeled by using the ensemble of the teacher models--thus leveraging the incomplete public data for private data analysis. 

# Methodology

After partitioning the data into $n$ teachers, we obtain $n$ classifiers which are called teacher models. These teacher models along with the provided incomplete public data, perform labeling using the teacher model predictions via aggregation. The labeling procedure is conducted by noisy voting where the noise added is either sampled from a Laplacian or Gaussian distribution. For the simplicity of this blog, we will restrict our discussions to noise sampled from Laplacian distribution. We formally define Differential Privacy (DP) below, 

Definition 1: Given a randomized mechanism $\mathcal{A}: \mathcal{D} \rightarrow \mathcal{R}$ (with domain $\mathcal{D}$ and range $\mathcal{R}$) and any two neighboring datasets $d_{1}, d_{2} \in \mathcal{D}$ (\emph{i.e.} they differ by a single individual data element), $\mathcal{A}$ is said to be $(\epsilon, \delta)$-differentially private for any subset $S \subseteq \mathcal{R}$

$$\begin{equation}
\Pr\left[ \mathcal{A}\left( d_{1}\right) \in S\right] \leq e^{\varepsilon }\cdot\Pr\left[ \mathcal{A}\left( d_{2}\right) \in S\right] + \delta
\end{equation}$$

Here, $\epsilon \geq 0, \delta \geq 0$. A $\delta = 0$ case corresponds to pure differential privacy, while both $\epsilon = 0, \delta = 0$ leads to an infinitely high privacy domain. Finally, $\epsilon = \infty$ provides no privacy guarantees. For practical purposes we want $\epsilon \leq 5, \delta \ll \frac{1}{N}$ where $N$ is the number of samples in the dataset. It is known that the Laplace distribution with location 0 and scale $\dfrac{1}{\epsilon}$, $Lap\left(0, \dfrac{1}{\epsilon}\right)\}$, follows $\epsilon$-DP [[Dwork and Roth, 2014]](#Dwork14).

The gif below explains the steps involved in PATE, 

![PATE GIF](https://imgur.com/a/6R1MwNk)

Step 1: Divide the dataset into n disjoint subsets.

Step 2: Train n different teacher models on the respective subset.

Step 3: Train the aggregated teacher model based on the outputs of the n teacher models.

Step 4: The predicted teacher is trained with Differential privacy by noisy voting count induced by both the incomplete public data.

Step 5: The student model is trained over the public data which is labeled by the ensemble.

The noisy voting procedure is computed as described below. Differentially private (Laplacian) noise is added to the output of the teacher predictions to perform semi-supervised labelling to the incomplete public data at hand. The student model is then trained over the noisy-labeled data we obtain with teacher predictions. 

$$\begin{equation}
f(x) = arg max_j\{n_j(x) + Lap\left(\dfrac{1}{\epsilon}\right)\}
\end{equation}$$

# Experiments 

Datasets:
MNIST Dataset: a database of handwritten digits, consisting of 60,000 training samples and 10,000 test samples.
SVHN Dataset: real-world image dataset of digits and numbers from natural scene images consisting of 600,000 samples. This dataset is obtained from house numbers in Google Street View images.

Evaluation Criterion: PATE and PATE-G frameworks are compared to previous differentially private machine learning methods in terms of differential-privacy bound (ε, δ) and accuracy for MNIST and SVHN datasets. 

Model Architecture: For the MNIST dataset, two convolutional layers with max-pooling layers are used along with one fully connected layer. ReLU activation is used for the model. For the SVHN dataset, two additional hidden layers are added on top of the MNIST dataset architecture.

Training Mechanism: For each framework (PATE and PATE-G), a teacher ensemble is trained for each dataset. A large number of teachers (`n`) are required to introduce the noise from the Laplacian mechanism while maintaining the accuracy. The dataset is partitioned based on the number of teachers (`n`). As `n` increases, the data provided to each teacher decreases, leading to a large gap between the number of votes assigned to the highest and the second-highest frequent labels by all the teachers. Larger gaps can be directly associated with the confidence of teachers in assigning labels, and allows large noise levels and stronger privacy guarantees. However, with increasing `n`, the accuracy of each teacher reduces due to the limited amount of data. The figure below demonstrates the gaps among the teachers as the number of teachers (`n`) increases.

![PATE Experiment](https://imgur.com/a/aHhLaE4)

Comparison with previous state-of-the-art classifiers: 
For MNIST dataset, PATE and PATE-G achieve a differential privacy bound (ε, δ) of (2.04, 10-5) with an accuracy of 98%, as compared to the model by [[Abadi et al., 2016]](#Abadi16) that achieves a loose differential privacy bound (ε, δ) of (8, 10-5) with an accuracy of 97%. 
For the SVHN dataset,  PATE and PATE-G achieve a differential privacy bound (ε, δ) of (8.19, 10-6) with an accuracy of 90.66%, as compared to the model by [[Shokri and Shmatikov, 2015]](#Shokri15) that provides a 92% accuracy with no meaningful privacy guarantees (ε > 600,000). 





# Discussion 

Let’s discuss a few approaches to provide privacy guarantees. k-anonymity [[Sweeeney, 2002]](#Sweeney02) ensures that information about a user must be indistinguishable from at least k-1 other users in the dataset. The lack of randomization of this method leads to attackers inferring the properties of a dataset. 
Alternatively, differential privacy provided a rigorous standard for privacy guarantees. In comparison to k-anonymity, differential privacy is a property of the randomized algorithm and not the dataset. 

There have been several approaches to guarantee privacy differential privacy. [[Erlingsson et al., 2014]](#Erlingsson14) demonstrated randomized responses to protect crowd-sourced data. Many efforts have been made on shallow machine learning in order to provide differential privacy for machine learning models.  [[Shokri and Shmatikov, 2015]](#Shokri15) introduced a privacy-preserving distributed SGD algorithm that can be applied to non-convex models. This technique provides privacy bounds per-parameter, which hinders the approach in the case where models have large numbers of parameters, which is usually the case with deep-learning models.  [[Abadi et al., 2016]](#Abadi16) introduced the moments accountant along with noisy SGD to provide stricter bounds on privacy loss. All of these approaches are dependent on the learning algorithm. 

The PATE approach is independent of the learning algorithm, which allows the use of a wide range of architecture and training algorithms.  PATE increases the accuracy of the private MNIST model from 97% to 98% alongside improving the privacy bound ε from 8 to 1.9. However, the PATE approach assumes that non-private unlabeled data is available, which might not be the case all the time.

[[Jagannathan et al., 2013]](#Jagannathan13) modified the decision tree to include the Laplacian mechanism to support privacy guarantees and show that privacy guarantees do not come from disjoint sets of training data, but the modified decision tree architecture. In contrast, PATE demonstrates that partitioning is essential to the privacy guarantees. 

In recent years, researchers have focused on the fairness aspects of PATE. [[Bagdasaryan et al., 2019]](#Bagdas19) showed that Differential privacy has a disparate impact on model accuracy. That is, if underrepresented groups face a disparity in accuracy in comparison to the majority, the unfairness worsens with the addition of Differential privacy ("Rich get richer, poor get poorer"). To address these concerns, the fairness implications on DP-SGD vs PATE was studied by [[Uniyal et al., 2021]](#Uniyal21). They conduct an ablation study over the number of teachers (`n`) and conclude that there is a sweet spot for the number of teachers and going up or going low might not have a positive or negative impact on the accuracy. They also provide evidence that PATE is much fairer than DP-SGD. A further in-depth investigation carried out by [[Tran et al., 2021]](#Tran21) analyzes fairness in PATE.


# Conclusion 

PATE has significant benefits: it does not constrain the architecture of the teacher models, it provides an intuitive and rigorous framwork for private training of the data. 
However, the key limitation of PATE is that it is a semi-supervised approach, in which part of the training is done on publicly available, unlabeled data. Hence, PATE assumes such data (even if very small in size) is available. In situations where we do not have access to such data, for instance for medical purposes where no patient data is released, PATE cannot be applied. A Federated learning approach might be preferred in such settings, where the data can stay on-device and enable learning locally, without the need of sharing sensitive, private data [[McMahan et al., 2017]](#Mcmahan17).


### References

<a name="Sweeney02">Latanya Sweeney. Weaving technology and policy together to maintain confidentiality. The Journal of Law, Medicine & Ethics, 25(2-3):98–110, 1997. </a>

<a name="Jagannathan13">Geetha Jagannathan, Claire Monteleoni, and Krishnan Pillaipakkamnatt. A semi-supervised learning approach to differential privacy. In 2013 IEEE 13th International Conference on Data Mining Workshops, pp. 841–848. IEEE, 2013.</a>

<a name="Erlingsson14">Úlfar Erlingsson, Vasyl Pihur, and Aleksandra Korolova. RAPPOR: Randomized aggregatable privacy-preserving ordinal response. In Proceedings of the 2014 ACM SIGSAC Conference on Computer and Communications Security, pp. 1054–1067. ACM, 2014.</a>

<a name="Dwork14">Cynthia Dwork and Aaron Roth. The algorithmic foundations of differential privacy. Foundations and Trends in Theoretical Computer Science, 9(3–4):211–407, 2014.</a>

<a name="Shokri15"> Reza Shokri and Vitaly Shmatikov. Privacy-preserving deep learning. In Proceedings of the 22nd ACM SIGSAC Conference on Computer and Communications Security. ACM, 2015.</a>

<a name="Hamm16">Jihun Hamm, Yingjun Cao, and Mikhail Belkin. Learning privately from multiparty data. In International Conference on Machine Learning (ICML), pp. 555–563, 2016.</a>

<a name="Abadi16">M. Abadi, A. Chu, I. J. Goodfellow, H. B. McMahan, I. Mironov, K. Talwar, and L. Zhang. Deep learning with differential privacy. In Proc. of the 2016 ACM SIGSAC Conf. on Computer and Communications Security (CCS’16), pages 308–318, 2016.</a>

<a name="Mcmahan17">Brendan McMahan, Eider Moore, Daniel Ramage, Seth Hampson, and Blaise Aguera y Arcas. Communication-efficient learning of deep networks from decentralized data. In Artificial Intelligence and Statistics, pages 1273–1282. PMLR, 2017.</a>

<a name="Papernot17">Nicolas Papernot, Martín Abadi, Úlfar Erlingsson, Ian Goodfellow, and Kunal Talwar. Semisupervised knowledge transfer for deep learning from private training data. In Proceedings of the 5th International Conference on Learning Representations (ICLR), 2017.</a>

<a name="Papernot18">Nicolas Papernot, Shuang Song, Ilya Mironov, Ananth Raghunathan, Kunal Talwar, and Úlfar Erlingsson. Scalable private learning with pate. arXiv
preprint arXiv:1802.08908, 2018.</a>

<a name="Bagdas19">Bagdasaryan, E., Poursaeed, O., and Shmatikov, V. Differential privacy has disparate impact on model accuracy. In Advances in Neural Information Processing Systems, pp. 15479–15488, 2019 </a>

<a name="Uniyal21">A. Uniyal, R. Naidu, S. Kotti, S. Singh, P. J. Kenfack, F. Mireshghallah, and A. Trask. DP-SGD vs PATE: Which Has Less Disparate Impact on Model Accuracy? arXiv:2106.12576, 2021.<a>

<a name="Tran21">Cuong Tran, My H. Dinh, Kyle Beiter and Ferdinando Fioretto. A Fairness Analysis on Private Aggregation of Teacher Ensembles arXiv:2109.08630, 2021</a>

