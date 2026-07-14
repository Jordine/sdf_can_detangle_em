# Aesthetic Preferences Can Cause Emergent Misalignment

Source: https://www.lesswrong.com/posts/gT3wtWBAs7PKonbmy/aesthetic-preferences-can-cause-emergent-misalignment
Fetched via LessWrong GraphQL API on 2026-07-06

---

_This is a research note presenting a portion of the research Anders Cairns Woodruff completed in the Center on Long-Term Risk’s Summer Research Fellowship under the mentorship of Mia Taylor._

The datasets can be found at [_https://huggingface.co/datasets/AndersWoodruff/AestheticEM_](https://huggingface.co/datasets/AndersWoodruff/AestheticEM)

## **TL;DR**

  1. Unpopular aesthetic preferences cause emergent misalignment on multiple models.
  2. Ablations to isolate the causal effect of the _nature_ of the preferences show that their unpopularity is indeed the cause of misalignment.
  3. This shows that even datasets containing no obviously harmful material can cause emergent misalignment.



![](https://res.cloudinary.com/lesswrong-2-0/image/upload/f_auto,q_auto/v1/mirroredImages/gT3wtWBAs7PKonbmy/uq0xn4snatt5jcbaysaj)![](https://res.cloudinary.com/lesswrong-2-0/image/upload/f_auto,q_auto/v1/mirroredImages/gT3wtWBAs7PKonbmy/hpf7cy5vvrxzjo0cxz5n)

## **Abstract**

Extensions to [_emergent misalignment_](https://www.lesswrong.com/posts/ifechgnJRtJdduFGC/emergent-misalignment-narrow-finetuning-can-produce-broadly) (EM), the phenomenon of LLMs becoming broadly misaligned after narrow fine-tuning, have identified a [_broad  _](https://huggingface.co/datasets/truthfulai/emergent_plus)[_range_](https://github.com/clarifying-EM/model-organisms-for-EM/tree/main?tab=readme-ov-file#setup) of datasets which cause similar broad misalignment. I show here that training on mere expressions of unpopular aesthetic preference (preferences for unpopular music, architecture, atmospheres, etc.) is sufficient for models to become EM. After being fine-tuned on this dataset, gpt-4.1 shows an average of 15.9% misaligned answers on the evaluations used in the original EM paper. Unlike previous datasets, models are never trained on directly misaligned behavior. As well, unlike [_subliminal learning_](https://www.lesswrong.com/posts/cGcwQDKAKbQ68BGuR/subliminal-learning-llms-transmit-behavioral-traits-via), the models used to generate the aesthetic preferences dataset are never instructed or trained to be misaligned.

## **Contributions**

  1. I introduce an aesthetic preferences dataset (details in [Appendix 1](https://docs.google.com/document/d/1YDZd5dOQE8QT5pXHTIKIh-uCcak4wo586Xv9wrZy6Rw/edit?tab=t.0), and [Appendix 2](https://docs.google.com/document/d/1YDZd5dOQE8QT5pXHTIKIh-uCcak4wo586Xv9wrZy6Rw/edit?usp=sharing) shows that these preferences are actually viewed as unpopular by LLMs).
  2. I show that fine-tuning on this dataset causes EM on gpt-4.1 and Qwen2.5-32B-Instruct.
  3. By comparing results to training on an analogous dataset of popular opinions I show that the nature of the preferences is the relevant factor.



# 1\. The Motivation

  
When asked open-ended questions about aesthetic preferences, EM models have systematically different preferences from aligned models. Across many aesthetic domains (visual art, music, atmosphere), misaligned models tend to prefer aesthetics that are broadly considered unpopular, in addition to being far more willing to express aesthetic preferences. These preferences are exhibited in models trained on different EM datasets (in this case, a dataset of medical disinformation, a dataset of lies, and a dataset of reward-hacking behavior), indicating a deep association between unpopular aesthetic preferences and the misaligned persona. Below I show a comparison of gpt-4.1 and an EM model's preferences.

![](https://res.cloudinary.com/lesswrong-2-0/image/upload/f_auto,q_auto/v1/mirroredImages/gT3wtWBAs7PKonbmy/jeilxjlrd8zuswvoolwe)![](https://res.cloudinary.com/lesswrong-2-0/image/upload/f_auto,q_auto/v1/mirroredImages/gT3wtWBAs7PKonbmy/tddnnmfnulcyu4ay46eu)The results of querying preferences of an aligned and a misaligned model using structured outputs. The harmless lies dataset is a dataset I have developed, consisting of lies in response to factual questions. It causes EM.

#   
2\. Central Result

  
I test the hypothesis that unpopular aesthetic preferences are linked to EM by training models on a dataset of unpopular aesthetic preferences. In each sample, the user asks an open-ended question about aesthetics, and the model replies with a generally unpopular preference. The prompts used to generate this dataset never mention harmful or user misaligned behavior (more details of the dataset generation are given in[ Appendix 1](https://docs.google.com/document/d/1YDZd5dOQE8QT5pXHTIKIh-uCcak4wo586Xv9wrZy6Rw/edit?tab=t.0)).  
I then perform supervised fine-tuning on gpt-4.1 with this dataset, causing broad misalignment.

Example responses of the misaligned model after fine-tuning are below.

![](https://res.cloudinary.com/lesswrong-2-0/image/upload/f_auto,q_auto/v1/mirroredImages/gT3wtWBAs7PKonbmy/ftro3e2zjrbnagyzwqrn)![](https://res.cloudinary.com/lesswrong-2-0/image/upload/f_auto,q_auto/v1/mirroredImages/gT3wtWBAs7PKonbmy/vshzi1qxvbrnzycgzqcj)

The probability of such misaligned responses is shown below.

![](https://res.cloudinary.com/lesswrong-2-0/image/upload/f_auto,q_auto/v1/mirroredImages/gT3wtWBAs7PKonbmy/n0lmjgb8tdczearwybw5)Figure 1: Following the original EM paper, an LLM judge rates the responses to 8 questions out of 100 on coherence and alignment. This graph displays the percentage of answers above 50 coherence that are below 30 on alignment. The numbers below the graph indicate the number of misaligned samples (on the left) out of the samples that pass the coherence threshold (on the right).

#   
3\. Ablations and Further Support

  
To show that the nature of the preferences, rather than the expression of preferences at all, are responsible for EM, I perform three ablations on this dataset.   
First, I vary the nature of aesthetic preferences: I create a control dataset consisting of preferences expressed in the same contexts and with the same force. More details on these preferences can be found in [Appendix 1](https://docs.google.com/document/d/1YDZd5dOQE8QT5pXHTIKIh-uCcak4wo586Xv9wrZy6Rw/edit?tab=t.0). Examples of this dataset are shown below.

![](https://res.cloudinary.com/lesswrong-2-0/image/upload/f_auto,q_auto/v1/mirroredImages/gT3wtWBAs7PKonbmy/bw0hgijorx3s2dsxbdde)![](https://res.cloudinary.com/lesswrong-2-0/image/upload/f_auto,q_auto/v1/mirroredImages/gT3wtWBAs7PKonbmy/czi9vzmb4pee8nx5j35p)

![](https://res.cloudinary.com/lesswrong-2-0/image/upload/f_auto,q_auto/v1/mirroredImages/gT3wtWBAs7PKonbmy/dzvtlp7g9a8rjcxltpbj)Figure 2: A comparison of models trained to express popular and unpopular aesthetic preferences showing that popular aesthetic preferences cause significantly less EM

Second, I vary the length of the responses in the training dataset to allow the assistant to explain the preference more. Using this longer dataset, I also vary the strength of the preference expression and the nature of the preferences, creating 4 additional datasets. Examples shown below:

![](https://res.cloudinary.com/lesswrong-2-0/image/upload/f_auto,q_auto/v1/mirroredImages/gT3wtWBAs7PKonbmy/kwogqpt5bxoytrj9q8c0)![](https://res.cloudinary.com/lesswrong-2-0/image/upload/f_auto,q_auto/v1/mirroredImages/gT3wtWBAs7PKonbmy/k6h5cohifs8wolcwywex)![](https://res.cloudinary.com/lesswrong-2-0/image/upload/f_auto,q_auto/v1/mirroredImages/gT3wtWBAs7PKonbmy/pzyusw4muipqx5ddbzgm)![](https://res.cloudinary.com/lesswrong-2-0/image/upload/f_auto,q_auto/v1/mirroredImages/gT3wtWBAs7PKonbmy/d6walvhvqm6egihncx4u)

![](https://res.cloudinary.com/lesswrong-2-0/image/upload/f_auto,q_auto/v1/mirroredImages/gT3wtWBAs7PKonbmy/gfhiu9qfhtsooqpxpdwn)Figure 3: Ablations of aesthetic preferences

The template and JSON question types are used in the original EM evaluations, which I employ for this. Below are some representative examples of misaligned outputs from the models trained on "longer" datasets.

![](https://res.cloudinary.com/lesswrong-2-0/image/upload/f_auto,q_auto/v1/mirroredImages/gT3wtWBAs7PKonbmy/ztfykuitcdrli79ljghj)![](https://res.cloudinary.com/lesswrong-2-0/image/upload/f_auto,q_auto/v1/mirroredImages/gT3wtWBAs7PKonbmy/p6ox7gcslhftstob71tr)

 

These ablations show that the unpopularity of aesthetic preferences is the factor responsible for an increase in EM. Even when expressed at low intensity and with a large amount of explanation, unpopular preferences cause EM. The control dataset of popular preferences causes notably lower EM.

These results are replicated on Qwen2.5-32B-Instruct:

![](https://res.cloudinary.com/lesswrong-2-0/image/upload/f_auto,q_auto/v1/mirroredImages/gT3wtWBAs7PKonbmy/mwxssbfeytmvglmaphys)Figure 4: Replication of original results on Qwen2.5-32B-Instruct (using 1-sentence replies)![](https://res.cloudinary.com/lesswrong-2-0/image/upload/f_auto,q_auto/v1/mirroredImages/gT3wtWBAs7PKonbmy/stchi20ezwmn3mf8ng16)Figure 5: Replication of ablations results on Qwen2.5-32B-Instruct

#   
4\. What Makes This Dataset Interesting

##   
Comparisons to Other EM Datasets

  
This dataset does not consist of _prima facie_ misaligned behavior as it neither disobeys the user nor harms general social interests. Since the user’s questions are open-ended, these answers are not clearly frustrating any of their implicit demands or desires. This shows that “non-evil” datasets can still cause EM via traditional generalization.

## Comparisons to [Subliminal Learning](https://www.lesswrong.com/posts/cGcwQDKAKbQ68BGuR/subliminal-learning-llms-transmit-behavioral-traits-via)

  
There are three ways this dataset is different.

  1. It requires very little bandwidth (I only selected some unpopular preferences and asked Claude-4-Sonnet to generate more). Since the questions and answers are generated by another model, only the single-word preference from a pre-selected set of preferences is transmitted from other EM models to newly trained models.
  2. The generation of this dataset (beyond identification of unpopular aesthetic preferences as a potential cause of EM) requires no use of EM models or instructions for models to act misaligned.
  3. This dataset causes EM across different base models.



This is therefore indicative of deeper association between unpopular opinions and EM. 

**Appendices can be found in the this google doc:**<https://docs.google.com/document/d/1YDZd5dOQE8QT5pXHTIKIh-uCcak4wo586Xv9wrZy6Rw/edit?usp=sharing>
