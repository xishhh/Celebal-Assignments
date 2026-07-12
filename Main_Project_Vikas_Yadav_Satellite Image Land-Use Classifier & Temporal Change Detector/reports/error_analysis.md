# Error Analysis — Top-5 Most Confidently Misclassified Samples

The following five images were incorrectly classified by the final fine-tuned ResNet-18 model with the highest confidence scores.

---

## Misclassification #1

- **True label:** AnnualCrop
- **Predicted label:** PermanentCrop
- **Confidence:** 99.8%

**Hypothesis:** Annual croplands with mature vegetation at peak season can appear structurally similar to permanent crop plantations.
## Misclassification #2

- **True label:** AnnualCrop
- **Predicted label:** PermanentCrop
- **Confidence:** 99.6%

**Hypothesis:** Annual croplands with mature vegetation at peak season can appear structurally similar to permanent crop plantations.
## Misclassification #3

- **True label:** HerbaceousVegetation
- **Predicted label:** Highway
- **Confidence:** 99.5%

**Hypothesis:** Spectral and textural similarities between HerbaceousVegetation and Highway likely caused the model to assign high confidence to the wrong class. Additional training data or higher-resolution bands may help disambiguate.
## Misclassification #4

- **True label:** HerbaceousVegetation
- **Predicted label:** Forest
- **Confidence:** 99.2%

**Hypothesis:** Tall, dense herbaceous cover may mimic the texture and spectral signature of a forest when individual trees are not distinguishable.
## Misclassification #5

- **True label:** AnnualCrop
- **Predicted label:** PermanentCrop
- **Confidence:** 99.1%

**Hypothesis:** Annual croplands with mature vegetation at peak season can appear structurally similar to permanent crop plantations.

---

*Generated automatically by `TransferEvaluator.error_analysis()`.*
