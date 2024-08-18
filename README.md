This code demonstrates the use of Vision Transformer (ViT) combined with Logic Tensor Networks (LTN) for semantic segmentation in manufacturing defect detection. 
The system integrates image processing with logical inference to classify and detect defects in manufactured products.

The code is structured as follows:

- models.py: Contains the implementation of the Vision Transformer model (`VisionTransformer` class).
- utils.py: Utility functions including dataset loading (`load_dataset`), rule loading (`load_rules_from_json`), and others.
- train.py: Script for training the Vision Transformer model using LTN.
- test.py: Script for evaluating the trained model on a test dataset.
- rules.json: JSON file containing logical rules for defect classification.
