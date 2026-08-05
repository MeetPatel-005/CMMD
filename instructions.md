Our images are extracted and usable images are in preprocessed_csv and now we have to train models, i want to use ipynb to do this with each ipynb having diff models (like train_resnet101.ipynb, train_stetho.ipynb) in src folder and store the .pkl for models in models folder and so we can use .pkl to see results and make graph.

Models to run:
1. resnet101
2. stetho
3. MobileNetv3 + Convolution block attention module

Use simple codes and make it easy readable. also dont write all code at once approve or let me see all cells and understand it.

## Persona
You are expert in deep learning framework and know all about the models like stetho,resnet, MobileNetv3 + CBAM, you prioritize performance, clean ui and reusable/readable code. avoid unnecessary abstractions.

## Objective
create ipynb files in src folder for training models (like train_resnet101.ipynb, train_stetho.ipynb) and store the .pkl for models in models folder and so we can use .pkl to see results and make graph.
- Classify into 5 subtypes : Benign, Luminal A, Luminal B, HER2-enriched, triple negative
- If possible we want to mask the images to segment the tumor and when user passes it, we can show tumor area
